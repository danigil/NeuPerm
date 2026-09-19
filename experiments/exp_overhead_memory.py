"""
Overhead experiment (memory) — NeuPerm.

Companion to ``exp_overhead.py`` / ``exp_overhead_baselines.py``. Measures the
peak *additional* resident-set memory (RSS) that each sanitization method
allocates while it runs. ``tracemalloc`` cannot see this: it only tracks the
Python allocator, so torch tensor storage (and NeuPerm's in-place index swaps)
read as ~0. We therefore sample the process RSS via ``psutil`` from a background
thread and report peak(RSS during op) - RSS(before op).

Methods (same settings as the wall-clock tables):
  - neuperm : permute_model, in place
  - quant8  : 8-bit per-channel quantize-dequantize, BN skipped, in place
  - noise   : additive Gaussian noise, sigma = 1e-4, in place
  - prune   : 20% random unstructured pruning, made permanent

Every method operates *in place on a fresh copy* of the weights; that copy is
made in untimed/un-sampled setup, so the reported delta is the scratch memory
the operation itself allocates (mirroring the wall-clock tables, which time the
in-place op and exclude the one-time copy). NeuPerm reuses storage and should
show ~0; the baselines allocate transient buffers / masks / output tensors.

Usage:
    cd /path/to/NeuPerm
    PYTHONPATH=.:experiments python experiments/exp_overhead_memory.py

Output: results/overhead_memory.csv
"""

import copy
import csv
import gc
import os
import threading
import time
import traceback

import psutil
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from neu_perm.config import RESULTS_DIR
from neu_perm.perm import permute_model
from neu_perm.quantization import quantize_dequantize_sd, get_bn_keys_from_sd

from exp_overhead import MODELS, load_model  # noqa: E402

SEED = 42
N_REPEATS = 5
NOISE_SIGMA = 1e-4
PRUNE_AMOUNT = 0.2
QUANT_BITS = 8
SAMPLE_INTERVAL = 2e-4  # 0.2 ms RSS polling

METHODS = os.environ.get(
    "MEM_METHODS", "neuperm,quant8,noise,prune").split(",")
OUT_CSV = os.path.join(RESULTS_DIR, os.environ.get(
    "MEM_OUT", "overhead_memory.csv"))

_PROC = psutil.Process()
_MB = 1024 ** 2


class _RSSSampler(threading.Thread):
    """Poll process RSS and hold the maximum seen while running."""

    def __init__(self, interval=SAMPLE_INTERVAL):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_evt = threading.Event()
        self.peak = _PROC.memory_info().rss

    def run(self):
        while not self._stop_evt.is_set():
            r = _PROC.memory_info().rss
            if r > self.peak:
                self.peak = r
            time.sleep(self.interval)

    def stop(self):
        self._stop_evt.set()


def _apply_noise(work, sigma):
    for v in work.values():
        if v.is_floating_point():
            noise = torch.randn(v.shape, dtype=torch.float32).mul_(sigma)
            v.add_(noise.to(v.dtype))


def _prune(model_copy, amount):
    for _, module in model_copy.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.random_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model_copy


def measure_once(method, sd, model):
    """Return peak additional RSS (MB) of one in-place run of *method*."""
    # --- untimed / un-sampled setup: fresh working copy ---
    if method == "prune":
        payload = copy.deepcopy(model)
    elif method == "quant8":
        payload = ({k: v.clone() for k, v in sd.items()},
                   get_bn_keys_from_sd(sd))
    elif method in ("noise", "neuperm"):
        payload = {k: v.clone() for k, v in sd.items()}
    else:
        raise ValueError(method)

    gc.collect()
    base = _PROC.memory_info().rss
    sampler = _RSSSampler()
    sampler.peak = base
    sampler.start()
    try:
        if method == "neuperm":
            permute_model(_model_key(model, sd), payload, inplace=True)
        elif method == "quant8":
            work, bn = payload
            quantize_dequantize_sd(work, n_bits=QUANT_BITS, per_channel=True,
                                   inplace=True, skip_keys=bn)
        elif method == "noise":
            _apply_noise(payload, NOISE_SIGMA)
        elif method == "prune":
            _prune(payload, PRUNE_AMOUNT)
    finally:
        sampler.stop()
        sampler.join()

    peak_delta = max(0.0, (sampler.peak - base) / _MB)
    del payload
    return {"peak_rss_delta_mb": peak_delta, "base_rss_mb": base / _MB}


# permute_model is keyed by the architecture name; we thread it through.
_CUR_NAME = {"name": None}


def _model_key(model, sd):
    return _CUR_NAME["name"]


def worker(name, method):
    """One clean measurement in a fresh process. Prints a RESULT line.

    Process isolation is essential: glibc malloc returns freed memory to its
    own pool, not the OS, so a second op in the same process sees RSS already
    at the prior high-water mark and reports a spurious ~0 delta. Each
    (model, method) is therefore measured in its own subprocess, right after a
    single model load, so the op's growth above the settled model RSS is real.
    Memory is deterministic given fixed tensor shapes, so one measurement (we
    take the max over a couple of internal tries) suffices --- no averaging.
    """
    torch.manual_seed(SEED)
    _CUR_NAME["name"] = name
    model, _sub = load_model(name)
    sd = model.state_dict()
    best = 0.0
    base = None
    for _ in range(2):
        m = measure_once(method, sd, model)
        best = max(best, m["peak_rss_delta_mb"])
        base = m["base_rss_mb"]
        gc.collect()
    print(f"RESULT\t{name}\t{method}\t{best:.3f}\t{base:.1f}", flush=True)


def driver():
    import subprocess
    import sys
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"[info] methods = {METHODS}")
    print(f"[info] writing rows to {OUT_CSV}")
    fieldnames = ["model_name", "method", "peak_rss_delta_mb",
                  "base_rss_mb", "hardware"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for name in MODELS:
            for method in METHODS:
                print(f"\n=== {name} / {method} ===", flush=True)
                proc = subprocess.run(
                    [sys.executable, "-u", __file__, name, method],
                    capture_output=True, text=True, env=env)
                line = next((ln for ln in proc.stdout.splitlines()
                             if ln.startswith("RESULT\t")), None)
                if line is None:
                    print(f"[error] no RESULT for {name}/{method}")
                    print(proc.stdout[-800:])
                    print(proc.stderr[-800:])
                    continue
                _, mdl, meth, delta, base = line.split("\t")
                writer.writerow({"model_name": mdl, "method": meth,
                                 "peak_rss_delta_mb": delta,
                                 "base_rss_mb": base, "hardware": "CPU"})
                fh.flush()
                print(f"  peak RSS delta = {delta} MB (base {base} MB)")
    print(f"\n[done] wrote {OUT_CSV}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        worker(sys.argv[1], sys.argv[2])
    else:
        driver()
