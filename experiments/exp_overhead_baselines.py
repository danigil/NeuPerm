"""
Overhead experiment (baselines) — NeuPerm.

Companion to ``exp_overhead.py``. Measures wall-clock seconds + peak CPU MB +
peak GPU MB of the *comparable* sanitization methods per architecture, using the
identical methodology (``time.perf_counter`` around the operation only,
``tracemalloc`` for peak CPU, ``torch.cuda`` for peak GPU; N repeats, first
discarded as warm-up). NeuPerm's own numbers come from ``exp_overhead.py`` and
are not recomputed here.

Methods (settings matched to the paper's comparison):
  - quant8   : 8-bit per-channel quantize-dequantize, BN layers skipped (PTQ-8)
  - noise    : additive Gaussian noise, sigma = 1e-4 (RN 0.0001)
  - prune    : random unstructured pruning, amount = 0.2, made permanent

Each method is timed as it would be deployed: it rewrites already-loaded weights
in place. Model loading and the one-time allocation of the benchmark's working
copy are both excluded from the timed region, identically for all three methods.

Usage:
    cd /path/to/NeuPerm
    PYTHONPATH=. python experiments/exp_overhead_baselines.py

Output: results/baselines_overhead.csv
"""

import copy
import csv
import gc
import os
import time
import tracemalloc
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from neu_perm.config import RESULTS_DIR
from neu_perm.quantization import quantize_dequantize_sd, get_bn_keys_from_sd

# Reuse the exact model list + loader from the NeuPerm overhead experiment so the
# two tables cover the same architectures under the same loading path.
from exp_overhead import (  # noqa: E402
    MODELS, load_model, tagged_out, warm_up_process)

SEED = 42
N_REPEATS = int(os.environ.get("OVERHEAD_REPEATS", "12"))  # see exp_overhead.py
NOISE_SIGMA = 1e-4
PRUNE_AMOUNT = 0.2
QUANT_BITS = 8

METHODS = os.environ.get("OVERHEAD_METHODS", "quant8,noise,prune").split(",")
OUT_CSV = tagged_out("baselines_overhead.csv")


def _apply_noise(work, sigma):
    """Add N(0, sigma^2) to every float tensor of *work*, in place.

    Torch-native and single-precision: the Gaussian is sampled in float32
    (matching the original float32 noise) and cast to each tensor's dtype, then
    added in place. This is the same operation as the original numpy-float64
    implementation---identical distribution on the same tensors, non-float
    tensors untouched---but avoids the float64 sampling, downcast, and
    numpy->torch bridge. Soundness (perturbation std == sigma, non-float tensors
    unchanged) is verified separately.
    """
    for v in work.values():
        if v.is_floating_point():
            noise = torch.randn(v.shape, dtype=torch.float32).mul_(sigma)
            v.add_(noise.to(v.dtype))


def _prune(model_copy, amount):
    """Random unstructured pruning on Conv2d/Linear weights, made permanent."""
    for _, module in model_copy.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.random_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model_copy


def measure_once(method: str, sd, model):
    """Run one method once on fresh inputs; return timing/memory dict.

    Setup that a deployer would not repeat per model (allocating the fresh
    working copy of the weights, deep-copying the model for pruning, computing
    BN keys for quant) is done OUTSIDE the timed region, so all three methods
    are timed on the same footing: the sanitization operation alone, run in
    place on a copy that already exists. This mirrors ``exp_overhead.py``, which
    times only ``permute_model(..., inplace=True)``.
    """
    cuda_avail = torch.cuda.is_available()

    # --- untimed setup ---
    # Each method gets a fresh copy of the weights so repeats do not accumulate,
    # and so the timed region is the sanitization operation itself (a deployer
    # overwrites the weights in place; the copy is a benchmark artifact).
    if method == "prune":
        payload = copy.deepcopy(model)
    elif method == "quant8":
        # Working copy AND skip_keys, both untimed. Passing ``inplace=False``
        # instead makes quantize_dequantize_sd deep-copy the whole state dict
        # *inside* the timed region, charging PTQ-8 for a copy that noise and
        # prune exclude. Matches exp_overhead_memory.py's setup for this method.
        payload = ({k: v.clone() for k, v in sd.items()},
                   get_bn_keys_from_sd(sd))
    elif method == "noise":
        payload = {k: v.clone() for k, v in sd.items()}  # fresh working weights
    else:
        payload = None

    gc.collect()
    if cuda_avail:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tracemalloc.start()
    t0 = time.perf_counter()
    if method == "quant8":
        work, bn = payload
        quantize_dequantize_sd(work, n_bits=QUANT_BITS, per_channel=True,
                               inplace=True, skip_keys=bn)
    elif method == "noise":
        _apply_noise(payload, NOISE_SIGMA)
    elif method == "prune":
        _prune(payload, PRUNE_AMOUNT)
    else:
        raise ValueError(method)
    wall = time.perf_counter() - t0
    peak_cpu = tracemalloc.get_traced_memory()[1] / (1024 ** 2)
    tracemalloc.stop()

    peak_gpu = torch.cuda.max_memory_allocated() / (1024 ** 2) if cuda_avail else 0.0

    del payload
    return {"wall_clock_s": wall, "peak_cpu_mb": peak_cpu, "peak_gpu_mb": peak_gpu}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cuda_avail = torch.cuda.is_available()
    hardware = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    if not cuda_avail:
        print("[warn] CUDA not available — peak_gpu_mb will be 0 for all rows.")
    print(f"[info] hardware = {hardware}")
    print(f"[info] writing rows to {OUT_CSV}")

    warm_up_process(
        lambda name, sd, model: [measure_once(m, sd, model) for m in METHODS])

    rows, failures = [], []
    fieldnames = ["model_name", "method", "wall_clock_s", "peak_cpu_mb",
                  "peak_gpu_mb", "hardware", "repeat_idx"]

    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for name in MODELS:
            print(f"\n=== {name} ===")
            try:
                model, _sub = load_model(name)
                sd = model.state_dict()
            except Exception as e:
                print(f"[error] failed to load {name}: {e}")
                traceback.print_exc()
                failures.append((name, "load", repr(e)))
                continue

            for method in METHODS:
                for rep in range(N_REPEATS):
                    try:
                        m = measure_once(method, sd, model)
                    except Exception as e:
                        print(f"[error] {method}({name}) rep {rep} failed: {e}")
                        traceback.print_exc()
                        failures.append((name, f"{method} rep {rep}", repr(e)))
                        continue
                    row = {
                        "model_name": name, "method": method,
                        "wall_clock_s": f"{m['wall_clock_s']:.6f}",
                        "peak_cpu_mb": f"{m['peak_cpu_mb']:.3f}",
                        "peak_gpu_mb": f"{m['peak_gpu_mb']:.3f}",
                        "hardware": hardware, "repeat_idx": rep,
                    }
                    rows.append(row)
                    writer.writerow(row)
                    fh.flush()
                    print(f"  {method} rep {rep}: wall={m['wall_clock_s']:.4f}s "
                          f"cpu_peak={m['peak_cpu_mb']:.2f}MB "
                          f"gpu_peak={m['peak_gpu_mb']:.2f}MB")

            del model, sd
            gc.collect()
            if cuda_avail:
                torch.cuda.empty_cache()

    # ---------- Summary (mean over post-warmup repeats) ----------
    print("\n" + "=" * 64)
    print("SUMMARY — mean wall_clock_s per (model, method), rep 0 discarded")
    print("=" * 64)
    agg = {}
    for r in rows:
        if int(r["repeat_idx"]) == 0:
            continue
        agg.setdefault((r["model_name"], r["method"]), []).append(
            float(r["wall_clock_s"]))
    hdr = f"{'model':<22}  {'method':<8}  {'n':>3}  {'mean_s':>10}  {'max_s':>10}"
    print(hdr)
    print("-" * len(hdr))
    for name in MODELS:
        for method in METHODS:
            vals = agg.get((name, method))
            if not vals:
                print(f"{name:<22}  {method:<8}  {'-':>3}  {'(no data)':>10}")
                continue
            mean = sum(vals) / len(vals)
            print(f"{name:<22}  {method:<8}  {len(vals):>3}  "
                  f"{mean:>10.4f}  {max(vals):>10.4f}")

    if failures:
        print("\nFailures:")
        for name, stage, err in failures:
            print(f"  - {name} [{stage}]: {err}")
    print(f"\n[done] wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
