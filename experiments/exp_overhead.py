"""
Overhead experiment — NeuPerm.

Measures wall-clock seconds + peak CPU MB + peak GPU MB of `permute_model`
per architecture. Excludes model loading from timing. 5 repeats per model.

Usage:
    cd /path/to/NeuPerm
    PYTHONPATH=. python experiments/exp_overhead.py

Output: results/neuperm_overhead.csv
"""

import csv
import gc
import os
import sys
import time
import tracemalloc
import traceback

import torch
import torchvision.models as tv_models

from neu_perm.models import model_map, get_model
from neu_perm.perm import permute_model
from neu_perm.config import RESULTS_DIR


def load_llama_3b(dtype=torch.bfloat16):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    return model, tok


def load_qwen_1_5b(dtype=torch.bfloat16):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    return model, tok


MODELS = [
    "vgg11",
    "vgg16",
    "resnet50",
    "resnet101",
    "densenet121",
    "efficientnet_b0",
    "efficientnet_b4",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "llama-3.2-1b",
    "qwen2.5-1.5b",
]

# Repeat count is env-overridable. The default is 12 (10 after discarding two
# warm-up repeats), not the original 5: n=4 post-warm-up proved too few to
# reach a 95% CI half-width under ~10% of the mean, because a single stalled
# repeat dominates the sample sd at that size, and one discarded repeat left
# residual warm-up bias (measured 2026-08-25 on an idle host). All three
# overhead scripts share this default, so a bare run of any one of them is
# comparable with a bare run of the others.
N_REPEATS = int(os.environ.get("OVERHEAD_REPEATS", "12"))
# Output name is overridable so a re-measurement can be written alongside the
# published run instead of overwriting it (same env-var pattern as
# exp_overhead_baselines.py).
def tagged_out(default_name: str) -> str:
    """Resolve this script's output path.

    ``OVERHEAD_OUT`` names the file outright. ``OVERHEAD_TAG`` instead inserts a
    tag before the extension of the script's own default, which is what a
    multi-pass run wants: exp_overhead.py and exp_overhead_baselines.py both
    read ``OVERHEAD_OUT``, so setting it for a whole session would silently point
    both at one file and destroy a pass. The tag keeps their names distinct.
    """
    explicit = os.environ.get("OVERHEAD_OUT")
    if explicit:
        return os.path.join(RESULTS_DIR, explicit)
    tag = os.environ.get("OVERHEAD_TAG", "")
    stem, ext = os.path.splitext(default_name)
    return os.path.join(RESULTS_DIR, f"{stem}{tag}{ext}")


OUT_CSV = tagged_out("neuperm_overhead.csv")

# A cold process inflates the first cell it measures: measured 2026-08-25, vgg11
# NeuPerm read 0.1780 s when measured first and 0.0972 s when one throwaway model
# was measured before it, against vgg16's 0.0949 s. Warming the process here means
# no cell of the table is ever the first thing this interpreter does. Results are
# discarded. Set OVERHEAD_WARMUP_MODEL="" to disable.
WARMUP_MODEL = os.environ.get("OVERHEAD_WARMUP_MODEL", "mobilenet_v3_small")
WARMUP_REPEATS = int(os.environ.get("OVERHEAD_WARMUP_REPEATS", "3"))


def warm_up_process(run_one):
    """Exercise the measurement path on a throwaway model; discard the results.

    *run_one* is called as ``run_one(name, sd, model)`` so both the NeuPerm and
    the baselines driver can share this, each timing whatever it normally times.
    """
    if not WARMUP_MODEL:
        print("[warmup] disabled (OVERHEAD_WARMUP_MODEL empty)", flush=True)
        return
    print(f"[warmup] {WARMUP_MODEL} x{WARMUP_REPEATS}, discarded", flush=True)
    model, _sub = load_model(WARMUP_MODEL)
    sd = model.state_dict()
    for _ in range(WARMUP_REPEATS):
        run_one(WARMUP_MODEL, sd, model)
    del model, sd
    gc.collect()


def _load_torchvision_model(name: str):
    """Load a torchvision CNN by name, untrained weights."""
    ctor = getattr(tv_models, name, None)
    if ctor is None:
        raise ValueError(f"torchvision.models has no attribute {name!r}")
    substitution = None
    try:
        model = ctor(weights=None)
    except TypeError:
        # very old torchvision API
        model = ctor(pretrained=False)
        substitution = f"{name}: fell back to pretrained=False (old torchvision API)"
    except Exception:
        # if weights=None fails for any other reason, try num_classes=1000
        model = ctor(num_classes=1000)
        substitution = f"{name}: fell back to num_classes=1000"
    return model, substitution


def load_model(name: str):
    """Return (model, substitution_note_or_None). NOT timed."""
    if name == "llama-3.2-1b":
        model, _tok = load_llama_3b()
        return model, None
    if name == "qwen2.5-1.5b":
        model, _tok = load_qwen_1_5b()
        return model, None
    if name in model_map:
        # neu_perm.models.get_model uses dataset metadata; 'imagenet12' uses 1000 classes
        try:
            model = get_model(name, dataset="imagenet12")
            return model, None
        except Exception:
            # fallback to torchvision direct
            return _load_torchvision_model(name)
    # not in neu_perm.models.model_map -> torchvision direct
    return _load_torchvision_model(name)


def measure_once(name: str, sd):
    """Run permute_model on `sd` once, return dict with measurements."""
    cuda_avail = torch.cuda.is_available()

    gc.collect()
    if cuda_avail:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tracemalloc.start()
    t0 = time.perf_counter()
    permute_model(name, sd, inplace=True)
    wall = time.perf_counter() - t0
    peak_cpu = tracemalloc.get_traced_memory()[1] / (1024 ** 2)
    tracemalloc.stop()

    if cuda_avail:
        peak_gpu = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_gpu = 0.0

    return {
        "wall_clock_s": wall,
        "peak_cpu_mb": peak_cpu,
        "peak_gpu_mb": peak_gpu,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        hardware = torch.cuda.get_device_name(0)
    else:
        hardware = "CPU"
        print("[warn] CUDA not available — peak_gpu_mb will be 0 for all rows.")

    print(f"[info] hardware = {hardware}")
    print(f"[info] writing rows to {OUT_CSV}")

    warm_up_process(lambda name, sd, model: measure_once(name, sd))

    rows = []
    substitutions = []
    failures = []

    fieldnames = [
        "model_name",
        "wall_clock_s",
        "peak_cpu_mb",
        "peak_gpu_mb",
        "hardware",
        "repeat_idx",
    ]

    # Open CSV for incremental writes so partial progress is preserved.
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for name in MODELS:
            print(f"\n=== {name} ===")
            try:
                model, sub = load_model(name)
            except Exception as e:
                print(f"[error] failed to load {name}: {e}")
                traceback.print_exc()
                failures.append((name, "load", repr(e)))
                continue

            if sub:
                print(f"[note] {sub}")
                substitutions.append(sub)

            try:
                sd = model.state_dict()
            except Exception as e:
                print(f"[error] failed to get state_dict for {name}: {e}")
                failures.append((name, "state_dict", repr(e)))
                del model
                gc.collect()
                continue

            # Free the model object; we only need sd for permute.
            del model
            gc.collect()

            for rep in range(N_REPEATS):
                try:
                    m = measure_once(name, sd)
                except Exception as e:
                    print(f"[error] permute_model({name}) rep {rep} failed: {e}")
                    traceback.print_exc()
                    failures.append((name, f"permute rep {rep}", repr(e)))
                    continue

                row = {
                    "model_name": name,
                    "wall_clock_s": f"{m['wall_clock_s']:.6f}",
                    "peak_cpu_mb": f"{m['peak_cpu_mb']:.3f}",
                    "peak_gpu_mb": f"{m['peak_gpu_mb']:.3f}",
                    "hardware": hardware,
                    "repeat_idx": rep,
                }
                rows.append(row)
                writer.writerow(row)
                fh.flush()
                print(
                    f"  rep {rep}: wall={m['wall_clock_s']:.4f}s  "
                    f"cpu_peak={m['peak_cpu_mb']:.2f}MB  "
                    f"gpu_peak={m['peak_gpu_mb']:.2f}MB"
                )

            del sd
            gc.collect()
            if cuda_avail:
                torch.cuda.empty_cache()

    # ---------- Summary ----------
    print("\n" + "=" * 60)
    print("SUMMARY — mean wall_clock_s per model")
    print("=" * 60)
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model_name"], []).append(float(r["wall_clock_s"]))

    header = f"{'model':<22}  {'n':>3}  {'mean_s':>10}  {'std_s':>10}  {'min_s':>10}  {'max_s':>10}"
    print(header)
    print("-" * len(header))
    for name in MODELS:
        if name not in by_model:
            print(f"{name:<22}  {'-':>3}  {'(no data)':>10}")
            continue
        vals = by_model[name]
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
        std = var ** 0.5
        print(
            f"{name:<22}  {n:>3}  {mean:>10.4f}  {std:>10.4f}  "
            f"{min(vals):>10.4f}  {max(vals):>10.4f}"
        )

    if substitutions:
        print("\nSubstitutions used:")
        for s in substitutions:
            print(f"  - {s}")

    if failures:
        print("\nFailures:")
        for name, stage, err in failures:
            print(f"  - {name} [{stage}]: {err}")

    print(f"\n[done] wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
