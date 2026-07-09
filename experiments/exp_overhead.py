"""Overhead experiment: wall-clock + peak memory of applying NeuPerm.

Reproduces Table `tab:overhead` (wall-clock seconds to apply `permute_model`
per architecture) and Table `tab:overhead_mem` (peak CPU MB via tracemalloc and
peak GPU MB via CUDA allocator). Model loading is excluded from timing — only
the permutation itself is measured, `N_REPEATS` times per model.

Configure the run in the `__main__` block at the bottom (model list, repeats,
seed), then:

    python experiments/exp_overhead.py

Output CSV: `<RESULTS_DIR>/neuperm_overhead.csv` with one row per (model,
repeat): model_name, wall_clock_s, peak_cpu_mb, peak_gpu_mb, hardware,
repeat_idx. The CNNs are loaded untrained (weights are irrelevant to timing);
the Llama/Qwen loaders pull from HuggingFace and need network + the transformers
package. Set NEUPERM_LLAMA_ID / NEUPERM_QWEN_ID to override the LLM checkpoints.
"""
import csv
import gc
import os
import time
import tracemalloc
import traceback

import numpy as np
import torch
import torchvision.models as tv_models

from neu_perm.models import model_map, get_model
from neu_perm.perm import permute_model
from neu_perm.config import RESULTS_DIR

# Default LLM checkpoints; overridable via env so no identity/path is baked in.
LLAMA_MODEL_ID = os.environ.get("NEUPERM_LLAMA_ID", "meta-llama/Llama-3.2-1B-Instruct")
QWEN_MODEL_ID = os.environ.get("NEUPERM_QWEN_ID", "Qwen/Qwen2.5-1.5B-Instruct")


def load_llama(dtype=torch.bfloat16):
    """Load the Llama causal-LM (HuggingFace); returns (model, tokenizer)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_ID, torch_dtype=dtype)
    return model, tok


def load_qwen(dtype=torch.bfloat16):
    """Load the Qwen causal-LM (HuggingFace); returns (model, tokenizer)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_ID, torch_dtype=dtype)
    return model, tok


def _load_torchvision_model(name: str):
    """Load a torchvision CNN by name with untrained weights.

    Returns (model, substitution_note_or_None); the note records any API
    fallback so a run on an older torchvision stays auditable.
    """
    ctor = getattr(tv_models, name, None)
    if ctor is None:
        raise ValueError(f"torchvision.models has no attribute {name!r}")
    substitution = None
    try:
        model = ctor(weights=None)
    except TypeError:
        # Very old torchvision API predates the `weights=` kwarg.
        model = ctor(pretrained=False)
        substitution = f"{name}: fell back to pretrained=False (old torchvision API)"
    except Exception:
        # If weights=None fails for any other reason, try num_classes=1000.
        model = ctor(num_classes=1000)
        substitution = f"{name}: fell back to num_classes=1000"
    return model, substitution


def load_model(name: str):
    """Return (model, substitution_note_or_None). NOT timed."""
    if name == "llama-3.2-1b":
        model, _tok = load_llama()
        return model, None
    if name == "qwen2.5-1.5b":
        model, _tok = load_qwen()
        return model, None
    if name in model_map:
        # neu_perm.models.get_model uses dataset metadata; 'imagenet12' -> 1000 classes.
        try:
            model = get_model(name, dataset="imagenet12")
            return model, None
        except Exception:
            return _load_torchvision_model(name)
    # Not in neu_perm.models.model_map -> torchvision direct.
    return _load_torchvision_model(name)


def measure_once(name: str, sd):
    """Run permute_model on `sd` once; return timing + peak-memory measurements."""
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

    peak_gpu = torch.cuda.max_memory_allocated() / (1024 ** 2) if cuda_avail else 0.0

    return {
        "wall_clock_s": wall,
        "peak_cpu_mb": peak_cpu,
        "peak_gpu_mb": peak_gpu,
    }


def run_overhead(models, n_repeats, out_csv):
    """Measure permute_model overhead for each model and write `out_csv`."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        hardware = torch.cuda.get_device_name(0)
    else:
        hardware = "CPU"
        print("[warn] CUDA not available — peak_gpu_mb will be 0 for all rows.")

    print(f"[info] hardware = {hardware}")
    print(f"[info] writing rows to {out_csv}")

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
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for name in models:
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

            for rep in range(n_repeats):
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
    for name in models:
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

    print(f"\n[done] wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    # --- run configuration ---
    models = [
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
    n_repeats = 5
    seed = 0
    out_csv = os.path.join(RESULTS_DIR, "neuperm_overhead.csv")
    # -------------------------

    # Seed so the (randomized) permutations are reproducible run-to-run.
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_overhead(models, n_repeats, out_csv)
