"""Tie-analysis for the Canonical-Ordering Adaptive Attack across a large corpus
of torchvision CNNs with ImageNet pretrained weights.

For each model, loads the pretrained state_dict and, for every 2D+ weight tensor
(conv / linear), computes per-output-channel L1/L2/variance metrics and measures:

  - n_layers analyzed
  - total permutable units (output channels)
  - fraction of units with a UNIQUE metric value (uniqueness_prob)
  - fraction of units inside a tie group of size >= 2
  - mean tie-group size across non-trivial groups

Aggregates per-model stats and writes a summary CSV + prints a table.
"""
import argparse
import copy
import math
import os
import sys
import time
import urllib.parse
import warnings
from typing import List

import pandas as pd
import torch
import torchvision

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub config so we don't need IMAGENET12_ROOT
import types as _types
_cfg_stub = _types.ModuleType("neu_perm.config")
_cfg_stub.REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_cfg_stub.RESULTS_DIR = os.path.join(_cfg_stub.REPO_ROOT, "results")
_cfg_stub.IMAGENET12_ROOT = ""
sys.modules["neu_perm.config"] = _cfg_stub
RESULTS_DIR = _cfg_stub.RESULTS_DIR


# ---------------------------------------------------------------------------
# Corpus: CNN models with ImageNet-1K pretrained weights (excludes transformers,
# quantized variants, and excessively large downloads)
# ---------------------------------------------------------------------------
CORPUS = [
    # Classic
    "alexnet",
    # VGG
    "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
    # ResNet / ResNeXt / Wide
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d",
    "wide_resnet50_2", "wide_resnet101_2",
    # DenseNet
    "densenet121", "densenet161", "densenet169", "densenet201",
    # Inception family
    "googlenet", "inception_v3",
    # MobileNet / MNASNet / SqueezeNet / ShuffleNet
    "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
    "mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3",
    "squeezenet1_0", "squeezenet1_1",
    "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0",
    # EfficientNet
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5",
    "efficientnet_v2_s", "efficientnet_v2_m",
    # RegNet (CNN-style, skip the >1GB ones)
    "regnet_x_400mf", "regnet_x_800mf", "regnet_x_1_6gf", "regnet_x_3_2gf",
    "regnet_x_8gf", "regnet_x_16gf",
    "regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf", "regnet_y_3_2gf",
    "regnet_y_8gf", "regnet_y_16gf",
    # ConvNeXt
    "convnext_tiny", "convnext_small", "convnext_base",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    print(msg, flush=True)


def load_pretrained_sd(model_name: str) -> dict:
    """Load pretrained state_dict, working around hash-mismatch errors."""
    weights = torchvision.models.get_model_weights(model_name).DEFAULT
    try:
        model = torchvision.models.get_model(model_name, weights=weights)
    except RuntimeError:
        url = weights.url
        fn = os.path.basename(urllib.parse.urlparse(url).path)
        cached = os.path.join(torch.hub.get_dir(), "checkpoints", fn)
        if not os.path.exists(cached):
            torch.hub.download_url_to_file(url, cached)
        sd = torch.load(cached, map_location="cpu")
        model = torchvision.models.get_model(model_name, weights=None)
        model.load_state_dict(sd)
    return copy.deepcopy(model.state_dict())


def _scalar_metric(w: torch.Tensor, metric: str) -> torch.Tensor:
    n = w.shape[0]
    flat = w.reshape(n, -1).float()
    if metric == "l1_norm":
        return flat.abs().sum(dim=1)
    if metric == "l2_norm":
        return flat.norm(p=2, dim=1)
    if metric == "variance":
        return flat.var(dim=1)
    raise ValueError(metric)


def analyze_layer(vals: torch.Tensor, tol: float) -> dict:
    """Count ties exactly and within tolerance."""
    n = vals.numel()
    if n <= 1:
        return {
            "n": n,
            "n_unique_exact": n,
            "n_unique_tol": n,
            "n_tied_exact": 0,
            "n_tied_tol": 0,
            "n_groups_ge2_exact": 0,
            "max_group_exact": 1 if n else 0,
        }
    sorted_v, _ = torch.sort(vals)
    # Exact ties
    n_unique_exact = int((sorted_v[1:] != sorted_v[:-1]).sum().item()) + 1
    # Approximate ties within tol
    diffs_tol = (sorted_v[1:] - sorted_v[:-1]).abs() > tol
    n_unique_tol = int(diffs_tol.sum().item()) + 1
    # Exact tie group stats
    eq = (sorted_v[1:] == sorted_v[:-1])
    groups_ge2 = 0
    max_group = 1
    cur = 1
    for same in eq.tolist():
        if same:
            cur += 1
        else:
            if cur >= 2:
                groups_ge2 += 1
                if cur > max_group:
                    max_group = cur
            cur = 1
    if cur >= 2:
        groups_ge2 += 1
        if cur > max_group:
            max_group = cur
    return {
        "n": n,
        "n_unique_exact": n_unique_exact,
        "n_unique_tol": n_unique_tol,
        "n_tied_exact": n - n_unique_exact,
        "n_tied_tol": n - n_unique_tol,
        "n_groups_ge2_exact": groups_ge2,
        "max_group_exact": max_group,
    }


def analyze_model(model_name: str, metrics: List[str], tol: float) -> dict:
    """Analyze tie statistics for every 2D+ weight tensor in the model."""
    sd = load_pretrained_sd(model_name)
    # Track per-layer aggregates
    agg = {m: {"n_total": 0, "n_unique_exact": 0, "n_unique_tol": 0,
               "n_layers": 0, "max_group": 0, "groups_ge2": 0} for m in metrics}

    for key, w in sd.items():
        if not torch.is_tensor(w) or not w.dtype.is_floating_point:
            continue
        if w.ndim < 2:
            continue  # skip 1-D tensors (biases, BN params)
        # Must have at least 2 output channels to have a meaningful ranking
        if w.shape[0] < 2:
            continue
        for metric in metrics:
            vals = _scalar_metric(w, metric)
            stats = analyze_layer(vals, tol)
            a = agg[metric]
            a["n_layers"] += 1
            a["n_total"] += stats["n"]
            a["n_unique_exact"] += stats["n_unique_exact"]
            a["n_unique_tol"] += stats["n_unique_tol"]
            a["groups_ge2"] += stats["n_groups_ge2_exact"]
            if stats["max_group_exact"] > a["max_group"]:
                a["max_group"] = stats["max_group_exact"]

    del sd
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="+", default=["l1_norm", "l2_norm", "variance"])
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--models", nargs="+", default=CORPUS)
    ap.add_argument(
        "--csv",
        default=os.path.join(RESULTS_DIR, "canonical_ties_corpus.csv"),
    )
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Resume support
    if os.path.exists(args.csv):
        existing = pd.read_csv(args.csv)
        done_models = set(existing["model_name"].tolist())
        rows = existing.to_dict("records")
        log(f"Loaded {len(done_models)} existing models from {args.csv}")
    else:
        done_models = set()
        rows = []

    for model_name in args.models:
        if model_name in done_models:
            log(f"  SKIP {model_name} (already analyzed)")
            continue
        log(f"\n=== {model_name} ===")
        t0 = time.time()
        try:
            agg = analyze_model(model_name, args.metrics, args.tol)
        except Exception as e:
            log(f"  ERROR: {e}")
            continue
        elapsed = time.time() - t0

        for metric, a in agg.items():
            n_total = a["n_total"]
            uniq_frac = a["n_unique_exact"] / n_total if n_total else 0.0
            uniq_frac_tol = a["n_unique_tol"] / n_total if n_total else 0.0
            row = {
                "model_name": model_name,
                "metric": metric,
                "n_layers": a["n_layers"],
                "n_total_units": n_total,
                "n_unique_exact": a["n_unique_exact"],
                "n_unique_tol": a["n_unique_tol"],
                "uniq_frac_exact": uniq_frac,
                "uniq_frac_tol": uniq_frac_tol,
                "n_groups_ge2": a["groups_ge2"],
                "max_tie_group": a["max_group"],
                "tolerance": args.tol,
                "elapsed": elapsed,
            }
            rows.append(row)
            if metric == "l1_norm":
                log(f"  L1: layers={a['n_layers']} units={n_total} "
                    f"unique_exact={uniq_frac*100:.2f}% "
                    f"unique_tol={uniq_frac_tol*100:.2f}% "
                    f"tied_groups={a['groups_ge2']} "
                    f"max_tie_group={a['max_group']} "
                    f"({elapsed:.1f}s)")

        # Save incrementally
        pd.DataFrame(rows).to_csv(args.csv, index=False)

    # Summary table (one row per model, L1 metric)
    df = pd.DataFrame(rows)
    log("\n" + "=" * 90)
    log("Summary (L1 metric)")
    log("=" * 90)
    l1 = df[df["metric"] == "l1_norm"].sort_values("uniq_frac_exact")
    log(f"{'model':<28s} {'units':>10s} {'uniq%':>8s} {'uniq(tol)%':>12s} {'tied_groups':>12s} {'max_group':>10s}")
    for _, r in l1.iterrows():
        log(f"{r['model_name']:<28s} {int(r['n_total_units']):>10d} "
            f"{r['uniq_frac_exact']*100:>7.2f}% "
            f"{r['uniq_frac_tol']*100:>11.2f}% "
            f"{int(r['n_groups_ge2']):>12d} "
            f"{int(r['max_tie_group']):>10d}")

    log(f"\nCSV saved to {args.csv}")


if __name__ == "__main__":
    main()
