"""Tie-analysis PoC: does zeroing the last X LSBs of every parameter create
enough ties to defeat the canonical-ordering attack?

Approach: for each small CNN, load pretrained weights, zero the low X bits of
every float32 parameter (by view-as-int32 / bitmask), then run the same
tie analysis as exp_canonical_tie_corpus.py.

For X=0 (baseline), X=1, X=2 we report exact-uniqueness fractions.
"""
import argparse
import copy
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

# Stub config
import types as _types
_cfg_stub = _types.ModuleType("neu_perm.config")
_cfg_stub.REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_cfg_stub.RESULTS_DIR = os.path.join(_cfg_stub.REPO_ROOT, "results")
_cfg_stub.IMAGENET12_ROOT = ""
sys.modules["neu_perm.config"] = _cfg_stub
RESULTS_DIR = _cfg_stub.RESULTS_DIR


SMALL_CORPUS = [
    "resnet18",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "squeezenet1_1",
    "mnasnet0_5",
    "shufflenet_v2_x0_5",
]

LSB_VALUES = [0, 1, 2]  # 0 = baseline (no zeroing)


def log(msg):
    print(msg, flush=True)


def load_pretrained_sd(model_name: str) -> dict:
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


def zero_lsbs(sd: dict, n_bits: int) -> dict:
    """Zero the low n_bits of every float32 parameter via view-as-int32 bitmask.

    Works on float32 only. For other dtypes (float16/bfloat16) the logic is
    analogous but we restrict to float32 for this PoC.
    """
    if n_bits <= 0:
        return sd
    out = {}
    mask = ~((1 << n_bits) - 1)  # e.g. n=1 -> ...11111110, n=2 -> ...11111100
    mask = torch.tensor(mask, dtype=torch.int32)
    for k, v in sd.items():
        if not torch.is_tensor(v):
            out[k] = v
            continue
        if v.dtype == torch.float32:
            iv = v.contiguous().view(torch.int32)
            iv = iv & mask
            out[k] = iv.view(torch.float32)
        else:
            out[k] = v
    return out


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


def analyze_layer(vals: torch.Tensor) -> dict:
    n = vals.numel()
    if n <= 1:
        return {"n": n, "n_unique_exact": n, "n_groups_ge2": 0, "max_group": 1}
    sorted_v, _ = torch.sort(vals)
    n_unique = int((sorted_v[1:] != sorted_v[:-1]).sum().item()) + 1
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
                max_group = max(max_group, cur)
            cur = 1
    if cur >= 2:
        groups_ge2 += 1
        max_group = max(max_group, cur)
    return {"n": n, "n_unique_exact": n_unique,
            "n_groups_ge2": groups_ge2, "max_group": max_group}


def analyze_model(sd: dict, metric: str = "l1_norm") -> dict:
    agg = {"n_layers": 0, "n_total": 0, "n_unique_exact": 0,
           "n_groups_ge2": 0, "max_group": 0}
    for key, w in sd.items():
        if not torch.is_tensor(w) or not w.dtype.is_floating_point:
            continue
        if w.ndim < 2 or w.shape[0] < 2:
            continue
        vals = _scalar_metric(w, metric)
        stats = analyze_layer(vals)
        agg["n_layers"] += 1
        agg["n_total"] += stats["n"]
        agg["n_unique_exact"] += stats["n_unique_exact"]
        agg["n_groups_ge2"] += stats["n_groups_ge2"]
        agg["max_group"] = max(agg["max_group"], stats["max_group"])
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=SMALL_CORPUS)
    ap.add_argument("--lsb", type=int, nargs="+", default=LSB_VALUES)
    ap.add_argument("--metric", default="l1_norm")
    ap.add_argument("--csv", default=os.path.join(RESULTS_DIR, "canonical_ties_lsb.csv"))
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    for model_name in args.models:
        log(f"\n=== {model_name} ===")
        t0 = time.time()
        try:
            sd_orig = load_pretrained_sd(model_name)
        except Exception as e:
            log(f"  ERROR loading: {e}")
            continue
        log(f"  loaded ({time.time()-t0:.1f}s)")

        for nb in args.lsb:
            sd = zero_lsbs(sd_orig, nb) if nb > 0 else sd_orig
            agg = analyze_model(sd, args.metric)
            n_total = agg["n_total"]
            uniq = agg["n_unique_exact"] / n_total if n_total else 0.0
            tied = n_total - agg["n_unique_exact"]
            log(f"  LSB={nb}: units={n_total} unique={uniq*100:.4f}% "
                f"tied={tied} groups_ge2={agg['n_groups_ge2']} "
                f"max_group={agg['max_group']}")
            rows.append({
                "model_name": model_name,
                "metric": args.metric,
                "lsb_zeroed": nb,
                "n_layers": agg["n_layers"],
                "n_total": n_total,
                "n_unique_exact": agg["n_unique_exact"],
                "n_tied": tied,
                "uniq_frac": uniq,
                "n_groups_ge2": agg["n_groups_ge2"],
                "max_tie_group": agg["max_group"],
            })

        pd.DataFrame(rows).to_csv(args.csv, index=False)

    # Summary
    df = pd.DataFrame(rows)
    log("\n" + "=" * 90)
    log("Summary (sorted by model, shows impact of zeroing LSBs)")
    log("=" * 90)
    log(f"{'model':<22s} {'LSB':>4s} {'units':>8s} {'uniq%':>10s} {'tied':>8s} "
        f"{'groups':>8s} {'max_grp':>8s}")
    for m in args.models:
        sub = df[df["model_name"] == m]
        for _, r in sub.iterrows():
            log(f"{r['model_name']:<22s} {int(r['lsb_zeroed']):>4d} "
                f"{int(r['n_total']):>8d} {r['uniq_frac']*100:>9.4f}% "
                f"{int(r['n_tied']):>8d} {int(r['n_groups_ge2']):>8d} "
                f"{int(r['max_tie_group']):>8d}")
        log("")
    log(f"CSV saved to {args.csv}")


if __name__ == "__main__":
    main()
