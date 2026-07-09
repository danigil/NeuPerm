"""NeuPerm-aware adaptive adversary: canonical (permutation-invariant) ordering attack.

Reproduces Table `tab:unshuffle_attack` and Lemma V. The adaptive adversary
embeds its payload against a *canonical* neuron/channel ordering (e.g. sort by
L1-norm) so that a random NeuPerm permutation cannot move a unit off its rank —
*unless* two units tie on the metric. This script measures how many units tie,
the resulting bit-error rate (BER) the adversary suffers under NeuPerm, the
theoretical recovery probability, and two tie-inducing analyses (a broad CNN
corpus and LSB-zeroing).

The script runs one of three sub-analyses, selected by the ``ANALYSIS`` config
var in the ``__main__`` block:

  ANALYSIS = "canonical_attack"   # per-model: ties + theory + attack BER + countermeasure + stego
  ANALYSIS = "tie_corpus"         # tie-uniqueness across a large torchvision CNN corpus
  ANALYSIS = "tie_lsb"            # does zeroing the low LSBs create enough ties to defeat the attack?

Each writes CSVs under ``RESULTS_DIR`` so the paper tables rebuild directly:

  canonical_attack  -> canonical_ties_<model>.csv, canonical_theory_<model>.csv,
                       canonical_attack_<model>.csv, canonical_countermeasure_<model>.csv,
                       canonical_stego_<model>.csv
  tie_corpus        -> canonical_ties_corpus.csv
  tie_lsb           -> canonical_ties_lsb.csv

Run (configure the block at the bottom, no CLI args):

    python experiments/exp_adaptive_canonical.py

Payloads are benign random bit-strings; runs are seeded for reproducibility.
"""

import copy
import gc
import os
import time
import urllib.parse
import warnings
from typing import List

import numpy as np
import pandas as pd
import torch
import torchvision

from neu_perm.config import RESULTS_DIR
from neu_perm.canonical import (
    compute_neuron_metric,
    get_permutable_sites,
    layer_tie_analysis,
    theoretical_recovery_probability,
    theoretical_uniqueness_bound,
    tie_group_sizes,
)
from neu_perm.countermeasures import tie_breaking_perturbation
from neu_perm.perm import permute_model
from neu_perm.steganography import (
    _map_perm_indices_to_sites,
    compute_analytical_model_ber,
    compute_ber,
    compute_detailed_tracked_ber,
    compute_tracked_model_ber,
    generate_payload,
    lsb_embed,
    lsb_extract,
    permute_model_tracked,
    total_embeddable_params,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def log(msg):
    print(msg, flush=True)


def _seed_everything(seed: int) -> None:
    """Seed numpy + torch global RNGs (NeuPerm draws permutations from torch)."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model_sd(model_name: str) -> dict:
    """Load a pretrained model (CNN or LLM) and return its CPU state_dict.

    Used by the ``canonical_attack`` sub-analysis, which needs the same model
    zoo NeuPerm supports (VGG/ResNet/DenseNet + Llama/Qwen).
    """
    if model_name in ("vgg11", "vgg16", "resnet50", "resnet101", "densenet121"):
        weights = torchvision.models.get_model_weights(model_name).DEFAULT
        model = torchvision.models.get_model(model_name, weights=weights)
        model = model.to("cpu").eval()
        return copy.deepcopy(model.state_dict())
    if model_name in ("llama-3.2-1b", "qwen2.5-1.5b"):
        from transformers import AutoModelForCausalLM

        hf_id = {
            "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
            "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        }[model_name]
        model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16)
        model = model.to("cpu").eval()
        return copy.deepcopy(model.state_dict())
    raise ValueError(f"Unknown model: {model_name}")


def load_pretrained_sd(model_name: str) -> dict:
    """Load a torchvision pretrained state_dict, working around hash mismatches.

    Used by the corpus / LSB tie analyses, which sweep the full torchvision zoo.
    """
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


def _channel_metric(w: torch.Tensor, metric: str) -> torch.Tensor:
    """Per-output-channel scalar metric for a raw weight tensor (corpus/LSB)."""
    n = w.shape[0]
    flat = w.reshape(n, -1).float()
    if metric == "l1_norm":
        return flat.abs().sum(dim=1)
    if metric == "l2_norm":
        return flat.norm(p=2, dim=1)
    if metric == "variance":
        return flat.var(dim=1)
    raise ValueError(metric)


# ===========================================================================
# Sub-analysis 1: canonical_attack (from exp_canonical.py)
# ===========================================================================


def run_tie_analysis(sd: dict, model_name: str, metrics: list) -> pd.DataFrame:
    """Per-site tie analysis across all metrics for a model."""
    dfs = [layer_tie_analysis(sd, model_name, m) for m in metrics]
    result = pd.concat(dfs, ignore_index=True)
    result["model_name"] = model_name
    return result


def compute_theoretical_bounds(sd: dict, model_name: str) -> pd.DataFrame:
    """Theoretical uniqueness / recovery bounds per permutation site."""
    sites = get_permutable_sites(model_name)
    rows = []
    for idx, site in enumerate(sites):
        metric_vals = compute_neuron_metric(sd, site, "l1_norm")
        n = metric_vals.shape[0]
        groups = tie_group_sizes(metric_vals)
        key_desc = list(site.keys.values())[0].rsplit(".", 1)[0]
        rows.append({
            "site_idx": idx,
            "site_key": key_desc,
            "n_neurons": n,
            "birthday_bound_fp32": theoretical_uniqueness_bound(n, 24),
            "birthday_bound_fp16": theoretical_uniqueness_bound(n, 11),
            "recovery_prob": theoretical_recovery_probability(groups),
            "n_tie_groups": len(groups),
            "model_name": model_name,
        })
    return pd.DataFrame(rows)


def run_canonical_attack(
    sd: dict,
    model_name: str,
    metrics: list,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Run the canonical-ordering attack ``n_repeats`` times and measure BER.

    Uses tracked permutation to compute exact BER per repeat, plus analytical
    expected BER (deterministic, same every repeat).
    """
    all_rows = []

    # Analytical expected BER (same for all repeats).
    for metric_name in metrics:
        agg_analytical = compute_analytical_model_ber(sd, model_name, metric_name)
        all_rows.append({
            "site_idx": -1,
            "site_kind": "analytical",
            "site_key": "all",
            "n_units": -1,
            "metric_name": metric_name,
            "ber": agg_analytical["aggregate_weighted"],
            "repeat_idx": -1,
            "model_name": model_name,
            "method": "canonical_attack_analytical",
        })

    for repeat_idx in range(n_repeats):
        sd_copy = copy.deepcopy(sd)
        sd_permuted, recorded_perms = permute_model_tracked(
            model_name, sd_copy, inplace=True
        )
        site_perms = _map_perm_indices_to_sites(model_name, recorded_perms)

        for metric_name in metrics:
            rows = compute_detailed_tracked_ber(sd, model_name, metric_name, site_perms)
            for row in rows:
                row["repeat_idx"] = repeat_idx
                row["model_name"] = model_name
                row["method"] = "canonical_attack"
            all_rows.extend(rows)

            agg = compute_tracked_model_ber(sd, model_name, metric_name, site_perms)
            all_rows.append({
                "site_idx": -1,
                "site_kind": "aggregate",
                "site_key": "all",
                "n_units": -1,
                "metric_name": metric_name,
                "ber": agg["aggregate_weighted"],
                "repeat_idx": repeat_idx,
                "model_name": model_name,
                "method": "canonical_attack",
            })

        del sd_copy, sd_permuted, recorded_perms
        gc.collect()

    return pd.DataFrame(all_rows)


def run_countermeasure_eval(
    sd: dict,
    model_name: str,
    metrics: list,
    n_repeats: int,
    cm_epsilons: list,
) -> pd.DataFrame:
    """Evaluate the tie-breaking countermeasure on top of NeuPerm.

    After NeuPerm + tie-breaking perturbation, re-measure analytical BER: the
    perturbation splits tie groups so the adversary's recovery degrades further.
    """
    all_rows = []
    for repeat_idx in range(n_repeats):
        sd_permuted = copy.deepcopy(sd)
        sd_permuted = permute_model(model_name, sd_permuted, inplace=True)

        for cm_eps in cm_epsilons:
            for metric_name in metrics:
                sd_cm = tie_breaking_perturbation(
                    sd_permuted,
                    model_name,
                    metric_name=metric_name,
                    epsilon=cm_eps,
                    inplace=False,
                )
                agg = compute_analytical_model_ber(sd_cm, model_name, metric_name)
                all_rows.append({
                    "model_name": model_name,
                    "metric_name": metric_name,
                    "cm_epsilon": cm_eps,
                    "ber_weighted": agg["aggregate_weighted"],
                    "ber_mean": agg["aggregate_mean"],
                    "repeat_idx": repeat_idx,
                    "method": "countermeasure",
                })

        del sd_permuted
        gc.collect()

    return pd.DataFrame(all_rows)


# LSB embedding depths swept in the stego analysis.
STEGO_SCHEMES = [
    ("lsb_1bit", {"n_lsb_bits": 1}),
    ("lsb_2bit", {"n_lsb_bits": 2}),
    ("lsb_4bit", {"n_lsb_bits": 4}),
    ("lsb_8bit", {"n_lsb_bits": 8}),
]


def run_stego_analysis(
    sd: dict,
    model_name: str,
    metrics: list,
    n_repeats: int,
    stego_schemes: list,
    payload_seed: int,
) -> pd.DataFrame:
    """Embed benign LSB payloads into ALL parameters, then run the attack.

    For each LSB depth: embed a random payload saturating capacity, verify
    round-trip on the clean stego-model, run tie analysis, apply NeuPerm and
    re-extract (extraction BER), and run the canonical-ordering attack.
    """
    all_rows = []
    n_total_params = total_embeddable_params(sd)
    schemes = stego_schemes if stego_schemes is not None else STEGO_SCHEMES

    for scheme_name, scheme_kwargs in schemes:
        n_lsb = scheme_kwargs["n_lsb_bits"]
        n_payload_bits = n_total_params * n_lsb
        payload = generate_payload(n_payload_bits, seed=payload_seed)

        log(f"    {scheme_name}: embedding {n_payload_bits:,} bits "
            f"({n_payload_bits / 8 / 1024:.1f} KB) into {n_total_params:,} params...")

        sd_stego = lsb_embed(sd, payload, n_lsb_bits=n_lsb, inplace=False)

        # Round-trip sanity check on the clean stego model.
        recovered_clean = lsb_extract(sd_stego, n_payload_bits, n_lsb_bits=n_lsb)
        clean_ber = compute_ber(payload, recovered_clean)
        all_rows.append({
            "scheme": scheme_name,
            "metric_name": "extraction",
            "analysis": "extraction_ber_clean",
            "value": clean_ber,
            "repeat_idx": -1,
            "model_name": model_name,
        })
        if clean_ber > 0.0:
            log(f"      WARNING: clean extraction BER = {clean_ber:.6f} (should be 0)")

        # Tie analysis on the stego model (does embedding change tie structure?).
        for metric_name in metrics:
            agg_analytical = compute_analytical_model_ber(sd_stego, model_name, metric_name)
            all_rows.append({
                "scheme": scheme_name,
                "metric_name": metric_name,
                "analysis": "tie_analytical_ber",
                "value": agg_analytical["aggregate_weighted"],
                "repeat_idx": -1,
                "model_name": model_name,
            })

        # Extraction + canonical attack after NeuPerm.
        for repeat_idx in range(n_repeats):
            sd_stego_copy = copy.deepcopy(sd_stego)
            sd_permuted, recorded_perms = permute_model_tracked(
                model_name, sd_stego_copy, inplace=True
            )
            site_perms = _map_perm_indices_to_sites(model_name, recorded_perms)

            recovered = lsb_extract(sd_permuted, n_payload_bits, n_lsb_bits=n_lsb)
            extraction_ber = compute_ber(payload[:len(recovered)], recovered)
            all_rows.append({
                "scheme": scheme_name,
                "metric_name": "extraction",
                "analysis": "extraction_ber_after_neuperm",
                "value": extraction_ber,
                "repeat_idx": repeat_idx,
                "model_name": model_name,
            })

            for metric_name in metrics:
                agg = compute_tracked_model_ber(sd_stego, model_name, metric_name, site_perms)
                all_rows.append({
                    "scheme": scheme_name,
                    "metric_name": metric_name,
                    "analysis": "canonical_attack_ber",
                    "value": agg["aggregate_weighted"],
                    "repeat_idx": repeat_idx,
                    "model_name": model_name,
                })

            del sd_stego_copy, sd_permuted, recorded_perms, recovered
            gc.collect()

        del sd_stego, payload
        gc.collect()

    return pd.DataFrame(all_rows)


def run_canonical_attack_model(
    model_name: str,
    metrics: list,
    n_repeats: int,
    run_countermeasure: bool,
    run_stego: bool,
    cm_epsilons: list,
    stego_schemes: list,
    payload_seed: int,
    seed: int,
) -> None:
    """Run every canonical-attack experiment for one model and write its CSVs."""
    log(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _seed_everything(seed)

    log("  Loading model...")
    t0 = time.time()
    sd = load_model_sd(model_name)
    log(f"  Loaded in {time.time() - t0:.1f}s")

    # 1. Tie analysis.
    log("  Running tie analysis...")
    t0 = time.time()
    df_ties = run_tie_analysis(sd, model_name, metrics)
    df_ties.to_csv(f"{RESULTS_DIR}/canonical_ties_{model_name}.csv", index=False)
    l1 = df_ties[df_ties["metric_name"] == "l1_norm"]["uniqueness_prob_exact"].mean()
    log(f"  Tie analysis done in {time.time() - t0:.1f}s (mean L1 uniqueness={l1:.4f})")

    # 2. Theoretical bounds.
    log("  Computing theoretical bounds...")
    df_theory = compute_theoretical_bounds(sd, model_name)
    df_theory.to_csv(f"{RESULTS_DIR}/canonical_theory_{model_name}.csv", index=False)

    # 3. Canonical attack BER.
    log(f"  Running canonical attack ({n_repeats} repeats)...")
    t0 = time.time()
    df_attack = run_canonical_attack(sd, model_name, metrics, n_repeats)
    df_attack.to_csv(f"{RESULTS_DIR}/canonical_attack_{model_name}.csv", index=False)
    log(f"  Attack done in {time.time() - t0:.1f}s")
    agg = df_attack[df_attack["site_kind"] == "aggregate"]
    for metric_name in metrics:
        m_data = agg[agg["metric_name"] == metric_name]
        log(f"    Aggregate BER ({metric_name}): "
            f"{m_data['ber'].mean():.4f} +/- {m_data['ber'].std():.4f}")

    # 4. Countermeasure (optional).
    if run_countermeasure:
        log("  Running countermeasure evaluation...")
        t0 = time.time()
        df_cm = run_countermeasure_eval(sd, model_name, metrics, n_repeats, cm_epsilons)
        df_cm.to_csv(f"{RESULTS_DIR}/canonical_countermeasure_{model_name}.csv", index=False)
        log(f"  Countermeasure done in {time.time() - t0:.1f}s")

    # 5. Stego-model analysis (optional).
    if run_stego:
        log("  Running stego-model analysis...")
        t0 = time.time()
        df_stego = run_stego_analysis(
            sd, model_name, metrics, n_repeats=min(n_repeats, 5),
            stego_schemes=stego_schemes, payload_seed=payload_seed,
        )
        df_stego.to_csv(f"{RESULTS_DIR}/canonical_stego_{model_name}.csv", index=False)
        log(f"  Stego analysis done in {time.time() - t0:.1f}s")
        for scheme in df_stego["scheme"].unique():
            sub = df_stego[df_stego["scheme"] == scheme]
            ext = sub[sub["analysis"] == "extraction_ber_after_neuperm"]
            can = sub[(sub["analysis"] == "canonical_attack_ber") & (sub["metric_name"] == "l1_norm")]
            tie = sub[(sub["analysis"] == "tie_analytical_ber") & (sub["metric_name"] == "l1_norm")]
            ext_mean = ext["value"].mean() if len(ext) > 0 else -1
            can_mean = can["value"].mean() if len(can) > 0 else -1
            tie_val = tie["value"].values[0] if len(tie) > 0 else -1
            log(f"    {scheme}: extraction_BER={ext_mean:.4f}, "
                f"canonical_BER(L1)={can_mean:.6f}, tie_BER(L1)={tie_val:.6f}")

    log(f"  Results saved to {RESULTS_DIR}/")


# ===========================================================================
# Sub-analysis 2: tie_corpus (from exp_canonical_tie_corpus.py)
# ===========================================================================

# CNN models with ImageNet-1K pretrained weights (excludes transformers,
# quantized variants, and excessively large downloads).
CORPUS = [
    "alexnet",
    "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d",
    "wide_resnet50_2", "wide_resnet101_2",
    "densenet121", "densenet161", "densenet169", "densenet201",
    "googlenet", "inception_v3",
    "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
    "mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3",
    "squeezenet1_0", "squeezenet1_1",
    "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5",
    "efficientnet_v2_s", "efficientnet_v2_m",
    "regnet_x_400mf", "regnet_x_800mf", "regnet_x_1_6gf", "regnet_x_3_2gf",
    "regnet_x_8gf", "regnet_x_16gf",
    "regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf", "regnet_y_3_2gf",
    "regnet_y_8gf", "regnet_y_16gf",
    "convnext_tiny", "convnext_small", "convnext_base",
]


def _corpus_analyze_layer(vals: torch.Tensor, tol: float) -> dict:
    """Count ties exactly and within tolerance for one layer's metric vector."""
    n = vals.numel()
    if n <= 1:
        return {
            "n": n, "n_unique_exact": n, "n_unique_tol": n,
            "n_tied_exact": 0, "n_tied_tol": 0,
            "n_groups_ge2_exact": 0, "max_group_exact": 1 if n else 0,
        }
    sorted_v, _ = torch.sort(vals)
    n_unique_exact = int((sorted_v[1:] != sorted_v[:-1]).sum().item()) + 1
    diffs_tol = (sorted_v[1:] - sorted_v[:-1]).abs() > tol
    n_unique_tol = int(diffs_tol.sum().item()) + 1
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
    return {
        "n": n,
        "n_unique_exact": n_unique_exact,
        "n_unique_tol": n_unique_tol,
        "n_tied_exact": n - n_unique_exact,
        "n_tied_tol": n - n_unique_tol,
        "n_groups_ge2_exact": groups_ge2,
        "max_group_exact": max_group,
    }


def _corpus_analyze_model(model_name: str, metrics: List[str], tol: float) -> dict:
    """Aggregate tie statistics over every 2D+ weight tensor in a model."""
    sd = load_pretrained_sd(model_name)
    agg = {m: {"n_total": 0, "n_unique_exact": 0, "n_unique_tol": 0,
               "n_layers": 0, "max_group": 0, "groups_ge2": 0} for m in metrics}
    for _, w in sd.items():
        if not torch.is_tensor(w) or not w.dtype.is_floating_point:
            continue
        if w.ndim < 2 or w.shape[0] < 2:
            continue
        for metric in metrics:
            vals = _channel_metric(w, metric)
            stats = _corpus_analyze_layer(vals, tol)
            a = agg[metric]
            a["n_layers"] += 1
            a["n_total"] += stats["n"]
            a["n_unique_exact"] += stats["n_unique_exact"]
            a["n_unique_tol"] += stats["n_unique_tol"]
            a["groups_ge2"] += stats["n_groups_ge2_exact"]
            a["max_group"] = max(a["max_group"], stats["max_group_exact"])
    del sd
    return agg


def run_tie_corpus(models: list, metrics: list, tol: float, csv_path: str) -> None:
    """Sweep the CNN corpus, measuring metric-uniqueness per model. Resumable."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        done_models = set(existing["model_name"].tolist())
        rows = existing.to_dict("records")
        log(f"Loaded {len(done_models)} existing models from {csv_path}")
    else:
        done_models = set()
        rows = []

    for model_name in models:
        if model_name in done_models:
            log(f"  SKIP {model_name} (already analyzed)")
            continue
        log(f"\n=== {model_name} ===")
        t0 = time.time()
        try:
            agg = _corpus_analyze_model(model_name, metrics, tol)
        except Exception as e:
            log(f"  ERROR: {e}")
            continue
        elapsed = time.time() - t0
        for metric, a in agg.items():
            n_total = a["n_total"]
            uniq_frac = a["n_unique_exact"] / n_total if n_total else 0.0
            uniq_frac_tol = a["n_unique_tol"] / n_total if n_total else 0.0
            rows.append({
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
                "tolerance": tol,
                "elapsed": elapsed,
            })
            if metric == "l1_norm":
                log(f"  L1: layers={a['n_layers']} units={n_total} "
                    f"unique_exact={uniq_frac*100:.2f}% "
                    f"unique_tol={uniq_frac_tol*100:.2f}% "
                    f"tied_groups={a['groups_ge2']} "
                    f"max_tie_group={a['max_group']} ({elapsed:.1f}s)")
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    log("\n" + "=" * 90 + "\nSummary (L1 metric)\n" + "=" * 90)
    l1 = df[df["metric"] == "l1_norm"].sort_values("uniq_frac_exact")
    log(f"{'model':<28s} {'units':>10s} {'uniq%':>8s} {'uniq(tol)%':>12s} "
        f"{'tied_groups':>12s} {'max_group':>10s}")
    for _, r in l1.iterrows():
        log(f"{r['model_name']:<28s} {int(r['n_total_units']):>10d} "
            f"{r['uniq_frac_exact']*100:>7.2f}% {r['uniq_frac_tol']*100:>11.2f}% "
            f"{int(r['n_groups_ge2']):>12d} {int(r['max_tie_group']):>10d}")
    log(f"\nCSV saved to {csv_path}")


# ===========================================================================
# Sub-analysis 3: tie_lsb (from exp_canonical_tie_lsb.py)
# ===========================================================================

# Small CNNs for the LSB-zeroing PoC.
SMALL_CORPUS = [
    "resnet18",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "squeezenet1_1",
    "mnasnet0_5",
    "shufflenet_v2_x0_5",
]

LSB_VALUES = [0, 1, 2]  # 0 = baseline (no zeroing)


def zero_lsbs(sd: dict, n_bits: int) -> dict:
    """Zero the low ``n_bits`` of every float32 parameter via view-as-int32 bitmask.

    float32 only; other dtypes pass through unchanged (this is a fp32 PoC).
    """
    if n_bits <= 0:
        return sd
    out = {}
    mask = ~((1 << n_bits) - 1)  # n=1 -> ...11111110, n=2 -> ...11111100
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


def _lsb_analyze_layer(vals: torch.Tensor) -> dict:
    """Exact-tie stats for one layer's metric vector."""
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


def _lsb_analyze_model(sd: dict, metric: str = "l1_norm") -> dict:
    """Aggregate exact-tie stats over every 2D+ weight tensor in a model."""
    agg = {"n_layers": 0, "n_total": 0, "n_unique_exact": 0,
           "n_groups_ge2": 0, "max_group": 0}
    for _, w in sd.items():
        if not torch.is_tensor(w) or not w.dtype.is_floating_point:
            continue
        if w.ndim < 2 or w.shape[0] < 2:
            continue
        vals = _channel_metric(w, metric)
        stats = _lsb_analyze_layer(vals)
        agg["n_layers"] += 1
        agg["n_total"] += stats["n"]
        agg["n_unique_exact"] += stats["n_unique_exact"]
        agg["n_groups_ge2"] += stats["n_groups_ge2"]
        agg["max_group"] = max(agg["max_group"], stats["max_group"])
    return agg


def run_tie_lsb(models: list, lsb_values: list, metric: str, csv_path: str) -> None:
    """Measure metric-uniqueness after zeroing the low LSBs of every param."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for model_name in models:
        log(f"\n=== {model_name} ===")
        t0 = time.time()
        try:
            sd_orig = load_pretrained_sd(model_name)
        except Exception as e:
            log(f"  ERROR loading: {e}")
            continue
        log(f"  loaded ({time.time()-t0:.1f}s)")

        for nb in lsb_values:
            sd = zero_lsbs(sd_orig, nb) if nb > 0 else sd_orig
            agg = _lsb_analyze_model(sd, metric)
            n_total = agg["n_total"]
            uniq = agg["n_unique_exact"] / n_total if n_total else 0.0
            tied = n_total - agg["n_unique_exact"]
            log(f"  LSB={nb}: units={n_total} unique={uniq*100:.4f}% "
                f"tied={tied} groups_ge2={agg['n_groups_ge2']} "
                f"max_group={agg['max_group']}")
            rows.append({
                "model_name": model_name,
                "metric": metric,
                "lsb_zeroed": nb,
                "n_layers": agg["n_layers"],
                "n_total": n_total,
                "n_unique_exact": agg["n_unique_exact"],
                "n_tied": tied,
                "uniq_frac": uniq,
                "n_groups_ge2": agg["n_groups_ge2"],
                "max_tie_group": agg["max_group"],
            })
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    log("\n" + "=" * 90 + "\nSummary (impact of zeroing LSBs)\n" + "=" * 90)
    log(f"{'model':<22s} {'LSB':>4s} {'units':>8s} {'uniq%':>10s} {'tied':>8s} "
        f"{'groups':>8s} {'max_grp':>8s}")
    for m in models:
        sub = df[df["model_name"] == m]
        for _, r in sub.iterrows():
            log(f"{r['model_name']:<22s} {int(r['lsb_zeroed']):>4d} "
                f"{int(r['n_total']):>8d} {r['uniq_frac']*100:>9.4f}% "
                f"{int(r['n_tied']):>8d} {int(r['n_groups_ge2']):>8d} "
                f"{int(r['max_tie_group']):>8d}")
        log("")
    log(f"CSV saved to {csv_path}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    # --- run configuration ---
    # Which sub-analysis to run: "canonical_attack" | "tie_corpus" | "tie_lsb".
    ANALYSIS = "canonical_attack"

    SEED = 42
    PAYLOAD_SEED = 42

    # -- canonical_attack config --
    ATTACK_MODELS = [
        "vgg11",
        "vgg16",
        "resnet50",
        "resnet101",
        "densenet121",
        "llama-3.2-1b",
        "qwen2.5-1.5b",
    ]
    ATTACK_METRICS = ["l1_norm", "l2_norm", "bias_value", "variance"]
    ATTACK_N_REPEATS = 10
    RUN_COUNTERMEASURE = True
    RUN_STEGO = True
    COUNTERMEASURE_EPSILONS = [1e-7, 1e-6, 1e-5]
    # None -> all of STEGO_SCHEMES (LSB depths 1,2,4,8); else a subset list.
    STEGO_SCHEMES_TO_RUN = None

    # -- tie_corpus config --
    CORPUS_MODELS = CORPUS
    CORPUS_METRICS = ["l1_norm", "l2_norm", "variance"]
    CORPUS_TOL = 1e-6
    CORPUS_CSV = os.path.join(RESULTS_DIR, "canonical_ties_corpus.csv")

    # -- tie_lsb config --
    LSB_MODELS = SMALL_CORPUS
    LSB_LSB_VALUES = LSB_VALUES
    LSB_METRIC = "l1_norm"
    LSB_CSV = os.path.join(RESULTS_DIR, "canonical_ties_lsb.csv")
    # -------------------------

    if ANALYSIS == "canonical_attack":
        for model_name in ATTACK_MODELS:
            run_canonical_attack_model(
                model_name=model_name,
                metrics=ATTACK_METRICS,
                n_repeats=ATTACK_N_REPEATS,
                run_countermeasure=RUN_COUNTERMEASURE,
                run_stego=RUN_STEGO,
                cm_epsilons=COUNTERMEASURE_EPSILONS,
                stego_schemes=STEGO_SCHEMES_TO_RUN,
                payload_seed=PAYLOAD_SEED,
                seed=SEED,
            )
        log("\nAll canonical-attack experiments complete.")
    elif ANALYSIS == "tie_corpus":
        _seed_everything(SEED)
        run_tie_corpus(CORPUS_MODELS, CORPUS_METRICS, CORPUS_TOL, CORPUS_CSV)
    elif ANALYSIS == "tie_lsb":
        _seed_everything(SEED)
        run_tie_lsb(LSB_MODELS, LSB_LSB_VALUES, LSB_METRIC, LSB_CSV)
    else:
        raise ValueError(
            f"Unknown ANALYSIS={ANALYSIS!r}; "
            f"expected 'canonical_attack', 'tie_corpus', or 'tie_lsb'"
        )
