"""
Experiment: Canonical-Ordering Adaptive Attack on NeuPerm

Evaluates whether an adversary can undo NeuPerm's permutations by establishing
a canonical ordering of neurons (e.g., sorting by L1-norm) before and after
permutation.

For each model:
  1. Tie analysis: measure how many neurons share metric values (ambiguous rank).
  2. Attack simulation: permute with NeuPerm, attempt canonical recovery, measure BER.
  3. (Optional) Countermeasure: add tie-breaking perturbation, re-measure BER.

Results are saved to results/ as CSVs.
"""

import argparse
import copy
import gc
import os
import sys
import time
from typing import Literal, Optional

import types as _types

import pandas as pd
import torch
import torchvision

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Bypass IMAGENET12_ROOT validation — this experiment only needs pretrained weights
_cfg_stub = _types.ModuleType("neu_perm.config")
_cfg_stub.REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_cfg_stub.RESULTS_DIR = os.path.join(_cfg_stub.REPO_ROOT, "results")
_cfg_stub.IMAGENET12_ROOT = ""
sys.modules["neu_perm.config"] = _cfg_stub
RESULTS_DIR = _cfg_stub.RESULTS_DIR

from neu_perm.canonical import (
    METRIC_NAMES,
    get_permutable_sites,
    layer_tie_analysis,
    theoretical_recovery_probability,
    theoretical_uniqueness_bound,
    tie_group_sizes,
    compute_neuron_metric,
)
from neu_perm.countermeasures import tie_breaking_perturbation
from neu_perm.perm import permute_model
from neu_perm.steganography import (
    compute_analytical_model_ber,
    compute_ber,
    compute_detailed_tracked_ber,
    compute_tracked_model_ber,
    generate_payload,
    lsb_embed,
    lsb_extract,
    permute_model_tracked,
    total_embeddable_params,
    _map_perm_indices_to_sites,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAMES = [
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

TORCHVISION_MODELS = (
    "vgg11",
    "vgg16",
    "resnet50",
    "resnet101",
    "densenet121",
    "efficientnet_b0",
    "efficientnet_b4",
    "mobilenet_v2",
    "mobilenet_v3_small",
)

METRICS = ["l1_norm", "l2_norm", "bias_value", "variance"]

N_REPEATS = 10

COUNTERMEASURE_EPSILONS = [1e-7, 1e-6, 1e-5]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_sd(model_name: str, device: str = "cpu") -> dict:
    """Load a pretrained model and return its state_dict (on CPU)."""
    if model_name in TORCHVISION_MODELS:
        weights = torchvision.models.get_model_weights(model_name).DEFAULT
        model = torchvision.models.get_model(model_name, weights=weights)
        model = model.to("cpu").eval()
        return copy.deepcopy(model.state_dict())
    elif model_name in ("llama-3.2-1b", "qwen2.5-1.5b"):
        from transformers import AutoModelForCausalLM

        hf_id = {
            "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
            "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        }[model_name]
        model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16)
        model = model.to("cpu").eval()
        return copy.deepcopy(model.state_dict())
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Experiment 1: Tie analysis
# ---------------------------------------------------------------------------


def run_tie_analysis(sd: dict, model_name: str, metrics: list) -> pd.DataFrame:
    """Run tie analysis across all metrics for a model."""
    dfs = []
    for metric_name in metrics:
        df = layer_tie_analysis(sd, model_name, metric_name)
        dfs.append(df)
    result = pd.concat(dfs, ignore_index=True)
    result["model_name"] = model_name
    return result


# ---------------------------------------------------------------------------
# Experiment 2: Canonical attack BER
# ---------------------------------------------------------------------------


def run_canonical_attack(
    sd: dict,
    model_name: str,
    metrics: list,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Run the canonical-ordering attack n_repeats times and measure BER.

    Uses tracked permutation to compute exact BER per repeat, plus
    analytical expected BER (deterministic, same every repeat).
    """
    all_rows = []

    # Analytical expected BER (same for all repeats)
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
        # Deep copy and permute with tracking
        sd_copy = copy.deepcopy(sd)
        sd_permuted, recorded_perms = permute_model_tracked(
            model_name, sd_copy, inplace=True
        )
        site_perms = _map_perm_indices_to_sites(model_name, recorded_perms)

        for metric_name in metrics:
            rows = compute_detailed_tracked_ber(
                sd, model_name, metric_name, site_perms
            )
            for row in rows:
                row["repeat_idx"] = repeat_idx
                row["model_name"] = model_name
                row["method"] = "canonical_attack"
            all_rows.extend(rows)

            # Aggregate
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

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Experiment 3: Countermeasure evaluation
# ---------------------------------------------------------------------------


def run_countermeasure_eval(
    sd: dict,
    model_name: str,
    metrics: list,
    n_repeats: int = 10,
    cm_epsilons: list = None,
) -> pd.DataFrame:
    """Evaluate tie-breaking countermeasure on top of NeuPerm.

    After NeuPerm + countermeasure, re-run canonical attack to check if
    the perturbation further disrupts the adversary's recovery.
    Uses analytical BER on the perturbed sd (tie structure changes after perturbation).
    """
    if cm_epsilons is None:
        cm_epsilons = COUNTERMEASURE_EPSILONS

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

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Experiment 4: Stego-model analysis (embed payload, then run attack)
# ---------------------------------------------------------------------------


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
    n_repeats: int = 5,
    stego_schemes: list = None,
) -> pd.DataFrame:
    """Embed steganographic payloads into ALL model parameters, then analyse.

    For each LSB embedding depth (1, 2, 4, 8 bits):
      1. Embed a random payload across every float parameter in the state dict.
      2. Verify round-trip extraction on the clean stego-model.
      3. Run tie analysis on the stego-model (does embedding change tie structure?).
      4. Apply NeuPerm, attempt extraction → measure extraction BER.
      5. Run canonical ordering attack on the stego-model → measure canonical BER.

    ``stego_schemes`` selects which LSB depths to run (defaults to all of
    ``STEGO_SCHEMES``); pass a subset to restrict e.g. to the 1-bit scheme.
    """
    all_rows = []
    n_total_params = total_embeddable_params(sd)
    schemes = stego_schemes if stego_schemes is not None else STEGO_SCHEMES

    for scheme_name, scheme_kwargs in schemes:
        n_lsb = scheme_kwargs["n_lsb_bits"]
        # Payload that saturates the model's capacity
        n_payload_bits = n_total_params * n_lsb
        payload = generate_payload(n_payload_bits, seed=42)

        print(f"    {scheme_name}: embedding {n_payload_bits:,} bits "
              f"({n_payload_bits / 8 / 1024:.1f} KB) into {n_total_params:,} params...")

        # Embed into ALL parameters
        sd_stego = lsb_embed(sd, payload, n_lsb_bits=n_lsb, inplace=False)

        # Verify round-trip on clean stego model (sanity check)
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
            print(f"      WARNING: clean extraction BER = {clean_ber:.6f} (should be 0)")

        # Tie analysis on stego model
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

        # Extraction after NeuPerm (does NeuPerm break the payload?)
        for repeat_idx in range(n_repeats):
            sd_stego_copy = copy.deepcopy(sd_stego)
            sd_permuted, recorded_perms = permute_model_tracked(
                model_name, sd_stego_copy, inplace=True
            )
            site_perms = _map_perm_indices_to_sites(model_name, recorded_perms)

            # Extraction BER after NeuPerm
            recovered = lsb_extract(
                sd_permuted, n_payload_bits, n_lsb_bits=n_lsb,
            )
            extraction_ber = compute_ber(payload[:len(recovered)], recovered)

            all_rows.append({
                "scheme": scheme_name,
                "metric_name": "extraction",
                "analysis": "extraction_ber_after_neuperm",
                "value": extraction_ber,
                "repeat_idx": repeat_idx,
                "model_name": model_name,
            })

            # Canonical attack BER on stego model
            for metric_name in metrics:
                agg = compute_tracked_model_ber(
                    sd_stego, model_name, metric_name, site_perms
                )
                all_rows.append({
                    "scheme": scheme_name,
                    "metric_name": metric_name,
                    "analysis": "canonical_attack_ber",
                    "value": agg["aggregate_weighted"],
                    "repeat_idx": repeat_idx,
                    "model_name": model_name,
                })

            # Free memory from this repeat
            del sd_stego_copy, sd_permuted, recorded_perms, recovered
            gc.collect()

        # Free stego model and payload before next scheme
        del sd_stego, payload
        gc.collect()

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Theoretical bounds
# ---------------------------------------------------------------------------


def compute_theoretical_bounds(sd: dict, model_name: str) -> pd.DataFrame:
    """Compute theoretical uniqueness bounds per site."""
    sites = get_permutable_sites(model_name)
    rows = []
    for idx, site in enumerate(sites):
        # Get the number of permutable units
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_single_model(
    model_name: str,
    metrics: list,
    n_repeats: int,
    run_countermeasure: bool = True,
    run_stego: bool = True,
    cm_epsilons: list = None,
    stego_schemes: list = None,
    seed: Optional[int] = None,
    results_dir: Optional[str] = None,
):
    """Run all canonical attack experiments for a single model.

    ``seed``, when given, seeds the global torch RNG once before any
    permutation is drawn, making the whole per-model run — every repeat of the
    attack, the countermeasure sweep and the stego analysis — exactly
    reproducible from the command line plus that one number. Left as None the
    run is unseeded, which is how the pre-existing result CSVs were produced.

    ``results_dir`` overrides where the CSVs are written (default:
    ``RESULTS_DIR``), so a re-run can be directed at ``exports/`` without
    overwriting the published artifacts in ``results/``.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    if seed is not None:
        torch.manual_seed(seed)
        print(f"  Seed: {seed} (torch.manual_seed, set once before any permutation)")
    else:
        print("  Seed: none (unseeded run)")

    os.makedirs(results_dir, exist_ok=True)

    # Load model
    print(f"  Loading model...")
    t0 = time.time()
    sd = load_model_sd(model_name)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # 1. Tie analysis
    print(f"  Running tie analysis...")
    t0 = time.time()
    df_ties = run_tie_analysis(sd, model_name, metrics)
    df_ties.to_csv(f"{results_dir}/canonical_ties_{model_name}.csv", index=False)
    print(f"  Tie analysis done in {time.time() - t0:.1f}s")
    print(f"    Mean uniqueness (exact, L1): {df_ties[df_ties['metric_name']=='l1_norm']['uniqueness_prob_exact'].mean():.4f}")

    # 2. Theoretical bounds
    print(f"  Computing theoretical bounds...")
    df_theory = compute_theoretical_bounds(sd, model_name)
    df_theory.to_csv(f"{results_dir}/canonical_theory_{model_name}.csv", index=False)

    # 3. Canonical attack BER
    print(f"  Running canonical attack ({n_repeats} repeats)...")
    t0 = time.time()
    df_attack = run_canonical_attack(sd, model_name, metrics, n_repeats)
    df_attack.to_csv(f"{results_dir}/canonical_attack_{model_name}.csv", index=False)
    print(f"  Attack done in {time.time() - t0:.1f}s")

    # Print aggregate BER summary
    agg = df_attack[df_attack["site_kind"] == "aggregate"]
    for metric_name in metrics:
        m_data = agg[agg["metric_name"] == metric_name]
        mean_ber = m_data["ber"].mean()
        std_ber = m_data["ber"].std()
        print(f"    Aggregate BER ({metric_name}): {mean_ber:.4f} +/- {std_ber:.4f}")

    # 4. Countermeasure (optional)
    if run_countermeasure:
        print(f"  Running countermeasure evaluation...")
        t0 = time.time()
        df_cm = run_countermeasure_eval(sd, model_name, metrics, n_repeats, cm_epsilons)
        df_cm.to_csv(f"{results_dir}/canonical_countermeasure_{model_name}.csv", index=False)
        print(f"  Countermeasure done in {time.time() - t0:.1f}s")

    # 5. Stego-model analysis
    if run_stego:
        print(f"  Running stego-model analysis...")
        t0 = time.time()
        df_stego = run_stego_analysis(
            sd, model_name, metrics, n_repeats=min(n_repeats, 5),
            stego_schemes=stego_schemes,
        )
        df_stego.to_csv(f"{results_dir}/canonical_stego_{model_name}.csv", index=False)
        print(f"  Stego analysis done in {time.time() - t0:.1f}s")

        # Print summary
        for scheme in df_stego["scheme"].unique():
            sub = df_stego[df_stego["scheme"] == scheme]
            ext = sub[sub["analysis"] == "extraction_ber_after_neuperm"]
            can = sub[(sub["analysis"] == "canonical_attack_ber") & (sub["metric_name"] == "l1_norm")]
            tie = sub[(sub["analysis"] == "tie_analytical_ber") & (sub["metric_name"] == "l1_norm")]
            ext_mean = ext["value"].mean() if len(ext) > 0 else -1
            can_mean = can["value"].mean() if len(can) > 0 else -1
            tie_val = tie["value"].values[0] if len(tie) > 0 else -1
            print(f"    {scheme}: extraction_BER={ext_mean:.4f}, canonical_BER(L1)={can_mean:.6f}, tie_BER(L1)={tie_val:.6f}")

    print(f"  Results saved to {results_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Canonical-ordering attack experiment")
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_NAMES,
        help="Model names to evaluate",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=METRICS,
        help="Canonical ordering metrics to test",
    )
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed the torch RNG once per model before any permutation is "
        "drawn, making the run reproducible. Omitted = unseeded (the default, "
        "and how the pre-existing result CSVs were produced).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=f"Directory to write the CSVs to (default: {RESULTS_DIR}).",
    )
    parser.add_argument("--no-countermeasure", action="store_true")
    parser.add_argument("--no-stego", action="store_true")
    parser.add_argument(
        "--lsb-bits",
        nargs="+",
        type=int,
        default=None,
        help="LSB depths to run in the stego analysis (default: all of "
        "STEGO_SCHEMES = 1 2 4 8). e.g. --lsb-bits 1 restricts to the 1-bit scheme.",
    )
    args = parser.parse_args()

    stego_schemes = None
    if args.lsb_bits is not None:
        wanted = set(args.lsb_bits)
        stego_schemes = [s for s in STEGO_SCHEMES if s[1]["n_lsb_bits"] in wanted]
        if not stego_schemes:
            parser.error(f"--lsb-bits {args.lsb_bits} matched no scheme in STEGO_SCHEMES")

    for model_name in args.models:
        run_single_model(
            model_name=model_name,
            metrics=args.metrics,
            n_repeats=args.n_repeats,
            run_countermeasure=not args.no_countermeasure,
            run_stego=not args.no_stego,
            stego_schemes=stego_schemes,
            seed=args.seed,
            results_dir=args.results_dir,
        )

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
