"""Experiment: Post-Training Quantization (PTQ) vs NeuPerm — Payload BER Comparison.

Measures how effectively PTQ at various bit-widths destroys steganographic
payloads embedded in model weights, and compares against NeuPerm.

For each model and LSB embedding depth (1, 2, 4, 8 bits):
  1. Embed a random payload across all float parameters.
  2. Apply quantize-dequantize at 8-bit and 4-bit (per-tensor and per-channel).
  3. Extract payload and measure BER.
  4. Apply NeuPerm and measure extraction BER for comparison.

Does NOT require ImageNet or any dataset — operates on state_dicts only.
Results saved to results/quantization_ber_{model}.csv.
"""

import argparse
import copy
import gc
import os
import sys
import time
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

from neu_perm.perm import permute_model
from neu_perm.quantization import quantize_dequantize_sd
from neu_perm.steganography import (
    compute_ber,
    generate_payload,
    lsb_embed,
    lsb_extract,
    total_embeddable_params,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAMES = ["vgg11", "vgg16", "resnet50", "resnet101", "densenet121"]

LSB_DEPTHS = [1, 2, 4, 8]

QUANT_CONFIGS = [
    ("ptq_8bit_pertensor", {"n_bits": 8, "per_channel": False}),
    ("ptq_8bit_perchannel", {"n_bits": 8, "per_channel": True}),
    ("ptq_4bit_pertensor", {"n_bits": 4, "per_channel": False}),
    ("ptq_4bit_perchannel", {"n_bits": 4, "per_channel": True}),
]

N_REPEATS = 10  # for NeuPerm (stochastic); PTQ is deterministic


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_sd(model_name: str) -> dict:
    """Load pretrained state_dict for a model."""
    if model_name in ["vgg11", "vgg16", "resnet50", "resnet101", "densenet121"]:
        weights = torchvision.models.get_model_weights(model_name).DEFAULT
        model = torchvision.models.get_model(model_name, weights=weights)
        return model.state_dict()
    elif model_name == "llama-3.2-1b":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B", torch_dtype=torch.float16,
        )
        return model.state_dict()
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_quantization_ber(
    sd: dict,
    model_name: str,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Measure payload BER after PTQ and NeuPerm for all LSB depths."""
    all_rows = []
    n_total_params = total_embeddable_params(sd)

    for n_lsb in LSB_DEPTHS:
        scheme_name = f"lsb_{n_lsb}bit"
        n_payload_bits = n_total_params * n_lsb
        payload = generate_payload(n_payload_bits, seed=42)

        print(f"  {scheme_name}: {n_payload_bits:,} bits into {n_total_params:,} params")

        # Embed payload
        sd_stego = lsb_embed(sd, payload, n_lsb_bits=n_lsb, inplace=False)

        # Sanity: clean extraction BER should be 0
        recovered_clean = lsb_extract(sd_stego, n_payload_bits, n_lsb_bits=n_lsb)
        clean_ber = compute_ber(payload, recovered_clean)
        all_rows.append({
            "model_name": model_name,
            "method": "clean_extraction",
            "scheme": scheme_name,
            "n_lsb_bits": n_lsb,
            "ber": clean_ber,
            "time": 0.0,
            "repeat_idx": -1,
        })
        if clean_ber > 0.0:
            print(f"    WARNING: clean BER = {clean_ber:.6f}")

        # --- PTQ baselines (deterministic — 1 repeat each) ---
        for config_name, config_kwargs in QUANT_CONFIGS:
            t0 = time.time()
            sd_q = quantize_dequantize_sd(
                sd_stego, inplace=False, **config_kwargs,
            )
            elapsed = time.time() - t0

            recovered = lsb_extract(sd_q, n_payload_bits, n_lsb_bits=n_lsb)
            ber = compute_ber(payload, recovered)

            all_rows.append({
                "model_name": model_name,
                "method": config_name,
                "scheme": scheme_name,
                "n_lsb_bits": n_lsb,
                "ber": ber,
                "time": elapsed,
                "repeat_idx": 0,
            })
            print(f"    {config_name}: BER={ber:.4f} ({elapsed:.2f}s)")

            del sd_q
            gc.collect()

        # --- NeuPerm (stochastic — n_repeats) ---
        for repeat_idx in range(n_repeats):
            sd_stego_copy = copy.deepcopy(sd_stego)
            t0 = time.time()
            sd_perm = permute_model(model_name, sd_stego_copy, inplace=True)
            elapsed = time.time() - t0

            recovered = lsb_extract(sd_perm, n_payload_bits, n_lsb_bits=n_lsb)
            ber = compute_ber(payload, recovered)

            all_rows.append({
                "model_name": model_name,
                "method": "neuperm",
                "scheme": scheme_name,
                "n_lsb_bits": n_lsb,
                "ber": ber,
                "time": elapsed,
                "repeat_idx": repeat_idx,
            })
            if repeat_idx == 0:
                print(f"    neuperm: BER={ber:.4f} ({elapsed:.2f}s) [repeat 0/{n_repeats}]")

            del sd_stego_copy, sd_perm
            gc.collect()

        del sd_stego, payload
        gc.collect()

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="PTQ vs NeuPerm payload BER comparison",
    )
    parser.add_argument(
        "--models", nargs="+", default=MODEL_NAMES,
        help="Model names to evaluate",
    )
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        t0 = time.time()
        sd = load_model_sd(model_name)
        print(f"  Loaded in {time.time() - t0:.1f}s")

        df = run_quantization_ber(sd, model_name, n_repeats=args.n_repeats)
        out_path = f"{RESULTS_DIR}/quantization_ber_{model_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved to {out_path}")

        # Print summary
        print(f"\n  Summary (mean BER across LSB depths):")
        for method in df["method"].unique():
            if method == "clean_extraction":
                continue
            sub = df[df["method"] == method]
            mean_ber = sub["ber"].mean()
            mean_time = sub["time"].mean()
            print(f"    {method:25s}: BER={mean_ber:.4f}, time={mean_time:.3f}s")

        del sd
        gc.collect()

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
