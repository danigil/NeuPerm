"""X2 component: Spread-spectrum payload (independent MaleficNet-class ablation) destruction
under NeuPerm for Llama-3.2-1B and Qwen2.5-1.5B.

Reports extraction BER before and after NeuPerm. Independent reimpl of the
spread-spectrum embed/extract in steganography.py; framed as an ablation, not a
second published attack.
"""
import copy, os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoModelForCausalLM
from neu_perm.config import RESULTS_DIR
from neu_perm.perm import permute_model
from neu_perm.steganography import (
    spread_spectrum_embed, spread_spectrum_extract, compute_ber, generate_payload,
)

MODELS = [
    ("llama-3.2-1b", "meta-llama/Llama-3.2-1B-Instruct"),
    ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct"),
]
N_BITS = 1024     # payload size in bits
AMPLITUDE = 1e-4
N_REPEATS = 5
CSV_PATH = os.path.join(RESULTS_DIR, "spread_spectrum_llm.csv")

rows = []
for model_key, hf_id in MODELS:
    print(f"=== {model_key} ===", flush=True)
    m = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16)
    sd_orig = copy.deepcopy(m.cpu().state_dict())
    del m
    torch.cuda.empty_cache()

    payload = generate_payload(N_BITS, seed=42)
    sd_stego = spread_spectrum_embed(sd_orig, model_key, payload,
                                     amplitude=AMPLITUDE, seed=123, inplace=False)

    # clean extraction (sanity)
    rec_clean = spread_spectrum_extract(sd_stego, model_key, sd_orig, N_BITS)
    ber_clean = compute_ber(payload, rec_clean)
    rows.append({"model": model_key, "n_bits": N_BITS, "amplitude": AMPLITUDE,
                 "analysis": "ber_clean", "repeat": -1, "ber": ber_clean})
    print(f"  clean BER = {ber_clean:.4f}", flush=True)

    # post-NeuPerm extraction
    for rep in range(N_REPEATS):
        torch.manual_seed(rep)
        sd_p = permute_model(model_key, copy.deepcopy(sd_stego), inplace=True)
        rec_p = spread_spectrum_extract(sd_p, model_key, sd_orig, N_BITS)
        ber_p = compute_ber(payload, rec_p)
        rows.append({"model": model_key, "n_bits": N_BITS, "amplitude": AMPLITUDE,
                     "analysis": "ber_after_neuperm", "repeat": rep, "ber": ber_p})
        print(f"  rep {rep}: BER after NeuPerm = {ber_p:.4f}", flush=True)
        del sd_p

    del sd_orig, sd_stego
    torch.cuda.empty_cache()

pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
print(f"wrote {CSV_PATH}", flush=True)
