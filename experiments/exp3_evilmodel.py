"""Experiment 3 (EvilModel): NeuPerm destroys byte-exact weight steganography.

Reproduces Table `tab:evilmodel` — a second weight-steganography attack besides
MaleficNet. Where MaleficNet uses CDMA spread-spectrum + LDPC error-correction
(measured by SNR), EvilModel 2.0 (Wang et al., Computers & Security 120:102807,
2022) is byte-exact substitution with NO error-correction — the opposite end of
the attack spectrum, so NeuPerm gives a hard binary result.

For each fp32 CNN:
  1. embed a benign random byte payload (default ~25% of capacity) by EvilModel
     half substitution (keep the high 2 bytes of each float32, overwrite the low
     2), storing len+SHA-256 in the carrier's bias (EvilModel integrity rule);
  2. extract PRE-NeuPerm  -> expect byte-exact, hash_ok=True, BER=0 (succeeds);
  3. apply NeuPerm (permute_model), N seeded repeats;
  4. extract POST-NeuPerm -> expect hash_ok=False, BER>0 (destroyed).

Metrics: SHA-256 pass/fail (the headline binary), bit error rate (BER), and the
embed perturbation (max/mean |weight delta|) as a host-accuracy proxy without an
ImageNet eval (NeuPerm's accuracy-invariance is established in exp1; half
substitution's value perturbation is < 2^-7 relative).

Payload is benign random bytes (generate_payload_bytes) — never real malware.

Configure the run in the `__main__` block at the bottom (model list, fraction,
x_bytes, repeats, seed), then:

    python experiments/exp3_evilmodel.py

Each row is one model x condition; results are written to
`<RESULTS_DIR>/evilmodel_neuperm.csv`.
"""
import copy
import os
import time
import urllib.parse
import warnings

import numpy as np
import pandas as pd
import torch
import torchvision

from neu_perm.config import RESULTS_DIR
import neu_perm.perm as perm
import neu_perm.steganography as st

warnings.filterwarnings("ignore")


def log(msg):
    print(msg, flush=True)


def load_pretrained_sd(model_name: str) -> dict:
    """Load a pretrained torchvision model state_dict as float32."""
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
    sd = copy.deepcopy(model.state_dict())
    return {k: (v.float() if torch.is_tensor(v) and v.dtype != torch.float32 else v)
            for k, v in sd.items()}


def embed_perturbation(sd_clean: dict, sd_stego: dict, model_name: str) -> dict:
    """Max/mean abs weight delta over carrier weights (host-accuracy proxy)."""
    keys = st._evilmodel_carrier_weight_keys(model_name)
    deltas = [(sd_stego[k] - sd_clean[k]).abs() for k in keys]
    flat = torch.cat([d.reshape(-1) for d in deltas])
    return {"perturb_max": float(flat.max()), "perturb_mean": float(flat.mean())}


def run_model(model_name: str, fraction: float, x_bytes: int, repeats: int,
              seed: int) -> dict:
    """Embed -> extract (pre) -> NeuPerm -> extract (post) for one model."""
    t0 = time.time()
    sd = load_pretrained_sd(model_name)
    log(f"  loaded ({time.time()-t0:.1f}s)")

    cap = st.evilmodel_capacity_bytes(sd, model_name, x_bytes)
    n_payload = int(fraction * cap)
    payload = st.generate_payload_bytes(n_payload, seed=seed)  # benign random bytes
    payload_bits = st.bytes_to_bits(payload)
    log(f"  capacity={cap/1e6:.1f} MB  payload={n_payload/1e6:.1f} MB "
        f"({fraction:.0%})  x_bytes={x_bytes}")

    sd_stego = st.evilmodel_embed(sd, model_name, payload, x_bytes=x_bytes, inplace=False)
    pert = embed_perturbation(sd, sd_stego, model_name)

    # PRE-NeuPerm: expect byte-exact recovery.
    pl_pre, ok_pre = st.evilmodel_extract(sd_stego, model_name, x_bytes=x_bytes)
    raw_pre = st.evilmodel_extract_raw(sd_stego, model_name, n_payload, x_bytes=x_bytes)
    ber_pre = float(np.mean(st.bytes_to_bits(raw_pre) != payload_bits))
    log(f"  PRE : hash_ok={ok_pre} byte_exact={pl_pre==payload} BER={ber_pre:.4f}")

    # POST-NeuPerm: seeded repeats, expect destroyed extraction.
    post_ok, post_ber = [], []
    for r in range(repeats):
        torch.manual_seed(seed + r)
        np.random.seed(seed + r)
        sd_perm = perm.permute_model(model_name, copy.deepcopy(sd_stego), inplace=True)
        _, ok = st.evilmodel_extract(sd_perm, model_name, x_bytes=x_bytes)
        raw = st.evilmodel_extract_raw(sd_perm, model_name, n_payload, x_bytes=x_bytes)
        post_ok.append(bool(ok))
        post_ber.append(float(np.mean(st.bytes_to_bits(raw) != payload_bits)))
    log(f"  POST: hash_ok_rate={np.mean(post_ok):.2f} "
        f"BER={np.mean(post_ber):.4f}+-{np.std(post_ber):.4f}")

    return {
        "model_name": model_name,
        "x_bytes": x_bytes,
        "scheme": {2: "half_substitution", 3: "msb_reservation"}[x_bytes],
        "capacity_bytes": cap,
        "payload_bytes": n_payload,
        "embedding_rate": fraction,
        "perturb_max": pert["perturb_max"],
        "perturb_mean": pert["perturb_mean"],
        "pre_hash_ok": ok_pre,
        "pre_byte_exact": pl_pre == payload,
        "pre_ber": ber_pre,
        "post_hash_ok_rate": float(np.mean(post_ok)),
        "post_ber_mean": float(np.mean(post_ber)),
        "post_ber_std": float(np.std(post_ber)),
        "repeats": repeats,
        "seed": seed,
    }


if __name__ == "__main__":
    # --- run configuration ---
    model_names = ["vgg11"]            # CNNs to attack (fp32 torchvision models)
    fraction = 0.25                    # payload size as a fraction of capacity
    x_bytes_list = [2]                 # 2=half substitution, 3=MSB reservation
    repeats = 5                        # seeded NeuPerm repeats for the POST stats
    seed = 7                           # base seed (payload + per-repeat permutation)
    # -------------------------

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "evilmodel_neuperm.csv")

    rows = []
    for model_name in model_names:
        for x_bytes in x_bytes_list:
            log(f"\n=== {model_name} (x_bytes={x_bytes}) ===")
            try:
                rows.append(run_model(model_name, fraction, x_bytes, repeats, seed))
            except Exception as e:
                log(f"  ERROR: {e}")
            pd.DataFrame(rows).to_csv(csv_path, index=False)  # checkpoint each model

    log("\n" + "=" * 80)
    log(f"{'model':<14s} {'scheme':<18s} {'pre_ok':>7s} {'pre_BER':>8s} "
        f"{'post_ok':>8s} {'post_BER':>9s}")
    for r in rows:
        log(f"{r['model_name']:<14s} {r['scheme']:<18s} {str(r['pre_hash_ok']):>7s} "
            f"{r['pre_ber']:>8.4f} {r['post_hash_ok_rate']:>8.2f} {r['post_ber_mean']:>9.4f}")
    log(f"\nCSV saved to {csv_path}")
