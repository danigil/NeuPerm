"""EvilModel (byte-substitution) vs NeuPerm — inject -> extract -> verify PoC.

Second weight-steganography attack besides MaleficNet, showing NeuPerm
generalizes beyond MaleficNet. Where MaleficNet uses CDMA
spread-spectrum + LDPC error-correction (measured by SNR), EvilModel 2.0
(Wang et al., Computers & Security 120:102807, 2022) is byte-exact substitution
with NO error-correction. This is the opposite end of the attack spectrum:
NeuPerm should destroy it completely, giving a HARD binary result.

For each fp32 CNN:
  1. embed a benign byte payload (default 25% of capacity) by half substitution
     (keep high 2 bytes of each float32, overwrite low 2), len+SHA-256 in the
     carrier's bias (EvilModel's integrity convention);
  2. extract PRE-NeuPerm -> expect byte-exact, hash_ok=True, BER=0 (succeeds);
  3. apply NeuPerm (permute_model), N seeded repeats;
  4. extract POST-NeuPerm -> expect hash_ok=False, BER>0 (destroyed).

Metrics: SHA-256 pass/fail (the headline binary), bit error rate (BER), and the
embed perturbation (max/mean |weight delta|) as a proxy for host-accuracy impact
without needing an ImageNet eval (NeuPerm's accuracy-invariance is established in
exp1; half substitution's value perturbation is < 2^-7 relative).

Usage:
  python experiments/exp_evilmodel_neuperm.py --models vgg11 --fraction 0.25 --repeats 5
"""
import argparse
import copy
import os
import sys
import time
import urllib.parse
import warnings

import numpy as np
import pandas as pd
import torch
import torchvision

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub config so the module imports without an IMAGENET12_ROOT.
import types as _types
_cfg_stub = _types.ModuleType("neu_perm.config")
_cfg_stub.REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_cfg_stub.RESULTS_DIR = os.path.join(_cfg_stub.REPO_ROOT, "results")
_cfg_stub.IMAGENET12_ROOT = ""
sys.modules["neu_perm.config"] = _cfg_stub
RESULTS_DIR = _cfg_stub.RESULTS_DIR

import neu_perm.perm as perm
import neu_perm.steganography as st


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
    t0 = time.time()
    sd = load_pretrained_sd(model_name)
    log(f"  loaded ({time.time()-t0:.1f}s)")

    cap = st.evilmodel_capacity_bytes(sd, model_name, x_bytes)
    n_payload = int(fraction * cap)
    payload = st.generate_payload_bytes(n_payload, seed=seed)
    payload_bits = st.bytes_to_bits(payload)
    log(f"  capacity={cap/1e6:.1f} MB  payload={n_payload/1e6:.1f} MB "
        f"({fraction:.0%})  x_bytes={x_bytes}")

    sd_stego = st.evilmodel_embed(sd, model_name, payload, x_bytes=x_bytes, inplace=False)
    pert = embed_perturbation(sd, sd_stego, model_name)

    # PRE-NeuPerm
    pl_pre, ok_pre = st.evilmodel_extract(sd_stego, model_name, x_bytes=x_bytes)
    raw_pre = st.evilmodel_extract_raw(sd_stego, model_name, n_payload, x_bytes=x_bytes)
    ber_pre = float(np.mean(st.bytes_to_bits(raw_pre) != payload_bits))
    log(f"  PRE : hash_ok={ok_pre} byte_exact={pl_pre==payload} BER={ber_pre:.4f}")

    # POST-NeuPerm (seeded repeats)
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
        f"BER={np.mean(post_ber):.4f}±{np.std(post_ber):.4f}")

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["vgg11"])
    ap.add_argument("--fraction", type=float, default=0.25)
    ap.add_argument("--x-bytes", type=int, nargs="+", default=[2], choices=[2, 3])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--csv", default=os.path.join(RESULTS_DIR, "evilmodel_neuperm.csv"))
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for model_name in args.models:
        for x_bytes in args.x_bytes:
            log(f"\n=== {model_name} (x_bytes={x_bytes}) ===")
            try:
                rows.append(run_model(model_name, args.fraction, x_bytes,
                                      args.repeats, args.seed))
            except Exception as e:
                log(f"  ERROR: {e}")
            pd.DataFrame(rows).to_csv(args.csv, index=False)

    log("\n" + "=" * 80)
    log(f"{'model':<14s} {'scheme':<18s} {'pre_ok':>7s} {'pre_BER':>8s} "
        f"{'post_ok':>8s} {'post_BER':>9s}")
    for r in rows:
        log(f"{r['model_name']:<14s} {r['scheme']:<18s} {str(r['pre_hash_ok']):>7s} "
            f"{r['pre_ber']:>8.4f} {r['post_hash_ok_rate']:>8.2f} {r['post_ber_mean']:>9.4f}")
    log(f"\nCSV saved to {args.csv}")


if __name__ == "__main__":
    main()
