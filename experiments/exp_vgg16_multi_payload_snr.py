"""VGG16 SNR: inject multiple payloads with MaleficNet, measure SNR for
Baseline, NeuPerm, PTQ 2/4/8-bit, noise (4 eps), and pruning (2 amounts).

Results are written incrementally to CSV so partial results are always available.
"""
import copy, hashlib, logging, os, sys, time, warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision

# ── MaleficNet code ──────────────────────────────────────────────────────────
MALEFICNET_DIR = Path("external_code/maleficnet")
sys.path.insert(0, str(MALEFICNET_DIR))

# Monkey-patch XOR malware loading
import utils.utils_bit as utils_bit
_orig = utils_bit.bits_from_file
def _xor_bits_from_file(path):
    path = str(path)
    xor_path = path + ".xor"
    if not os.path.exists(path) and os.path.exists(xor_path):
        with open(xor_path, "rb") as f:
            raw = bytes(b ^ 0xFF for b in f.read())
        return utils_bit.bits_from_bytes(raw)
    return _orig(path)
utils_bit.bits_from_file = _xor_bits_from_file
import injector as _inj_mod
_inj_mod.bits_from_file = _xor_bits_from_file

from injector import Injector
from extractor import Extractor

# ── NeuPerm imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from neu_perm.config import RESULTS_DIR
from neu_perm.perm import permute_model
from neu_perm.quantization import quantize_dequantize_sd, get_bn_keys_from_sd

warnings.filterwarnings("ignore")
logging.getLogger('PIL').setLevel(logging.CRITICAL)
log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

# ── Config ───────────────────────────────────────────────────────────────────
from payload_source import payload_dir, require_payload
MALWARE_DIR = payload_dir()  # default: repo payloads/synthetic (inert substitutes); override via NEUPERM_PAYLOAD_DIR/MALWARE_DIR
OUT_DIR = Path("/tmp/maleficnet_models/vgg16_snr")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "vgg16"
PAYLOADS = ["stuxnet", "destover", "asprox", "bladabindi",
            "kovter", "cerber", "eq.drug"]
SEED = 42
CHUNK_FACTOR = 6
GAMMA = 0.0009  # MaleficNet default; raise to 0.005 if baseline SNR < 1 dB

NOISE_EPSILONS = [1e-4, 1e-3, 1e-2, 1e-1]
PRUNE_AMOUNTS = [0.01, 0.05]
PTQ_BITS = [8, 4, 2]

def get_csv_path(payload=None):
    if payload:
        return Path(RESULTS_DIR) / f"vgg16_snr_{payload}.csv"
    return Path(RESULTS_DIR) / "vgg16_snr.csv"


def load_pretrained_vgg16():
    weights = torchvision.models.get_model_weights(MODEL_NAME).DEFAULT
    model = torchvision.models.get_model(MODEL_NAME, weights=weights)
    return model


def get_extract_dir(payload):
    d = Path(f"/tmp/maleficnet_malware/extract_{payload}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_injector(payload):
    return Injector(
        seed=SEED, device="cpu",
        malware_path=MALWARE_DIR / payload,
        result_path=get_extract_dir(payload), logger=log, chunk_factor=CHUNK_FACTOR,
    )


def make_extractor(injector, payload):
    return Extractor(
        seed=SEED, device="cpu", result_path=get_extract_dir(payload), logger=log,
        malware_length=len(injector.payload),
        hash_length=len(injector.hash),
        chunk_factor=CHUNK_FACTOR,
    )


def extract_snr(sd, extractor, message_length, payload):
    """Load sd into a fresh VGG16, run extractor, return SNR."""
    m = torchvision.models.get_model(MODEL_NAME, weights=None)
    m.load_state_dict(sd)
    m.eval()
    snr = extractor.extract(m, message_length, payload, ret_snr=True)
    return snr


def apply_noise(sd, eps):
    """Add Gaussian noise to all float tensors in sd (in-place copy)."""
    sd = copy.deepcopy(sd)
    for k, v in sd.items():
        if v.is_floating_point():
            noise = torch.from_numpy(
                np.random.normal(0, eps, v.shape).astype(
                    {torch.float16: np.float16, torch.float32: np.float32,
                     torch.float64: np.float64}.get(v.dtype, np.float32)
                )
            )
            sd[k] = v + noise
    return sd


def apply_pruning(sd, amount):
    """Random unstructured pruning on Conv2d and Linear weights."""
    m = torchvision.models.get_model(MODEL_NAME, weights=None)
    m.load_state_dict(copy.deepcopy(sd))
    for _, module in m.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.random_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # make pruning permanent
    return m.state_dict()


def append_result(result, csv_path):
    """Append a single result dict to the CSV, creating it if needed."""
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    else:
        df = pd.DataFrame([result])
    df.to_csv(csv_path, index=False)


def is_done(payload, method, csv_path):
    """Check if this (payload, method) combo already exists in CSV."""
    if not csv_path.exists():
        return False
    df = pd.read_csv(csv_path)
    return len(df[(df["payload_name"] == payload) & (df["method"] == method)]) > 0


def run_for_payload(payload, csv_path):
    print(f"\n{'='*60}\n=== {MODEL_NAME} / {payload} ===\n{'='*60}", flush=True)

    print("[1] Loading pretrained VGG16...", flush=True)
    model = load_pretrained_vgg16().eval()

    print(f"[2] Injecting {payload} (gamma={GAMMA})...", flush=True)
    injector = make_injector(payload)
    print(f"  payload={len(injector.payload)} bits, hash={len(injector.hash)} bits", flush=True)
    t0 = time.time()
    new_sd, message_length, _, _ = injector.inject(model, gamma=GAMMA)
    print(f"  injected in {time.time()-t0:.0f}s, message_length={message_length} bits", flush=True)

    # Save injected checkpoint
    injected_path = OUT_DIR / f"{MODEL_NAME}_{payload}_injected.pt"
    torch.save(new_sd, injected_path)

    sd_injected = OrderedDict(
        {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in new_sd.items()}
    )
    extractor = make_extractor(injector, payload)

    def measure(sd, label, method):
        if is_done(payload, method, csv_path):
            print(f"  SKIP {label} (already done)", flush=True)
            return
        t1 = time.time()
        snr = extract_snr(sd, extractor, message_length, payload)
        elapsed = time.time() - t1
        print(f"  {label}: SNR = {snr:.4f}  ({elapsed:.0f}s)", flush=True)
        append_result({"model_name": MODEL_NAME, "payload_name": payload,
                       "method": method, "snr": snr}, csv_path)

    print("[3] Measuring SNR across methods...", flush=True)

    # Baseline
    measure(sd_injected, "Baseline", "baseline")

    # NeuPerm
    sd_np = permute_model(MODEL_NAME, copy.deepcopy(sd_injected), inplace=True)
    measure(sd_np, "NeuPerm", "neuperm")
    del sd_np

    # PTQ
    bn_keys = get_bn_keys_from_sd(sd_injected)
    for bits in PTQ_BITS:
        method = f"ptq{bits}_perchannel"
        sd_q = quantize_dequantize_sd(
            sd_injected, n_bits=bits, per_channel=True,
            inplace=False, skip_keys=bn_keys,
        )
        measure(sd_q, f"PTQ-{bits} per-channel", method)
        del sd_q

    # Noise
    for eps in NOISE_EPSILONS:
        method = f"noise_{eps}"
        sd_n = apply_noise(sd_injected, eps)
        measure(sd_n, f"Noise eps={eps}", method)
        del sd_n

    # Pruning
    for amount in PRUNE_AMOUNTS:
        method = f"prune_{amount}"
        sd_p = apply_pruning(sd_injected, amount)
        measure(sd_p, f"Prune amount={amount}", method)
        del sd_p

    print(f"[4] Done with {payload}.", flush=True)
    del model, sd_injected, new_sd, extractor, injector


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--payload', type=str, default=None,
                        help='Run a single payload (for parallel execution)')
    args = parser.parse_args()

    if args.payload:
        payloads = [args.payload]
    else:
        payloads = PAYLOADS

    for payload in payloads:
        csv_path = get_csv_path(payload)
        print(f"Results CSV: {csv_path}", flush=True)

        all_methods = (
            ["baseline", "neuperm"]
            + [f"ptq{b}_perchannel" for b in PTQ_BITS]
            + [f"noise_{e}" for e in NOISE_EPSILONS]
            + [f"prune_{a}" for a in PRUNE_AMOUNTS]
        )
        if all(is_done(payload, m, csv_path) for m in all_methods):
            print(f"SKIP {payload} (all methods done)", flush=True)
            continue
        run_for_payload(payload, csv_path)

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
