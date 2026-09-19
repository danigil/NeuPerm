"""EfficientNet-B4 SNR: inject multiple payloads (stuxnet, destover, asprox,
bladabindi) with gamma=0.005, measure Baseline and NeuPerm SNR only.
"""
import copy, hashlib, logging, os, sys, time, warnings
from pathlib import Path

import pandas as pd
import torch
import torchvision

MALEFICNET_DIR = Path(os.environ.get(
    "MALEFICNET_DIR", "external_code/maleficnet"))
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from neu_perm.config import RESULTS_DIR
RESULTS_DIR = os.environ.get("SNR_RESULTS_DIR", RESULTS_DIR)  # non-destructive: new-impl runs to a separate dir
from neu_perm.perm import permute_model

warnings.filterwarnings("ignore")
logging.getLogger('PIL').setLevel(logging.CRITICAL)
log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

from payload_source import payload_dir, require_payload
MALWARE_DIR = payload_dir()  # default: repo payloads/synthetic (inert substitutes); override via NEUPERM_PAYLOAD_DIR/MALWARE_DIR
EXTRACT_DIR = Path(os.environ.get("EXTRACT_DIR", "/tmp/maleficnet_extract"))
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path(os.environ.get("OUT_DIR", "checkpoints/maleficnet/efficientnetb4_snr"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "efficientnet_b4"
PAYLOADS = os.environ.get("SNR_PAYLOADS", "stuxnet,destover,asprox,bladabindi").split(",")
SEED = 42
CHUNK_FACTOR = 6
GAMMA = 0.005
SNR_METHODS = os.environ.get("SNR_METHODS", "baseline,neuperm").split(",")


def load_pretrained_model():
    import urllib.parse
    weights = torchvision.models.get_model_weights(MODEL_NAME).DEFAULT
    try:
        model = torchvision.models.get_model(MODEL_NAME, weights=weights)
    except RuntimeError:
        url = weights.url
        fn = os.path.basename(urllib.parse.urlparse(url).path)
        cached = os.path.join(torch.hub.get_dir(), "checkpoints", fn)
        if not os.path.exists(cached):
            torch.hub.download_url_to_file(url, cached)
        sd = torch.load(cached, map_location="cpu")
        model = torchvision.models.get_model(MODEL_NAME, weights=None)
        model.load_state_dict(sd)
    return model


def make_injector(payload):
    return Injector(
        seed=SEED, device="cpu",
        malware_path=MALWARE_DIR / payload,
        result_path=EXTRACT_DIR, logger=log, chunk_factor=CHUNK_FACTOR,
    )


def make_extractor(injector):
    return Extractor(
        seed=SEED, device="cpu", result_path=EXTRACT_DIR, logger=log,
        malware_length=len(injector.payload),
        hash_length=len(injector.hash),
        chunk_factor=CHUNK_FACTOR,
    )


def run_for_payload(payload, csv_path):
    print(f"\n{'='*60}\n=== {MODEL_NAME} / {payload} ===\n{'='*60}", flush=True)

    print("[1/4] Loading pretrained EfficientNet-B4...", flush=True)
    model = load_pretrained_model().eval()

    print(f"[2/4] Injecting {payload} (gamma={GAMMA})...", flush=True)
    injector = make_injector(payload)
    print(f"  payload={len(injector.payload)} bits, hash={len(injector.hash)} bits", flush=True)
    t0 = time.time()
    new_sd, message_length, _, _ = injector.inject(model, gamma=GAMMA)
    print(f"  injected in {time.time()-t0:.0f}s, message_length={message_length} bits", flush=True)

    injected_path = OUT_DIR / f"{MODEL_NAME}_{payload}_injected.pt"
    torch.save(new_sd, injected_path)

    sd_injected = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in new_sd.items()}
    extractor = make_extractor(injector)

    def extract(sd, label):
        m = torchvision.models.get_model(MODEL_NAME, weights=None).eval()
        m.load_state_dict(sd)
        snr = extractor.extract(m, message_length, payload, ret_snr=True)
        print(f"  {label}: SNR = {snr:.4f}", flush=True)
        return snr

    print("[3/4] Measuring SNR...", flush=True)
    results = []
    if "baseline" in SNR_METHODS:
        snr = extract(sd_injected, "Baseline")
        results.append({"model_name": MODEL_NAME, "payload_name": payload,
                        "method": "baseline", "snr": snr})

    if "neuperm" in SNR_METHODS:
        sd_neuperm = permute_model(MODEL_NAME, copy.deepcopy(sd_injected), inplace=True)
        snr = extract(sd_neuperm, "NeuPerm")
        results.append({"model_name": MODEL_NAME, "payload_name": payload,
                        "method": "neuperm", "snr": snr})

    print("[4/4] Saving incremental results...", flush=True)
    # Append to CSV
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        # Drop any old rows for this payload so re-runs replace them
        existing = existing[existing["payload_name"] != payload]
        df = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
    else:
        df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"  Saved to {csv_path}", flush=True)

    del model, sd_injected, new_sd, extractor, injector


def main():
    csv_path = Path(RESULTS_DIR) / f"{MODEL_NAME}_multi_payload_snr.csv"

    # Resume support: skip payloads already completed
    done_payloads = set()
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        for p in existing["payload_name"].unique():
            methods = set(existing[existing["payload_name"] == p]["method"].tolist())
            if {"baseline", "neuperm"}.issubset(methods):
                done_payloads.add(p)
        print(f"Already done: {sorted(done_payloads)}", flush=True)

    for payload in PAYLOADS:
        if payload in done_payloads:
            print(f"SKIP {payload} (already done)", flush=True)
            continue
        run_for_payload(payload, csv_path)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
