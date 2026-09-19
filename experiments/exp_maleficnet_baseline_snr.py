"""Compute baseline SNR for original (undisrupted) MaleficNet models."""
import hashlib
import logging
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch

MALEFICNET_DIR = Path("external_code/maleficnet")
sys.path.insert(0, str(MALEFICNET_DIR))

from models.densenet import Model
from extractor import Extractor
from utils.utils_bit import bits_from_bytes

warnings.filterwarnings("ignore")
logging.getLogger('PIL').setLevel(logging.CRITICAL)

log = logging.getLogger()
log.setLevel(logging.INFO)

ORIG_DIR = Path("/tmp/maleficnet_models")
from payload_source import payload_dir, require_payload
MALWARE_DIR = payload_dir()  # default: repo payloads/synthetic (inert substitutes); override via NEUPERM_PAYLOAD_DIR/MALWARE_DIR
EXTRACT_DIR = Path("/tmp/maleficnet_extract")
MSG_LENGTHS_CSV = MALEFICNET_DIR / "message_lengths.csv"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

CHUNK_FACTOR = 6
SEED = 42
NUM_CLASSES = 10
DIM = 32

FILENAME_TO_MODEL = {
    "densenet": "densenet121",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "vgg11": "vgg11",
}

PAYLOAD_NAMES = [
    "stuxnet", "destover", "asprox", "bladabindi",
    "zeus-bank", "eq.drug", "kovter", "cerber", "ardamax",
]
FILENAME_PAYLOAD_ALIASES = {"ed": "eq.drug", "eq": "eq.drug"}

SKIP = {"pre"}
# ardamax / zeus-bank are the two samples the artifact does not ship (documented
# "Missing" in REPRODUCTION_GUIDE.md §2.3; not part of the paper's Table 5). They
# are excluded here by name — an explicit, auditable exclusion, not a silent skip.
# Any OTHER payload missing from the payload dir is a hard error (require_payload).
KNOWN_ABSENT = {"ardamax", "zeus-bank"}


def load_payload_xor(payload_name):
    xor_path = MALWARE_DIR / f"{payload_name}.xor"
    with open(xor_path, 'rb') as f:
        xored = f.read()
    raw_bytes = bytes(b ^ 0xFF for b in xored)
    payload_bits = bits_from_bytes(raw_bytes)
    hash_str = hashlib.sha256(
        ''.join(str(l) for l in payload_bits).encode('utf-8')).hexdigest()
    hash_bits = bits_from_bytes([char for char in hash_str.encode('utf-8')])
    return payload_bits, hash_bits


def parse_filename(filename):
    stem = Path(filename).stem
    model_name = None
    for prefix, name in FILENAME_TO_MODEL.items():
        if stem.startswith(prefix + "_"):
            model_name = name
            break
    if model_name is None:
        return None, None

    payload_name = None
    for p in PAYLOAD_NAMES:
        if f"_{p}_" in stem:
            payload_name = p
            break
    if payload_name is None:
        for alias, real_name in FILENAME_PAYLOAD_ALIASES.items():
            if f"_{alias}_" in stem:
                payload_name = real_name
                break
    return model_name, payload_name


def get_message_length(model_name, payload_name):
    df = pd.read_csv(MSG_LENGTHS_CSV)
    row = df[(df['model_name'] == model_name) & (df['payload_name'] == payload_name)]
    if len(row) == 0:
        raise ValueError(f"No message length for {model_name}/{payload_name}")
    return int(row['message_length'].values[0])


def main():
    # Get original model files (skip pre, ardamax, zeus-bank)
    files = sorted(ORIG_DIR.glob("*.pt"))
    files = [f for f in files if not any(f"_{s}_" in f.name for s in SKIP)]

    results = []
    for pt_path in files:
        model_name, payload_name = parse_filename(pt_path.name)
        if model_name is None or payload_name is None:
            print(f"SKIP {pt_path.name}", flush=True)
            continue

        # ardamax / zeus-bank are not shipped (see KNOWN_ABSENT): skip by name.
        if payload_name in KNOWN_ABSENT:
            print(f"SKIP {pt_path.name} (payload {payload_name} not shipped — "
                  f"see REPRODUCTION_GUIDE.md §2.3)", flush=True)
            continue
        # Any other missing payload is a hard error, never a silent skip.
        xor_path = require_payload(payload_name, MALWARE_DIR)

        print(f"\n{pt_path.name} -> {model_name}/{payload_name}", flush=True)

        try:
            model = Model(input_shape=DIM, num_classes=NUM_CLASSES,
                          only_pretrained=False, model=model_name)
            model.load_state_dict(torch.load(pt_path, map_location="cpu"))
            model.eval()

            payload_bits, hash_bits = load_payload_xor(payload_name)
            extractor = Extractor(
                seed=SEED, device="cpu", result_path=EXTRACT_DIR,
                logger=log, malware_length=len(payload_bits),
                hash_length=len(hash_bits), chunk_factor=CHUNK_FACTOR,
            )
            message_length = get_message_length(model_name, payload_name)
            snr = extractor.extract(model, message_length, payload_name, ret_snr=True)
            print(f"  SNR = {snr:.4f}", flush=True)

            results.append({
                'model_name': model_name,
                'payload_name': payload_name,
                'quant_method': 'baseline',
                'snr': snr,
                'filename': pt_path.name,
            })

            df = pd.DataFrame(results)
            csv_path = RESULTS_DIR / "maleficnet_baseline_snr.csv"
            df.to_csv(csv_path, index=False)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {e}", flush=True)

    # Merge with PTQ results
    csv_ptq = RESULTS_DIR / "maleficnet_ptq_snr.csv"
    csv_baseline = RESULTS_DIR / "maleficnet_baseline_snr.csv"
    csv_combined = RESULTS_DIR / "maleficnet_ptq_snr.csv"

    df_ptq = pd.read_csv(csv_ptq)
    df_base = pd.read_csv(csv_baseline)
    df_all = pd.concat([df_base, df_ptq], ignore_index=True)
    df_all.to_csv(csv_combined, index=False)
    print(f"\nMerged {len(df_all)} total results into {csv_combined}", flush=True)


if __name__ == "__main__":
    main()
