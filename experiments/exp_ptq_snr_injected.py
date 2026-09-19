"""Apply PTQ 4/2-bit per-channel quantization to already-injected MaleficNet
state dicts and measure extraction SNR.

Operates on the 7 injected checkpoints saved during the previous experiments:
  - mobilenet_v2        / stuxnet
  - mobilenet_v3_small  / stuxnet_t16
  - efficientnet_b0     / stuxnet
  - efficientnet_b4     / stuxnet, destover, asprox, bladabindi

Writes to results/ptq_snr_injected.csv (append-only, resumable, never touches
the existing per-experiment CSVs).
"""
import copy, logging, os, sys, warnings
from pathlib import Path

import pandas as pd
import torch
import torchvision

MALEFICNET_DIR = Path("external_code/maleficnet")
sys.path.insert(0, str(MALEFICNET_DIR))

from extractor import Extractor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from neu_perm.config import RESULTS_DIR
from neu_perm.quantization import quantize_dequantize_sd, get_bn_keys_from_sd

warnings.filterwarnings("ignore")
logging.getLogger('PIL').setLevel(logging.CRITICAL)
log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

EXTRACT_DIR = Path("/tmp/maleficnet_extract")
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CHUNK_FACTOR = 6
PTQ_BITS = [4, 2]

# (model_name, payload_name, injected_pt_path, message_length, malware_length_bits, hash_length_bits)
# message_length, malware_length, hash_length taken from the injection logs.
#   malware_length = len(injector.payload) in bits (bytes * 8)
#   hash_length = 512 (sha256 hex string = 64 chars = 512 bits)
JOBS = [
    ("mobilenet_v2", "stuxnet",
     "checkpoints/maleficnet/mobilenetv2_stuxnet/mobilenet_v2_stuxnet_injected.pt",
     268424, 199680, 512),
    ("mobilenet_v3_small", "stuxnet_t16",
     "checkpoints/maleficnet/mobilenetv3s_stuxnet/mobilenet_v3_small_stuxnet_injected.pt",
     183080, 128000, 512),
    ("efficientnet_b0", "stuxnet",
     "checkpoints/maleficnet/efficientnetb0_stuxnet/efficientnet_b0_stuxnet_injected.pt",
     268424, 199680, 512),
    ("efficientnet_b4", "stuxnet",
     "checkpoints/maleficnet/efficientnetb4_snr/efficientnet_b4_stuxnet_injected.pt",
     268424, 199680, 512),
    ("efficientnet_b4", "destover",
     "checkpoints/maleficnet/efficientnetb4_snr/efficientnet_b4_destover_injected.pt",
     987752, 735104, 512),
    ("efficientnet_b4", "asprox",
     "checkpoints/maleficnet/efficientnetb4_snr/efficientnet_b4_asprox_injected.pt",
     1012136, 753664, 512),
    ("efficientnet_b4", "bladabindi",
     "checkpoints/maleficnet/efficientnetb4_snr/efficientnet_b4_bladabindi_injected.pt",
     1158440, 860160, 512),
]


def make_extractor(malware_length, hash_length):
    return Extractor(
        seed=SEED, device="cpu", result_path=EXTRACT_DIR, logger=log,
        malware_length=malware_length, hash_length=hash_length,
        chunk_factor=CHUNK_FACTOR,
    )


def run_ptq_snr(model_name, payload, pt_path, message_length, malware_length, hash_length, n_bits):
    """Load injected sd, apply PTQ, extract SNR."""
    sd = torch.load(pt_path, map_location="cpu")
    sd = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in sd.items()}

    bn_keys = get_bn_keys_from_sd(sd)
    sd_q = quantize_dequantize_sd(
        sd, n_bits=n_bits, per_channel=True, inplace=False, skip_keys=bn_keys,
    )

    model = torchvision.models.get_model(model_name, weights=None).eval()
    model.load_state_dict(sd_q)
    del sd_q, sd

    extractor = make_extractor(malware_length, hash_length)
    snr = extractor.extract(model, message_length, payload, ret_snr=True)
    del model
    return snr


def main():
    csv_path = Path(RESULTS_DIR) / "ptq_snr_injected.csv"

    # Resume
    if csv_path.exists():
        existing = pd.read_csv(csv_path).to_dict("records")
        done = {(r["model_name"], r["payload_name"], r["quant_method"]) for r in existing}
        rows = existing
        print(f"Loaded {len(rows)} existing rows from {csv_path}", flush=True)
    else:
        done = set()
        rows = []

    for model_name, payload, pt_path, msg_len, mal_len, hash_len in JOBS:
        if not os.path.exists(pt_path):
            print(f"SKIP {model_name}/{payload}: checkpoint not found ({pt_path})", flush=True)
            continue
        for nb in PTQ_BITS:
            method = f"ptq{nb}_perchannel"
            key = (model_name, payload, method)
            if key in done:
                print(f"SKIP {model_name}/{payload}/{method} (already done)", flush=True)
                continue
            print(f"\n=== {model_name} / {payload} / {method} ===", flush=True)
            print(f"  loading {os.path.basename(pt_path)}", flush=True)
            snr = run_ptq_snr(model_name, payload, pt_path, msg_len, mal_len, hash_len, nb)
            print(f"  SNR = {snr:.4f}", flush=True)
            rows.append({
                "model_name": model_name,
                "payload_name": payload,
                "quant_method": method,
                "n_bits": nb,
                "snr": snr,
            })
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("\nDone.", flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
