"""Inject stuxnet into pretrained MobileNetV2 and measure SNR under disruption methods.

Steps:
1. Load pretrained MobileNetV2 (ImageNet weights)
2. Inject stuxnet payload using MaleficNet's Injector
3. Apply disruption methods: baseline (none), NeuPerm, noise (eps=1e-4), PTQ8 per-channel
4. Measure SNR for each using MaleficNet's Extractor
"""
import copy, hashlib, logging, os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision

# MaleficNet code path
MALEFICNET_DIR = Path(os.environ.get(
    "MALEFICNET_DIR", "external_code/maleficnet"))
sys.path.insert(0, str(MALEFICNET_DIR))

# Monkey-patch bits_from_file to read XOR-encoded malware files
import utils.utils_bit as utils_bit
_orig_bits_from_file = utils_bit.bits_from_file
def _xor_bits_from_file(path):
    path = str(path)
    # Prefer XOR-encoded variant if the raw file is missing
    xor_path = path + ".xor"
    if not os.path.exists(path) and os.path.exists(xor_path):
        with open(xor_path, "rb") as f:
            raw = bytes(b ^ 0xFF for b in f.read())
        return utils_bit.bits_from_bytes(raw)
    return _orig_bits_from_file(path)
utils_bit.bits_from_file = _xor_bits_from_file
# Also patch the already-imported name in injector.py
import injector as _inj_mod
_inj_mod.bits_from_file = _xor_bits_from_file

from injector import Injector
from extractor import Extractor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from neu_perm.config import RESULTS_DIR
RESULTS_DIR = os.environ.get("SNR_RESULTS_DIR", RESULTS_DIR)  # non-destructive: new-impl runs to a separate dir
from neu_perm.perm import permute_model
from neu_perm.quantization import quantize_dequantize_sd, get_bn_keys_from_sd

warnings.filterwarnings("ignore")
logging.getLogger('PIL').setLevel(logging.CRITICAL)
log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

from payload_source import payload_dir, require_payload
MALWARE_DIR = payload_dir()  # default: repo payloads/synthetic (inert substitutes); override via NEUPERM_PAYLOAD_DIR/MALWARE_DIR
EXTRACT_DIR = Path(os.environ.get("EXTRACT_DIR", "/tmp/maleficnet_extract"))
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path(os.environ.get("OUT_DIR", "checkpoints/maleficnet/mobilenetv2_stuxnet"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "mobilenet_v2"
PAYLOAD = os.environ.get("SNR_PAYLOAD", "stuxnet")
SEED = 42
CHUNK_FACTOR = 6
GAMMA = 0.005  # Increased from MaleficNet's default 0.0009 to boost signal on MobileNetV2
EPS_NOISE = 1e-4
PTQ_BITS = 8
SNR_METHODS = os.environ.get("SNR_METHODS", "baseline,neuperm,noise,ptq8").split(",")


def load_pretrained_mobilenet_v2():
    import urllib.parse
    weights = torchvision.models.get_model_weights(MODEL_NAME).DEFAULT
    try:
        model = torchvision.models.get_model(MODEL_NAME, weights=weights)
    except RuntimeError:
        url = weights.url
        fn = os.path.basename(urllib.parse.urlparse(url).path)
        cached = os.path.join(torch.hub.get_dir(), 'checkpoints', fn)
        if not os.path.exists(cached):
            torch.hub.download_url_to_file(url, cached)
        sd = torch.load(cached, map_location='cpu')
        model = torchvision.models.get_model(MODEL_NAME, weights=None)
        model.load_state_dict(sd)
    return model


def make_injector():
    return Injector(
        seed=SEED, device="cpu",
        malware_path=MALWARE_DIR / PAYLOAD,
        result_path=EXTRACT_DIR, logger=log, chunk_factor=CHUNK_FACTOR,
    )


def make_extractor(injector):
    return Extractor(
        seed=SEED, device="cpu", result_path=EXTRACT_DIR, logger=log,
        malware_length=len(injector.payload),
        hash_length=len(injector.hash),
        chunk_factor=CHUNK_FACTOR,
    )


def apply_noise(sd, eps):
    out = copy.deepcopy(sd)
    ws = []
    keys = []
    for k, v in out.items():
        if v.dtype.is_floating_point:
            ws.append(v.flatten().numpy())
            keys.append(k)
    flat = np.concatenate(ws)
    flat = flat + np.random.normal(0, eps, flat.shape).astype(flat.dtype)
    offset = 0
    for k in keys:
        shape = out[k].shape
        n = out[k].numel()
        out[k] = torch.from_numpy(flat[offset:offset+n].reshape(shape))
        offset += n
    return out


def run_extract(sd):
    """Load state dict into a fresh MobileNetV2 and run Extractor."""
    model = torchvision.models.get_model(MODEL_NAME, weights=None).eval()
    model.load_state_dict(sd)
    injector = make_injector()
    extractor = make_extractor(injector)
    message_length = injector.get_message_length(model) if not hasattr(injector, '_cached_ml') else injector._cached_ml
    snr = extractor.extract(model, message_length, PAYLOAD, ret_snr=True)
    return snr, message_length


def main():
    print(f"=== MobileNetV2 / stuxnet injection + disruption SNR ===", flush=True)

    # Step 1: Load pretrained
    print("\n[1/5] Loading pretrained MobileNetV2...", flush=True)
    model = load_pretrained_mobilenet_v2()
    model.eval()

    # Step 2: Inject
    print("\n[2/5] Injecting stuxnet payload...", flush=True)
    injector = make_injector()
    print(f"  Payload length: {len(injector.payload)} bits", flush=True)
    print(f"  Hash length: {len(injector.hash)} bits", flush=True)
    new_sd, message_length, _, _ = injector.inject(model, gamma=GAMMA)
    print(f"  Message length (encoded): {message_length} bits", flush=True)

    # Save injected state dict
    injected_path = OUT_DIR / f"{MODEL_NAME}_stuxnet_injected.pt"
    torch.save(new_sd, injected_path)
    print(f"  Saved: {injected_path}", flush=True)

    # Make sure new_sd is a proper dict (detach + cpu for safety)
    sd_injected = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in new_sd.items()}

    # Step 3: Build extractor once (shared state)
    print("\n[3/5] Setting up extractor...", flush=True)
    extractor = make_extractor(injector)

    def extract_from_sd(sd, label):
        m = torchvision.models.get_model(MODEL_NAME, weights=None).eval()
        m.load_state_dict(sd)
        snr = extractor.extract(m, message_length, PAYLOAD, ret_snr=True)
        print(f"  {label}: SNR = {snr:.4f}", flush=True)
        return snr

    # Step 4: Apply disruptions and measure SNR
    print("\n[4/5] Measuring SNR under disruption methods...", flush=True)
    results = []

    # Baseline (injected, no disruption)
    if "baseline" in SNR_METHODS:
        snr = extract_from_sd(sd_injected, "Baseline (injected)")
        results.append({"method": "baseline", "snr": snr})

    # NeuPerm
    if "neuperm" in SNR_METHODS:
        sd_neuperm = permute_model(MODEL_NAME, copy.deepcopy(sd_injected), inplace=True)
        snr = extract_from_sd(sd_neuperm, "NeuPerm")
        results.append({"method": "neuperm", "snr": snr})
        del sd_neuperm

    # Noise eps=1e-4
    if "noise" in SNR_METHODS:
        sd_noise = apply_noise(sd_injected, EPS_NOISE)
        snr = extract_from_sd(sd_noise, f"Noise eps={EPS_NOISE}")
        results.append({"method": f"noise_{EPS_NOISE}", "snr": snr})
        del sd_noise

    # PTQ 8-bit per-channel (skip BN)
    if "ptq8" in SNR_METHODS:
        bn_keys = get_bn_keys_from_sd(sd_injected)
        sd_ptq = quantize_dequantize_sd(
            sd_injected, n_bits=PTQ_BITS, per_channel=True,
            inplace=False, skip_keys=bn_keys,
        )
        snr = extract_from_sd(sd_ptq, f"PTQ-{PTQ_BITS}bit perchannel")
        results.append({"method": f"ptq{PTQ_BITS}_perchannel", "snr": snr})
        del sd_ptq

    # Step 5: Save CSV
    print("\n[5/5] Saving results...", flush=True)
    df = pd.DataFrame(results)
    df["model_name"] = MODEL_NAME
    df["payload_name"] = PAYLOAD
    csv_path = Path(RESULTS_DIR) / f"mobilenetv2_{PAYLOAD}_snr.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}", flush=True)
    print(df.to_string(index=False), flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
