"""Extract MaleficNet payloads from PTQ-quantized state dicts and measure SNR.

Loads each quantized .pt file, initializes the MaleficNet Model wrapper,
runs the CDMA-based extractor to compute the Signal-to-Noise Ratio (SNR),
and saves results to a CSV.

Usage:
    python -u experiments/exp_maleficnet_extract_snr.py \
        --model_file densenet_cifar10_stuxnet_model_ptq8_perchannel.pt

    # Or run all files:
    python -u experiments/exp_maleficnet_extract_snr.py --all
"""
import argparse
import hashlib
import logging
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch

# Add MaleficNet code to path
MALEFICNET_DIR = Path("external_code/maleficnet")
sys.path.insert(0, str(MALEFICNET_DIR))

from models.densenet import Model
from models.llms import LLMModel
from extractor import Extractor
from utils.utils_bit import bits_from_bytes

# Suppress noisy warnings
warnings.filterwarnings("ignore")
logging.getLogger('PIL').setLevel(logging.CRITICAL)

log = logging.getLogger()
log.setLevel(logging.INFO)

QUANT_SD_DIR = Path("checkpoints/maleficnet/quant_sd")
from payload_source import payload_dir, require_payload
MALWARE_DIR = payload_dir()  # default: repo payloads/synthetic (inert substitutes); override via NEUPERM_PAYLOAD_DIR/MALWARE_DIR
EXTRACT_DIR = Path("/tmp/maleficnet_extract")
MSG_LENGTHS_CSV = MALEFICNET_DIR / "message_lengths.csv"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

CHUNK_FACTOR = 6
SEED = 42
NUM_CLASSES = 10
DIM = 32

# Map filename prefix to model architecture name
FILENAME_TO_MODEL = {
    "densenet": "densenet121",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "vgg11": "vgg11",
    "vgg16": "vgg16",
    "llama-3.2-1b": "llama-3.2-1b",
}

LLM_MODELS = {"llama-3.2-1b"}

# Known payload names that appear in filenames.
# Map from filename token to actual payload name (for cases where they differ).
PAYLOAD_NAMES = [
    "stuxnet", "destover", "asprox", "bladabindi",
    "zeus-bank", "eq.drug", "kovter", "cerber", "ardamax",
]
FILENAME_PAYLOAD_ALIASES = {
    "ed": "eq.drug",
    "eq": "eq.drug",
}


def load_payload_xor(payload_name: str):
    """Load XOR-encoded payload and decode it. Returns (payload_bits, hash_bits).

    Malware files are stored XOR-encoded (each byte ^ 0xFF) to avoid
    antivirus deletion. This function decodes and computes the same
    payload/hash that MaleficNet's Injector would produce.
    """
    xor_path = MALWARE_DIR / f"{payload_name}.xor"
    with open(xor_path, 'rb') as f:
        xored = f.read()
    # Decode: XOR with 0xFF
    raw_bytes = bytes(b ^ 0xFF for b in xored)
    # Convert to bits (same as bits_from_bytes in MaleficNet)
    payload_bits = bits_from_bytes(raw_bytes)
    # Compute hash (same as Injector.__init__)
    hash_str = hashlib.sha256(
        ''.join(str(l) for l in payload_bits).encode('utf-8')).hexdigest()
    hash_bits = bits_from_bytes([char for char in hash_str.encode('utf-8')])
    return payload_bits, hash_bits


def parse_filename(filename: str):
    """Extract model_name, payload_name, and quant_method from filename.

    E.g. 'densenet_cifar10_stuxnet_model_ptq8_perchannel.pt'
      -> ('densenet121', 'stuxnet', 'ptq8_perchannel')
    """
    stem = Path(filename).stem  # remove .pt

    # Find model name
    model_name = None
    for prefix, name in FILENAME_TO_MODEL.items():
        if stem.startswith(prefix + "_"):
            model_name = name
            break
    if model_name is None:
        return None, None, None

    # Find payload name
    payload_name = None
    for p in PAYLOAD_NAMES:
        if f"_{p}_" in stem:
            payload_name = p
            break
    if payload_name is None:
        # Check aliases (e.g. 'ed' -> 'eq.drug')
        for alias, real_name in FILENAME_PAYLOAD_ALIASES.items():
            if f"_{alias}_" in stem:
                payload_name = real_name
                break
    if payload_name is None:
        return model_name, None, None

    # Extract quant method (everything after '_model_')
    quant_method = stem.split("_model_", 1)[-1] if "_model_" in stem else None

    return model_name, payload_name, quant_method


def get_message_length(model_name: str, payload_name: str) -> int:
    """Look up pre-computed message length from CSV."""
    df = pd.read_csv(MSG_LENGTHS_CSV)
    row = df[(df['model_name'] == model_name) & (df['payload_name'] == payload_name)]
    if len(row) == 0:
        raise ValueError(f"No message length for {model_name}/{payload_name}")
    return int(row['message_length'].values[0])


def extract_snr(model_path: Path, model_name: str, payload_name: str) -> float:
    """Load a quantized model and extract the SNR."""
    # Initialize model
    if model_name in LLM_MODELS:
        model = LLMModel(only_pretrained=True, model=model_name)
    else:
        model = Model(input_shape=DIM, num_classes=NUM_CLASSES,
                      only_pretrained=False, model=model_name)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Load payload (XOR-decoded) to get lengths
    payload_bits, hash_bits = load_payload_xor(payload_name)

    # Initialize extractor
    extractor = Extractor(
        seed=SEED, device="cpu",
        result_path=EXTRACT_DIR,
        logger=log,
        malware_length=len(payload_bits),
        hash_length=len(hash_bits),
        chunk_factor=CHUNK_FACTOR,
    )

    # Get message length
    message_length = get_message_length(model_name, payload_name)

    # Extract SNR
    snr = extractor.extract(model, message_length, payload_name, ret_snr=True)
    return snr


def process_file(filename: str):
    """Process a single .pt file and return result dict."""
    model_name, payload_name, quant_method = parse_filename(filename)
    if model_name is None or payload_name is None or quant_method is None:
        print(f"SKIP {filename} (could not parse)", flush=True)
        return None

    model_path = QUANT_SD_DIR / filename
    print(f"\n{'='*60}", flush=True)
    print(f"{filename}", flush=True)
    print(f"  model={model_name}, payload={payload_name}, method={quant_method}", flush=True)
    print(f"{'='*60}", flush=True)

    try:
        snr = extract_snr(model_path, model_name, payload_name)
        print(f"  SNR = {snr:.4f}", flush=True)
        return {
            'model_name': model_name,
            'payload_name': payload_name,
            'quant_method': quant_method,
            'snr': snr,
            'filename': filename,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ERROR: {e}", flush=True)
        return {
            'model_name': model_name,
            'payload_name': payload_name,
            'quant_method': quant_method,
            'snr': None,
            'filename': filename,
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default=None,
                        help='Single .pt file to process')
    parser.add_argument('--all', action='store_true',
                        help='Process all .pt files in QUANT_SD_DIR')
    args = parser.parse_args()

    if args.model_file:
        files = [args.model_file]
    elif args.all:
        files = sorted(f.name for f in QUANT_SD_DIR.glob("*.pt"))
    else:
        parser.error("Specify --model_file or --all")

    csv_path = RESULTS_DIR / "maleficnet_ptq_snr.csv"

    # Load existing results to append to (and skip already-processed files)
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        existing_files = set(existing_df['filename'].tolist())
        results = existing_df.to_dict('records')
        files = [f for f in files if f not in existing_files]
        print(f"Loaded {len(results)} existing results, {len(files)} new file(s) to process", flush=True)
    else:
        results = []
        print(f"Processing {len(files)} file(s)", flush=True)

    for filename in files:
        result = process_file(filename)
        if result is not None:
            results.append(result)

            # Save incrementally
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)
            print(f"  Saved to {csv_path}", flush=True)

    print(f"\nDone. {len(results)} total results saved.", flush=True)


if __name__ == "__main__":
    main()
