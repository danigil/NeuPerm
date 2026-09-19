"""Per-payload Qwen MaleficNet inject + extract + neuperm SNR.

Usage:
    python exp_maleficnet_qwen_one.py <payload_name>

Single-payload variant of exp_maleficnet_qwen_snr.py so multiple payloads can
run in parallel processes. Each invocation writes its row into a per-payload
shard CSV: results/maleficnet_qwen_{baseline,neuperm}_snr_{payload}.csv.
The orchestrator concatenates the shards into the final CSV at the end.

CRITICAL: malware bits live only in memory; the XOR-encoded .xor files on disk
are never decoded to a tracked path.
"""
import gc
import hashlib
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MALEFICNET_DIR = Path("external_code/maleficnet")
sys.path.insert(0, str(MALEFICNET_DIR))
NEUPERM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NEUPERM_DIR))

from models.llms import LLMModel
from injector import Injector
from extractor import Extractor
from utils.utils_bit import bits_from_bytes

from neu_perm.perm import permute_model

warnings.filterwarnings("ignore")
logging.getLogger('PIL').setLevel(logging.CRITICAL)
log = logging.getLogger()
log.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

XOR_DIR = Path("/tmp/maleficnet_xor")  # writable staging for externally-built payloads
from payload_source import payload_dir  # noqa: E402


def resolve_xor(payload: str) -> Path:
    """Prefer a payload staged in XOR_DIR; otherwise fall back to the shipped
    substitutes in <repo>/payloads/synthetic (or a real-originals dir set via
    NEUPERM_PAYLOAD_DIR / MALWARE_DIR)."""
    staged = XOR_DIR / f"{payload}.xor"
    if staged.exists():
        return staged
    return payload_dir() / f"{payload}.xor"


EXTRACT_DIR = Path("/tmp/maleficnet_extract")
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path("checkpoints/maleficnet")
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "qwen2.5-1.5b"
CHUNK_FACTOR = 6
SEED = 42
GAMMA = 0.0009
DEVICE = "cpu"


def load_payload_bits(xor_path: Path):
    with open(xor_path, 'rb') as f:
        xored = f.read()
    raw_bytes = bytes(b ^ 0xFF for b in xored)
    payload_bits = bits_from_bytes(raw_bytes)
    hash_str = hashlib.sha256(
        ''.join(str(l) for l in payload_bits).encode('utf-8')).hexdigest()
    hash_bits = bits_from_bytes([char for char in hash_str.encode('utf-8')])
    return payload_bits, hash_bits


def build_fresh_qwen() -> LLMModel:
    return LLMModel(only_pretrained=True, model=MODEL_NAME)


def _setup_injector(xor_path, payload_bits, hash_bits):
    injector = Injector(
        seed=SEED, device=DEVICE, malware_path=xor_path,
        result_path=EXTRACT_DIR, logger=log, chunk_factor=CHUNK_FACTOR,
    )
    injector.payload = payload_bits
    injector.message = payload_bits + hash_bits
    # Rebuild G if the constructor used a different k for this message size.
    expected_k = 3048 if len(injector.message) > 4000 else 96
    if injector.G.shape[1] != expected_k:
        log.warning(f"LDPC k mismatch ({injector.G.shape[1]} vs {expected_k}); "
                    f"rebuilding")
        from pyldpc import make_ldpc
        d_v, d_c = 3, 12
        n = expected_k * (d_c // d_v)
        injector.H, injector.G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True, seed=SEED)
    return injector


def inject_payload(payload_name: str, xor_path: Path):
    out_path = CKPT_DIR / f"qwen2.5-1.5b_none_{payload_name}_model.pt"
    payload_bits, hash_bits = load_payload_bits(xor_path)

    if out_path.exists():
        log.info(f"[{payload_name}] stegomodel already on disk; recomputing msg_len only")
        clean = build_fresh_qwen()
        injector = _setup_injector(xor_path, payload_bits, hash_bits)
        msg_len = injector.get_message_length(clean)
        del clean
        gc.collect()
        return out_path, msg_len, len(payload_bits), len(hash_bits), None

    log.info(f"[{payload_name}] building fresh Qwen2.5-1.5B...")
    model = build_fresh_qwen()
    log.info(f"[{payload_name}] payload bits = {len(payload_bits)}  hash bits = {len(hash_bits)}")
    injector = _setup_injector(xor_path, payload_bits, hash_bits)

    start = time.time()
    result = injector.inject(model, gamma=GAMMA)
    elapsed = time.time() - start
    log.info(f"[{payload_name}] inject() returned in {elapsed:.1f}s")

    if result is None:
        log.warning(f"[{payload_name}] CAPACITY EXCEEDED")
        del model
        gc.collect()
        return None, None, len(payload_bits), len(hash_bits), "capacity_exceeded"

    stego_sd, message_length, malware_length, hash_length = result
    torch.save(stego_sd, out_path)
    log.info(f"[{payload_name}] saved stegomodel: {out_path} (msg_len={message_length})")

    del model
    gc.collect()
    return out_path, message_length, malware_length, hash_length, None


def extract_snr_on_sd(sd, payload_name, malware_length, hash_length, message_length):
    shell = build_fresh_qwen()
    shell.load_state_dict(sd, strict=True)
    shell.eval()

    extractor = Extractor(
        seed=SEED, device=DEVICE, result_path=EXTRACT_DIR, logger=log,
        malware_length=malware_length, hash_length=hash_length,
        chunk_factor=CHUNK_FACTOR,
    )
    snr = extractor.extract(shell, message_length, payload_name, ret_snr=True)
    del shell, extractor
    gc.collect()
    return snr


def main(payload: str):
    xor_path = resolve_xor(payload)
    out_filename = f"qwen2.5-1.5b_none_{payload}_model.pt"

    # Sharded CSVs to avoid races between parallel runs.
    baseline_csv = RESULTS_DIR / f"maleficnet_qwen_baseline_snr_{payload}.csv"
    neuperm_csv = RESULTS_DIR / f"maleficnet_qwen_neuperm_snr_{payload}.csv"

    if not xor_path.exists():
        log.warning(f"[{payload}] no .xor file at {xor_path}")
        pd.DataFrame([dict(
            model_name=MODEL_NAME, payload_name=payload,
            quant_method='baseline', snr=None,
            filename=out_filename, error='missing_xor',
        )]).to_csv(baseline_csv, index=False)
        pd.DataFrame([dict(
            model_name=MODEL_NAME, payload_name=payload,
            quant_method='neuperm', snr=None,
            filename=out_filename, error='missing_xor',
        )]).to_csv(neuperm_csv, index=False)
        return

    # ---- Inject ----
    try:
        out_path, msg_len, mw_len, hs_len, cap_status = inject_payload(payload, xor_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        log.error(f"[{payload}] inject FAILED: {e}")
        pd.DataFrame([dict(
            model_name=MODEL_NAME, payload_name=payload,
            quant_method='baseline', snr=None,
            filename=out_filename, error=str(e),
        )]).to_csv(baseline_csv, index=False)
        pd.DataFrame([dict(
            model_name=MODEL_NAME, payload_name=payload,
            quant_method='neuperm', snr=None,
            filename=out_filename, error=str(e),
        )]).to_csv(neuperm_csv, index=False)
        return

    if cap_status == "capacity_exceeded":
        pd.DataFrame([dict(
            model_name=MODEL_NAME, payload_name=payload,
            quant_method='baseline', snr=None,
            filename=out_filename, error='capacity_exceeded',
        )]).to_csv(baseline_csv, index=False)
        pd.DataFrame([dict(
            model_name=MODEL_NAME, payload_name=payload,
            quant_method='neuperm', snr=None,
            filename=out_filename, error='capacity_exceeded',
        )]).to_csv(neuperm_csv, index=False)
        return

    # ---- Baseline extract ----
    try:
        stego_sd = torch.load(out_path, map_location='cpu', weights_only=False)
        snr_b = extract_snr_on_sd(stego_sd, payload, mw_len, hs_len, msg_len)
        log.info(f"[{payload}] BASELINE SNR = {snr_b:.4f}")
        pd.DataFrame([dict(
            model_name=MODEL_NAME, payload_name=payload,
            quant_method='baseline', snr=snr_b,
            filename=out_filename, error='',
        )]).to_csv(baseline_csv, index=False)
    except Exception as e:
        import traceback
        traceback.print_exc()
        log.error(f"[{payload}] baseline extract FAILED: {e}")
        pd.DataFrame([dict(
            model_name=MODEL_NAME, payload_name=payload,
            quant_method='baseline', snr=None,
            filename=out_filename, error=str(e),
        )]).to_csv(baseline_csv, index=False)
        stego_sd = None

    # ---- NeuPerm extract ----
    try:
        if stego_sd is None:
            stego_sd = torch.load(out_path, map_location='cpu', weights_only=False)
        permute_model(MODEL_NAME, stego_sd, inplace=True)
        snr_n = extract_snr_on_sd(stego_sd, payload, mw_len, hs_len, msg_len)
        log.info(f"[{payload}] NEUPERM SNR = {snr_n:.4f}")
        pd.DataFrame([dict(
            model_name=MODEL_NAME, payload_name=payload,
            quant_method='neuperm', snr=snr_n,
            filename=out_filename, error='',
        )]).to_csv(neuperm_csv, index=False)
    except Exception as e:
        import traceback
        traceback.print_exc()
        log.error(f"[{payload}] neuperm extract FAILED: {e}")
        pd.DataFrame([dict(
            model_name=MODEL_NAME, payload_name=payload,
            quant_method='neuperm', snr=None,
            filename=out_filename, error=str(e),
        )]).to_csv(neuperm_csv, index=False)

    log.info(f"[{payload}] DONE")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: exp_maleficnet_qwen_one.py <payload>")
        sys.exit(1)
    main(sys.argv[1])
