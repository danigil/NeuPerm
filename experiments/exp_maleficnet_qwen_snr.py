"""Inject MaleficNet payloads into Qwen2.5-1.5B and measure baseline + post-NeuPerm SNR.

Direct-injection (non-pretraining) variant: load clean Qwen, run Injector.inject(),
save stegomodel state_dict to disk, then Extractor.extract(ret_snr=True) for both
the baseline and post-NeuPerm versions.

Outputs two CSVs in RESULTS_DIR:
    maleficnet_qwen_baseline_snr.csv   (quant_method='baseline')
    maleficnet_qwen_neuperm_snr.csv    (quant_method='neuperm')

Schema matches existing maleficnet_ptq_snr.csv:
    model_name,payload_name,quant_method,snr,filename,error

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

# ---------------- Config ----------------
XOR_DIR = Path("/tmp/maleficnet_xor")  # writable staging for externally-built payloads
from payload_source import payload_dir  # noqa: E402
EXTRACT_DIR = Path("/tmp/maleficnet_extract")
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path("checkpoints/maleficnet")
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MSG_LENGTHS_CSV = MALEFICNET_DIR / "message_lengths.csv"

MODEL_NAME = "qwen2.5-1.5b"
CHUNK_FACTOR = 6
SEED = 42
GAMMA = 0.0009
DEVICE = "cpu"  # injector/extractor work on numpy state-dict flatten

# Time-box per payload — historical Llama-3.2-1B inject took ~10 min for small
# payloads and up to ~45 min for large ones. We give a soft warning at 30 min
# but do not abort, since the user wants full coverage when achievable.
TIMEBOX_SEC = 60 * 60

# Run all 9 payloads listed in the task
PAYLOADS = [
    "stuxnet", "asprox", "bladabindi", "destover", "kovter",
    "cerber", "ardamax", "zeus-bank", "eq.drug",
]

# Note: ardamax.xor and zeus-bank.xor are MISSING from the XOR zip
# (feasibility scout flagged this). Mark as capacity_exceeded / missing.


def write_xor_for_missing():
    """For ardamax and zeus-bank we don't have .xor on disk and the raw zip is
    elsewhere. Build them on-the-fly into /tmp/maleficnet_xor/ from
    maleficnet_malwares.zip (raw binaries). Stays in /tmp (gitignored)."""
    import zipfile
    raw_zip = MALEFICNET_DIR / "malware" / "maleficnet_malwares.zip"
    needed = ["ardamax", "zeus-bank"]
    if not raw_zip.exists():
        log.warning(f"Raw malware zip not found at {raw_zip}; "
                    f"ardamax / zeus-bank will be skipped")
        return
    # Malware zips from TheZoo are conventionally password-protected with 'infected'.
    PWD = b'infected'
    with zipfile.ZipFile(raw_zip, 'r') as zf:
        names_in_zip = zf.namelist()
        for name in needed:
            xor_path = XOR_DIR / f"{name}.xor"
            if xor_path.exists():
                continue
            # Find the matching entry in the zip
            match = None
            for n in names_in_zip:
                base = Path(n).name
                if base == name or base == f"{name}.bin" or Path(base).stem == name:
                    match = n
                    break
            if match is None:
                log.warning(f"  {name}: no raw binary in {raw_zip.name}")
                continue
            try:
                raw = zf.read(match, pwd=PWD)
            except RuntimeError as e:
                # Try without password (in case entry isn't encrypted)
                try:
                    raw = zf.read(match)
                except RuntimeError:
                    log.warning(f"  {name}: cannot decrypt {match}: {e}")
                    continue
            xored = bytes(b ^ 0xFF for b in raw)
            with open(xor_path, 'wb') as f:
                f.write(xored)
            log.info(f"  built {xor_path.name} from {match} (len={len(raw)} bytes)")


def load_payload_to_xor_path(payload_name: str) -> Path:
    """Return a path to a .xor file for the given payload.

    Prefers a payload staged in XOR_DIR (e.g. ardamax/zeus-bank built from the
    external raw zip on the lab machine); otherwise falls back to the shipped
    inert substitutes in <repo>/payloads/synthetic (or a real-originals dir set
    via NEUPERM_PAYLOAD_DIR / MALWARE_DIR). Returns None only when the payload is
    in neither location."""
    p = XOR_DIR / f"{payload_name}.xor"
    if p.exists():
        return p
    fallback = payload_dir() / f"{payload_name}.xor"
    if fallback.exists():
        return fallback
    return None


def load_payload_bits(xor_path: Path):
    """Decode XOR malware to bits + hash (in-memory only)."""
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


def inject_payload(payload_name: str, xor_path: Path):
    """Direct-injection (non-pretraining) path.

    1. Build fresh clean Qwen via LLMModel.
    2. Run Injector.inject() with gamma=0.0009 on the in-memory state_dict.
    3. Save the resulting stegomodel state_dict to CKPT_DIR.

    Returns (out_path, message_length, malware_length, hash_length, n_params)
    or (None, ..., "capacity_exceeded") on failure.
    """
    out_path = CKPT_DIR / f"qwen2.5-1.5b_none_{payload_name}_model.pt"
    if out_path.exists():
        log.info(f"[{payload_name}] stegomodel already on disk at {out_path}")
        # We still need message_length / hash_length for extraction.
        payload_bits, hash_bits = load_payload_bits(xor_path)
        # Recompute message length via the Injector path (no state mutation).
        clean = build_fresh_qwen()
        injector = Injector(
            seed=SEED, device=DEVICE, malware_path=xor_path,
            result_path=EXTRACT_DIR, logger=log, chunk_factor=CHUNK_FACTOR,
        )
        # The Injector.payload is set from xor_path which is XOR-encoded
        # — we must hand-substitute the decoded payload bits & rebuild .message.
        injector.payload = payload_bits
        injector.message = payload_bits + hash_bits
        msg_len = injector.get_message_length(clean)
        del clean
        gc.collect()
        return out_path, msg_len, len(payload_bits), len(hash_bits), None

    log.info(f"[{payload_name}] building fresh Qwen2.5-1.5B...")
    model = build_fresh_qwen()
    payload_bits, hash_bits = load_payload_bits(xor_path)
    log.info(f"[{payload_name}] payload bits = {len(payload_bits)}  hash bits = {len(hash_bits)}")

    injector = Injector(
        seed=SEED, device=DEVICE, malware_path=xor_path,
        result_path=EXTRACT_DIR, logger=log, chunk_factor=CHUNK_FACTOR,
    )
    # CRITICAL: Injector.__init__ reads the XOR-encoded bytes as the payload
    # (it calls bits_from_file(xor_path)). Override with the DECODED bits so
    # the embedded payload is the real malware bits, matching how the existing
    # Llama stegomodels were produced.
    injector.payload = payload_bits
    injector.message = payload_bits + hash_bits
    # Rebuild LDPC matrices for the correct message size — the constructor's
    # k=96 vs k=3048 branch depends on len(self.message). For the decoded
    # payload this is far above 4000 bits for all payloads, so k=3048 is
    # already correct (matches Llama runs).
    # Sanity: ensure the LDPC k matches what the constructor would pick.
    expected_k = 3048 if len(injector.message) > 4000 else 96
    if injector.G.shape[1] != expected_k:
        log.warning(f"[{payload_name}] LDPC k mismatch ({injector.G.shape[1]} vs {expected_k}); "
                    f"rebuilding")
        from pyldpc import make_ldpc
        d_v, d_c = 3, 12
        n = expected_k * (d_c // d_v)
        injector.H, injector.G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True, seed=SEED)

    start = time.time()
    result = injector.inject(model, gamma=GAMMA)
    elapsed = time.time() - start
    log.info(f"[{payload_name}] inject() returned in {elapsed:.1f}s")

    if result is None:
        # Spreading codes bigger than the model — capacity exceeded
        log.warning(f"[{payload_name}] CAPACITY EXCEEDED for Qwen2.5-1.5B")
        del model
        gc.collect()
        return None, None, len(payload_bits), len(hash_bits), "capacity_exceeded"

    stego_sd, message_length, malware_length, hash_length = result

    # The result is an OrderedDict already on CPU. Save it.
    torch.save(stego_sd, out_path)
    log.info(f"[{payload_name}] saved stegomodel: {out_path} (msg_len={message_length})")

    del model
    gc.collect()
    return out_path, message_length, malware_length, hash_length, None


def extract_snr_on_sd(sd, payload_name, malware_length, hash_length, message_length):
    """Build a fresh LLMModel shell, load the state_dict in, run extractor."""
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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Build .xor for ardamax / zeus-bank from raw zip (in-memory XOR), to /tmp.
    write_xor_for_missing()

    baseline_csv = RESULTS_DIR / "maleficnet_qwen_baseline_snr.csv"
    neuperm_csv = RESULTS_DIR / "maleficnet_qwen_neuperm_snr.csv"

    baseline_rows = []
    neuperm_rows = []

    # Resume support: load existing rows
    if baseline_csv.exists():
        baseline_rows = pd.read_csv(baseline_csv).to_dict('records')
        done_b = {r['payload_name'] for r in baseline_rows if pd.notna(r.get('snr'))}
        log.info(f"resuming baseline: {len(done_b)} payloads already done: {done_b}")
    else:
        done_b = set()
    if neuperm_csv.exists():
        neuperm_rows = pd.read_csv(neuperm_csv).to_dict('records')
        done_n = {r['payload_name'] for r in neuperm_rows if pd.notna(r.get('snr'))}
        log.info(f"resuming neuperm: {len(done_n)} payloads already done: {done_n}")
    else:
        done_n = set()

    for payload in PAYLOADS:
        log.info(f"\n{'='*70}\n PAYLOAD: {payload}\n{'='*70}")
        xor_path = load_payload_to_xor_path(payload)
        out_filename = f"qwen2.5-1.5b_none_{payload}_model.pt"

        if xor_path is None:
            log.warning(f"[{payload}] no .xor file; skipping")
            baseline_rows.append(dict(
                model_name=MODEL_NAME, payload_name=payload,
                quant_method='baseline', snr=None,
                filename=out_filename, error='missing_xor',
            ))
            neuperm_rows.append(dict(
                model_name=MODEL_NAME, payload_name=payload,
                quant_method='neuperm', snr=None,
                filename=out_filename, error='missing_xor',
            ))
            pd.DataFrame(baseline_rows).to_csv(baseline_csv, index=False)
            pd.DataFrame(neuperm_rows).to_csv(neuperm_csv, index=False)
            continue

        # ---- INJECTION ----
        t_inj_start = time.time()
        try:
            out_path, msg_len, mw_len, hs_len, cap_status = inject_payload(payload, xor_path)
        except Exception as e:
            import traceback
            traceback.print_exc()
            log.error(f"[{payload}] inject FAILED: {e}")
            baseline_rows.append(dict(
                model_name=MODEL_NAME, payload_name=payload,
                quant_method='baseline', snr=None,
                filename=out_filename, error=str(e),
            ))
            neuperm_rows.append(dict(
                model_name=MODEL_NAME, payload_name=payload,
                quant_method='neuperm', snr=None,
                filename=out_filename, error=str(e),
            ))
            pd.DataFrame(baseline_rows).to_csv(baseline_csv, index=False)
            pd.DataFrame(neuperm_rows).to_csv(neuperm_csv, index=False)
            continue

        if cap_status == "capacity_exceeded":
            baseline_rows.append(dict(
                model_name=MODEL_NAME, payload_name=payload,
                quant_method='baseline', snr=None,
                filename=out_filename, error='capacity_exceeded',
            ))
            neuperm_rows.append(dict(
                model_name=MODEL_NAME, payload_name=payload,
                quant_method='neuperm', snr=None,
                filename=out_filename, error='capacity_exceeded',
            ))
            pd.DataFrame(baseline_rows).to_csv(baseline_csv, index=False)
            pd.DataFrame(neuperm_rows).to_csv(neuperm_csv, index=False)
            continue

        inj_elapsed = time.time() - t_inj_start
        if inj_elapsed > TIMEBOX_SEC:
            log.warning(f"[{payload}] injection took {inj_elapsed:.0f}s > {TIMEBOX_SEC}s; "
                        f"continuing anyway (already finished)")

        # ---- BASELINE EXTRACTION ----
        if payload in done_b:
            log.info(f"[{payload}] baseline SNR already in CSV, skipping")
        else:
            try:
                stego_sd = torch.load(out_path, map_location='cpu', weights_only=False)
                snr_b = extract_snr_on_sd(stego_sd, payload, mw_len, hs_len, msg_len)
                log.info(f"[{payload}] BASELINE SNR = {snr_b:.4f}")
                baseline_rows.append(dict(
                    model_name=MODEL_NAME, payload_name=payload,
                    quant_method='baseline', snr=snr_b,
                    filename=out_filename, error='',
                ))
                # Keep stego_sd for the NeuPerm pass below to avoid re-loading
            except Exception as e:
                import traceback
                traceback.print_exc()
                log.error(f"[{payload}] baseline extract FAILED: {e}")
                baseline_rows.append(dict(
                    model_name=MODEL_NAME, payload_name=payload,
                    quant_method='baseline', snr=None,
                    filename=out_filename, error=str(e),
                ))
                stego_sd = None
            pd.DataFrame(baseline_rows).to_csv(baseline_csv, index=False)

        # ---- NEUPERM EXTRACTION ----
        if payload in done_n:
            log.info(f"[{payload}] neuperm SNR already in CSV, skipping")
        else:
            try:
                if 'stego_sd' not in dir() or stego_sd is None:
                    stego_sd = torch.load(out_path, map_location='cpu', weights_only=False)
                # permute in-place — this is the disruption step.
                permute_model(MODEL_NAME, stego_sd, inplace=True)
                snr_n = extract_snr_on_sd(stego_sd, payload, mw_len, hs_len, msg_len)
                log.info(f"[{payload}] NEUPERM SNR = {snr_n:.4f}")
                neuperm_rows.append(dict(
                    model_name=MODEL_NAME, payload_name=payload,
                    quant_method='neuperm', snr=snr_n,
                    filename=out_filename, error='',
                ))
            except Exception as e:
                import traceback
                traceback.print_exc()
                log.error(f"[{payload}] neuperm extract FAILED: {e}")
                neuperm_rows.append(dict(
                    model_name=MODEL_NAME, payload_name=payload,
                    quant_method='neuperm', snr=None,
                    filename=out_filename, error=str(e),
                ))
            pd.DataFrame(neuperm_rows).to_csv(neuperm_csv, index=False)

        # Cleanup
        try:
            del stego_sd
        except Exception:
            pass
        gc.collect()

    log.info("\nDONE. CSVs:")
    log.info(f"  {baseline_csv}")
    log.info(f"  {neuperm_csv}")


if __name__ == "__main__":
    main()
