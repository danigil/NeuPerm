"""Experiment 2: NeuPerm destroys a MaleficNet-class spread-spectrum payload.

Reproduces Table 5 (`tab:exp2_in12`) and Fig 3 (`fig:exp2_diff_new`): the
signal quality of a spread-spectrum payload hidden in model parameters, measured
before and after NeuPerm, across CNNs and LLMs. NeuPerm's permutation reorders
the neurons the payload is keyed to, so the correlator can no longer recover it.

The script runs in one of two modes, selected by ``MODE`` in the ``__main__``
block:

  MODE = "spread_spectrum_llm"  (DEFAULT, self-contained, runs out of the box)
      Embeds a spread-spectrum payload into an LLM with ``neu_perm.steganography``
      and reports extraction bit-error-rate (BER) clean vs. after NeuPerm. This
      is an independent reimplementation of the MaleficNet-class spread-spectrum
      embed/extract — no external fork, no real malware. Framed as an ablation.

  MODE = "maleficnet_fork"
      Full MaleficNet injector/extractor SNR on CNNs and LLMs, using the external
      MIT MaleficNet fork. Injects a payload with the fork's Injector, then
      measures the fork's CDMA extraction SNR clean vs. after NeuPerm (and, if
      configured, under noise / PTQ / pruning). Requires the external fork and a
      payload directory; both are located via environment variables (below). It
      never ships or names real malware — payloads are read generically from
      ``NEUPERM_PAYLOAD_DIR`` (default: the repo's ``payloads/`` canaries).

Environment variables (maleficnet_fork mode only):
  NEUPERM_MALEFICNET_DIR  Path to the external MIT MaleficNet fork; inserted on
                          sys.path. If unset/missing the mode prints guidance and
                          exits cleanly (see payloads/README.md and the
                          reproduction guide) — it does NOT crash on import.
  NEUPERM_PAYLOAD_DIR     Directory of payload files, read generically by name.
                          Default: the repo's ``payloads/`` directory (the canary
                          ``*.bin`` files produced by ``payloads/make_canary.py``).
                          A ``<name>.bin`` is read raw; a ``<name>.xor`` is
                          XOR-0xFF decoded in memory (real-malware reproduction).

Run:

    python experiments/exp2_maleficnet_snr.py            # default SS-LLM mode
    NEUPERM_MALEFICNET_DIR=/path/to/mymalefic \
        python experiments/exp2_maleficnet_snr.py        # after setting MODE

Results are written to one CSV under ``RESULTS_DIR`` (default
``results/exp2_snr.csv``), one row per (mode, model, payload, condition, metric).
"""
import copy
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from neu_perm.config import REPO_ROOT, RESULTS_DIR
from neu_perm.perm import permute_model

# Unified result-CSV schema (one row per measurement).
CSV_COLUMNS = [
    "mode", "model", "payload", "condition", "metric", "value",
    "repeat", "seed", "gamma", "error",
]


def _write_rows(rows: List[dict], csv_path: str) -> None:
    """Persist the accumulated result rows to ``csv_path`` (overwrite)."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(csv_path, index=False)


# ===========================================================================
# MODE 1: spread_spectrum_llm  (self-contained, default)
# ===========================================================================
#
# Folds exp_spreadspectrum_llm.py: embed a random spread-spectrum payload into
# an LLM via neu_perm.steganography, measure extraction BER clean vs. after
# NeuPerm. Independent reimpl of the MaleficNet-class scheme; no fork, no malware.


def run_spread_spectrum_llm(
    models: List[Tuple[str, str]],
    n_bits: int,
    amplitude: float,
    n_repeats: int,
    payload_seed: int,
    embed_seed: int,
    csv_path: str,
) -> pd.DataFrame:
    """Spread-spectrum LLM ablation: BER clean vs. after NeuPerm.

    Parameters
    ----------
    models : list of (model_key, hf_id)
        ``model_key`` must be a key ``permute_model`` / the steganography module
        understand (e.g. ``"llama-3.2-1b"``); ``hf_id`` is the HuggingFace repo.
    n_bits : int
        Payload size in bits.
    amplitude : float
        Spread-spectrum embedding strength.
    n_repeats : int
        Number of independent NeuPerm permutations to average over.
    payload_seed, embed_seed : int
        Seeds for the random payload and the spreading code.
    """
    from transformers import AutoModelForCausalLM

    from neu_perm.steganography import (
        compute_ber,
        generate_payload,
        spread_spectrum_embed,
        spread_spectrum_extract,
    )

    payload_name = f"spread_spectrum_{n_bits}bit"
    rows: List[dict] = []

    for model_key, hf_id in models:
        print(f"=== {model_key} ({hf_id}) ===", flush=True)
        m = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16)
        sd_orig = copy.deepcopy(m.cpu().state_dict())
        del m
        torch.cuda.empty_cache()

        payload = generate_payload(n_bits, seed=payload_seed)
        sd_stego = spread_spectrum_embed(
            sd_orig, model_key, payload, amplitude=amplitude, seed=embed_seed, inplace=False
        )

        # Clean extraction (sanity): should recover the payload near-perfectly.
        rec_clean = spread_spectrum_extract(sd_stego, model_key, sd_orig, n_bits)
        ber_clean = compute_ber(payload, rec_clean)
        rows.append(dict(
            mode="spread_spectrum_llm", model=model_key, payload=payload_name,
            condition="clean", metric="ber", value=ber_clean, repeat=-1,
            seed=embed_seed, gamma="", error="",
        ))
        print(f"  clean BER = {ber_clean:.4f}", flush=True)
        _write_rows(rows, csv_path)

        # Post-NeuPerm extraction: permutation scrambles the payload coordinates.
        for rep in range(n_repeats):
            torch.manual_seed(rep)
            sd_p = permute_model(model_key, copy.deepcopy(sd_stego), inplace=True)
            rec_p = spread_spectrum_extract(sd_p, model_key, sd_orig, n_bits)
            ber_p = compute_ber(payload, rec_p)
            rows.append(dict(
                mode="spread_spectrum_llm", model=model_key, payload=payload_name,
                condition="after_neuperm", metric="ber", value=ber_p, repeat=rep,
                seed=rep, gamma="", error="",
            ))
            print(f"  rep {rep}: BER after NeuPerm = {ber_p:.4f}", flush=True)
            del sd_p
            _write_rows(rows, csv_path)

        del sd_orig, sd_stego
        torch.cuda.empty_cache()

    print(f"wrote {csv_path}", flush=True)
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


# ===========================================================================
# MODE 2: maleficnet_fork  (external MIT MaleficNet fork required)
# ===========================================================================
#
# Folds the 10 fork scripts (baseline/extract/quant SNR + the per-CNN and Qwen
# injection scripts) into one inject -> disrupt -> extract loop. All external
# paths are env-var driven; payloads are read generically by name.

LLM_MODEL_NAMES = {"llama-3.2-1b", "qwen2.5-1.5b"}


def _payload_dir() -> Path:
    """Payload directory from NEUPERM_PAYLOAD_DIR (default: repo payloads/)."""
    return Path(os.environ.get("NEUPERM_PAYLOAD_DIR", os.path.join(REPO_ROOT, "payloads")))


def _resolve_payload_file(payload_dir: Path, name: str) -> Path:
    """Resolve a payload name to a file in ``payload_dir``.

    Prefers ``<name>.bin`` (canary, read raw), then ``<name>.xor`` (real sample,
    XOR-0xFF decoded at read time), then a bare ``<name>``. Fails loudly if none
    exist so a missing payload can't silently corrupt an SNR measurement.
    """
    for candidate in (f"{name}.bin", f"{name}.xor", name):
        p = payload_dir / candidate
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No payload file for {name!r} in {payload_dir} "
        f"(expected {name}.bin canary or {name}.xor). Run payloads/make_canary.py."
    )


def _install_xor_bits_patch(utils_bit, injector_mod) -> None:
    """Patch the fork's ``bits_from_file`` to XOR-0xFF decode ``.xor`` payloads.

    Canary ``.bin`` files are read raw by the original function; ``.xor`` files
    (real-malware reproduction) are decoded in memory only — never written back
    to a tracked path. Patches both the ``utils_bit`` name and the copy already
    imported into ``injector``.
    """
    original = utils_bit.bits_from_file

    def _xor_bits_from_file(path):
        if str(path).endswith(".xor"):
            with open(path, "rb") as f:
                raw = bytes(b ^ 0xFF for b in f.read())
            return utils_bit.bits_from_bytes(raw)
        return original(path)

    utils_bit.bits_from_file = _xor_bits_from_file
    injector_mod.bits_from_file = _xor_bits_from_file


def _build_model(fork, model_name: str, pretrained: bool):
    """Build a carrier model: torchvision CNN or the fork's LLMModel."""
    if model_name in LLM_MODEL_NAMES:
        return fork.LLMModel(only_pretrained=True, model=model_name)
    import torchvision
    if pretrained:
        weights = torchvision.models.get_model_weights(model_name).DEFAULT
        return torchvision.models.get_model(model_name, weights=weights)
    return torchvision.models.get_model(model_name, weights=None)


def _extract_snr(fork, model_name: str, sd, payload_name: str,
                 message_length: int, malware_length: int, hash_length: int,
                 seed: int, chunk_factor: int, extract_dir: Path) -> float:
    """Load ``sd`` into a fresh carrier shell and run the fork's SNR extractor."""
    shell = _build_model(fork, model_name, pretrained=False)
    shell.load_state_dict(sd)
    shell.eval()
    extractor = fork.Extractor(
        seed=seed, device="cpu", result_path=extract_dir, logger=fork.log,
        malware_length=malware_length, hash_length=hash_length,
        chunk_factor=chunk_factor,
    )
    return extractor.extract(shell, message_length, payload_name, ret_snr=True)


def _apply_noise(sd, eps: float):
    """Add zero-mean Gaussian noise (std ``eps``) to every float tensor."""
    out = copy.deepcopy(sd)
    for k, v in out.items():
        if torch.is_tensor(v) and v.is_floating_point():
            np_dtype = {torch.float16: np.float16, torch.float32: np.float32,
                        torch.float64: np.float64}.get(v.dtype, np.float32)
            out[k] = v + torch.from_numpy(np.random.normal(0, eps, tuple(v.shape)).astype(np_dtype))
    return out


def _apply_ptq(sd, n_bits: int):
    """Per-channel post-training quantize/dequantize, skipping BatchNorm params."""
    from neu_perm.quantization import get_bn_keys_from_sd, quantize_dequantize_sd
    bn_keys = get_bn_keys_from_sd(sd)
    return quantize_dequantize_sd(sd, n_bits=n_bits, per_channel=True, inplace=False, skip_keys=bn_keys)


def _apply_prune(model_name: str, sd, amount: float):
    """Random unstructured pruning of Conv2d/Linear weights (CNNs only)."""
    import torch.nn as nn
    import torch.nn.utils.prune as prune
    import torchvision
    m = torchvision.models.get_model(model_name, weights=None)
    m.load_state_dict(copy.deepcopy(sd))
    for _, module in m.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.random_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return m.state_dict()


def _load_fork():
    """Import the external MaleficNet fork or return None with guidance.

    Returns a namespace object exposing Injector, Extractor, LLMModel, log, or
    ``None`` if NEUPERM_MALEFICNET_DIR is unset/missing (message already printed).
    """
    fork_dir = os.environ.get("NEUPERM_MALEFICNET_DIR")
    if not fork_dir or not os.path.isdir(fork_dir):
        print(
            "MODE 'maleficnet_fork' needs the external MIT MaleficNet fork.\n"
            "  Set NEUPERM_MALEFICNET_DIR to the fork checkout, e.g.:\n"
            "    NEUPERM_MALEFICNET_DIR=/path/to/mymalefic python experiments/exp2_maleficnet_snr.py\n"
            "  Payloads come from NEUPERM_PAYLOAD_DIR (default: this repo's payloads/;\n"
            "  run payloads/make_canary.py first). See payloads/README.md and the\n"
            "  reproduction guide. Exiting cleanly.",
            flush=True,
        )
        return None

    sys.path.insert(0, os.path.abspath(fork_dir))
    import logging

    import injector as injector_mod
    import utils.utils_bit as utils_bit
    from extractor import Extractor
    from injector import Injector

    warnings.filterwarnings("ignore")
    logging.getLogger("PIL").setLevel(logging.CRITICAL)
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    if not log.handlers:
        log.addHandler(logging.StreamHandler())

    _install_xor_bits_patch(utils_bit, injector_mod)

    class _Fork:
        pass

    fork = _Fork()
    fork.Injector = Injector
    fork.Extractor = Extractor
    fork.log = log
    try:
        from models.llms import LLMModel
        fork.LLMModel = LLMModel
    except Exception:  # LLM support optional; only needed for LLM jobs.
        fork.LLMModel = None
    return fork


def run_maleficnet_fork(
    jobs: List[Tuple[str, str, float]],
    seed: int,
    chunk_factor: int,
    noise_epsilons: List[float],
    ptq_bits: List[int],
    prune_amounts: List[float],
    csv_path: str,
) -> Optional[pd.DataFrame]:
    """Inject each payload, then measure fork-extraction SNR under each condition.

    ``jobs`` is a list of ``(model_name, payload_name, gamma)``. The primary two
    conditions are ``clean`` (injected, undisrupted) and ``after_neuperm``; the
    ``noise_*`` / ``ptq*`` / ``prune_*`` grids are optional (empty by default).
    Returns ``None`` if the external fork is unavailable.
    """
    fork = _load_fork()
    if fork is None:
        return None

    np.random.seed(seed)
    torch.manual_seed(seed)

    payload_dir = _payload_dir()
    extract_dir = Path(tempfile.mkdtemp(prefix="neuperm_snr_"))
    print(f"payload dir: {payload_dir}\nextract scratch: {extract_dir}", flush=True)

    rows: List[dict] = []

    def measure(model_name, payload_name, gamma, sd, condition, message_length,
                malware_length, hash_length):
        try:
            snr = _extract_snr(fork, model_name, sd, payload_name, message_length,
                               malware_length, hash_length, seed, chunk_factor, extract_dir)
            print(f"  {model_name}/{payload_name} [{condition}]: SNR = {snr:.4f}", flush=True)
            err = ""
        except Exception as e:  # record failure, keep going (fail-loud per row)
            import traceback
            traceback.print_exc()
            snr, err = None, str(e)
            print(f"  {model_name}/{payload_name} [{condition}]: ERROR {e}", flush=True)
        rows.append(dict(
            mode="maleficnet_fork", model=model_name, payload=payload_name,
            condition=condition, metric="snr", value=snr, repeat=0,
            seed=seed, gamma=gamma, error=err,
        ))
        _write_rows(rows, csv_path)

    for model_name, payload_name, gamma in jobs:
        print(f"\n{'='*60}\n=== {model_name} / {payload_name} (gamma={gamma}) ===\n{'='*60}", flush=True)
        is_llm = model_name in LLM_MODEL_NAMES
        if is_llm and fork.LLMModel is None:
            print(f"  SKIP {model_name}: fork has no LLM support (models.llms)", flush=True)
            continue

        try:
            payload_path = _resolve_payload_file(payload_dir, payload_name)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}", flush=True)
            rows.append(dict(
                mode="maleficnet_fork", model=model_name, payload=payload_name,
                condition="clean", metric="snr", value=None, repeat=0,
                seed=seed, gamma=gamma, error="missing_payload",
            ))
            _write_rows(rows, csv_path)
            continue

        # --- inject ---
        model = _build_model(fork, model_name, pretrained=True)
        model.eval()
        injector = fork.Injector(
            seed=seed, device="cpu", malware_path=payload_path,
            result_path=extract_dir, logger=fork.log, chunk_factor=chunk_factor,
        )
        malware_length = len(injector.payload)
        hash_length = len(injector.hash)
        print(f"  payload={malware_length} bits, hash={hash_length} bits", flush=True)

        result = injector.inject(model, gamma=gamma)
        if result is None:  # spreading codes exceed model capacity
            print(f"  CAPACITY EXCEEDED for {model_name}/{payload_name}", flush=True)
            rows.append(dict(
                mode="maleficnet_fork", model=model_name, payload=payload_name,
                condition="clean", metric="snr", value=None, repeat=0,
                seed=seed, gamma=gamma, error="capacity_exceeded",
            ))
            _write_rows(rows, csv_path)
            del model, injector
            continue

        new_sd, message_length = result[0], result[1]
        sd_injected = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                       for k, v in new_sd.items()}
        del model, new_sd

        # --- clean (baseline) ---
        measure(model_name, payload_name, gamma, sd_injected, "clean",
                message_length, malware_length, hash_length)

        # --- after NeuPerm (the disruption under test) ---
        sd_np = permute_model(model_name, copy.deepcopy(sd_injected), inplace=True)
        measure(model_name, payload_name, gamma, sd_np, "after_neuperm",
                message_length, malware_length, hash_length)
        del sd_np

        # --- optional comparison disruptions (CNN only for prune) ---
        for eps in noise_epsilons:
            sd_n = _apply_noise(sd_injected, eps)
            measure(model_name, payload_name, gamma, sd_n, f"noise_{eps}",
                    message_length, malware_length, hash_length)
            del sd_n
        for nb in ptq_bits:
            sd_q = _apply_ptq(sd_injected, nb)
            measure(model_name, payload_name, gamma, sd_q, f"ptq{nb}_perchannel",
                    message_length, malware_length, hash_length)
            del sd_q
        if not is_llm:
            for amt in prune_amounts:
                sd_p = _apply_prune(model_name, sd_injected, amt)
                measure(model_name, payload_name, gamma, sd_p, f"prune_{amt}",
                        message_length, malware_length, hash_length)
                del sd_p

        del sd_injected, injector

    print(f"\nwrote {csv_path}", flush=True)
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


if __name__ == "__main__":
    # --- run configuration ---
    # Which mode to run. "spread_spectrum_llm" is self-contained and runs out of
    # the box; "maleficnet_fork" needs the external fork (see module docstring).
    MODE = "spread_spectrum_llm"

    # Output CSV (unified schema, one row per measurement).
    CSV_PATH = os.path.join(RESULTS_DIR, "exp2_snr.csv")

    # -- spread_spectrum_llm settings --
    SS_MODELS = [
        ("llama-3.2-1b", "meta-llama/Llama-3.2-1B-Instruct"),
        ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct"),
    ]
    SS_N_BITS = 1024        # payload size in bits
    SS_AMPLITUDE = 1e-4     # spread-spectrum embedding strength
    SS_N_REPEATS = 5        # NeuPerm permutations averaged over
    SS_PAYLOAD_SEED = 42
    SS_EMBED_SEED = 123

    # -- maleficnet_fork settings --
    # (model_name, payload_name, gamma). payload_name resolves to a file in
    # NEUPERM_PAYLOAD_DIR (<name>.bin canary by default). CNNs use torchvision
    # pretrained weights; LLMs use the fork's LLMModel.
    FORK_JOBS = [
        ("efficientnet_b0", "stuxnet", 0.005),
        ("efficientnet_b4", "stuxnet", 0.005),
        ("mobilenet_v2", "stuxnet", 0.005),
        ("mobilenet_v3_small", "stuxnet_t16", 0.005),
        ("vgg16", "stuxnet", 0.0009),
        ("qwen2.5-1.5b", "stuxnet", 0.0009),
    ]
    FORK_SEED = 42
    FORK_CHUNK_FACTOR = 6
    # Optional comparison-disruption grids (empty = only clean + after_neuperm).
    FORK_NOISE_EPSILONS: List[float] = []
    FORK_PTQ_BITS: List[int] = []
    FORK_PRUNE_AMOUNTS: List[float] = []
    # -------------------------

    if MODE == "spread_spectrum_llm":
        run_spread_spectrum_llm(
            models=SS_MODELS, n_bits=SS_N_BITS, amplitude=SS_AMPLITUDE,
            n_repeats=SS_N_REPEATS, payload_seed=SS_PAYLOAD_SEED,
            embed_seed=SS_EMBED_SEED, csv_path=CSV_PATH,
        )
    elif MODE == "maleficnet_fork":
        run_maleficnet_fork(
            jobs=FORK_JOBS, seed=FORK_SEED, chunk_factor=FORK_CHUNK_FACTOR,
            noise_epsilons=FORK_NOISE_EPSILONS, ptq_bits=FORK_PTQ_BITS,
            prune_amounts=FORK_PRUNE_AMOUNTS, csv_path=CSV_PATH,
        )
    else:
        raise ValueError(
            f"Unknown MODE {MODE!r}; expected 'spread_spectrum_llm' or 'maleficnet_fork'"
        )
