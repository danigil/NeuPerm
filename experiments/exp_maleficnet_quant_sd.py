"""Apply per-channel quantization and NeuPerm to MaleficNet models, save state dicts.

Operates directly on state dicts — loads the original MaleficNet .pt files,
applies quantize-dequantize (skipping BatchNorm parameters), and resaves.

For each .pt model in the input directory, produces:
  - {basename}_ptq8_perchannel.pt
  - {basename}_ptq4_perchannel.pt
  - {basename}_neuperm.pt

The state dicts retain the 'model.' prefix from MaleficNet's Model wrapper.
"""
import copy, os, sys, time
from collections import OrderedDict
from pathlib import Path

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neu_perm.quantization import quantize_dequantize_sd, get_bn_keys_from_sd
from neu_perm.perm import permute_model

INPUT_DIR = Path("/tmp/maleficnet_models")
OUTPUT_DIR = Path("checkpoints/maleficnet/quant_sd")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUANT_CONFIGS = [
    (8, True,  "ptq8_perchannel"),
    (4, True,  "ptq4_perchannel"),
]

# Map filename model prefix to perm_map key
FILENAME_TO_PERM_NAME = {
    "densenet": "densenet121",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "vgg11": "vgg11",
    "vgg16": "vgg16",
}


def strip_prefix(sd, prefix="model."):
    return OrderedDict((k[len(prefix):] if k.startswith(prefix) else k, v) for k, v in sd.items())


def add_prefix(sd, prefix="model."):
    return OrderedDict((prefix + k, v) for k, v in sd.items())


def get_perm_name(filename):
    for prefix, perm_name in FILENAME_TO_PERM_NAME.items():
        if filename.startswith(prefix + "_"):
            return perm_name
    return None


def log(msg):
    print(msg, flush=True)


def main():
    pt_files = sorted(INPUT_DIR.glob("*.pt"))
    log(f"Found {len(pt_files)} .pt files in {INPUT_DIR}")

    for pt_path in pt_files:
        basename = pt_path.stem
        perm_name = get_perm_name(pt_path.name)
        if perm_name is None:
            log(f"  SKIP {pt_path.name} (unknown model)")
            continue

        log(f"\n{'='*60}\n{pt_path.name} (perm_name={perm_name})\n{'='*60}")

        sd_full = torch.load(pt_path, map_location="cpu")
        has_prefix = any(k.startswith("model.") for k in sd_full.keys())
        sd_bare = strip_prefix(sd_full) if has_prefix else sd_full

        # Identify BN keys to skip during quantization
        bn_keys = get_bn_keys_from_sd(sd_bare)
        log(f"  BN keys to skip: {len(bn_keys)}")

        # Per-channel quantization configs
        for n_bits, per_channel, suffix in QUANT_CONFIGS:
            t0 = time.time()
            sd_q = quantize_dequantize_sd(
                sd_bare, n_bits=n_bits, per_channel=per_channel,
                inplace=False, skip_keys=bn_keys,
            )
            elapsed = time.time() - t0

            sd_out = add_prefix(sd_q) if has_prefix else sd_q
            out_path = OUTPUT_DIR / f"{basename}_{suffix}.pt"
            torch.save(sd_out, out_path)
            log(f"  {suffix}: saved ({elapsed:.2f}s)")

        # NeuPerm
        t0 = time.time()
        sd_perm = permute_model(perm_name, copy.deepcopy(sd_bare), inplace=True)
        elapsed = time.time() - t0

        sd_out = add_prefix(sd_perm) if has_prefix else sd_perm
        out_path = OUTPUT_DIR / f"{basename}_neuperm.pt"
        torch.save(sd_out, out_path)
        log(f"  neuperm: saved ({elapsed:.2f}s)")

    log(f"\nDone. Output in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
