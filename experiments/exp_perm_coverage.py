"""
Permutable-parameter coverage — the source for Table 1 (`tab:neuperm_cnns`).

Coverage is the fraction of a model's *learnable parameters* whose value changes
position when ``permute_model`` is applied:

    coverage = |{i in P : sd[i] != permute_model(sd)[i]}| / |P|
    P = the scalars belonging to entries of ``model.named_parameters()``

Two choices in that definition are deliberate, because they change the number and
were previously unrecorded (DR-20260816-neuperm-table1-correction-retracted):

1. **Pretrained weights.** Coverage is measured on the checkpoints a user actually
   applies NeuPerm to. Randomly initialised models measure ~1 pp lower on the
   MBConv architectures, because a random-init block has more scalars that
   coincide by value after being moved and so read as "unchanged".

2. **Learnable parameters only.** BatchNorm buffers (``running_mean``,
   ``running_var``, ``num_batches_tracked``) are excluded. They are permuted along
   with their layer, but at random initialisation they are constant vectors
   (all-zero and all-one), so permuting them changes nothing elementwise and they
   depress the ratio for a reason that has nothing to do with permutation
   coverage. Excluding them makes the measurement independent of that artifact.

The previously published series was measured on random-init weights *with*
buffers excluded, a basis recorded only in prose in NP2_SNR_INGEST_HANDOFF.md and
reproduced by no committed script. This file replaces that.

Usage:
    PYTHONPATH=. python experiments/exp_perm_coverage.py
    PYTHONPATH=. python experiments/exp_perm_coverage.py --models vgg16 resnet50
    PYTHONPATH=. python experiments/exp_perm_coverage.py --include-llms
    PYTHONPATH=. python experiments/exp_perm_coverage.py --basis random-init

Output: results/perm_coverage.csv (override with --out)
"""

import argparse
import copy
import csv
import os
import sys

import torch
import torchvision.models as tv_models

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neu_perm.perm import permute_model

CNN_MODELS = [
    "densenet121",
    "resnet50",
    "resnet101",
    "vgg11",
    "vgg16",
    "efficientnet_b0",
    "efficientnet_b4",
    "mobilenet_v2",
    "mobilenet_v3_small",
]

LLM_MODELS = ["llama-3.2-1b", "qwen2.5-1.5b"]

_LLM_IDS = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}

DEFAULT_SEED = 20260817
OUT_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "perm_coverage.csv")


def load_model(name: str, basis: str):
    """Return (state_dict, set_of_parameter_keys) for *name*."""
    if name in _LLM_IDS:
        from transformers import AutoModelForCausalLM

        if basis == "random-init":
            raise ValueError(
                f"{name}: random-init basis is not supported for LLMs; "
                "instantiating them without weights is not meaningful here"
            )
        model = AutoModelForCausalLM.from_pretrained(
            _LLM_IDS[name], torch_dtype=torch.float16
        )
    else:
        ctor = getattr(tv_models, name, None)
        if ctor is None:
            raise ValueError(f"torchvision.models has no attribute {name!r}")
        model = ctor(weights=None if basis == "random-init" else "DEFAULT")
    model.eval()
    param_keys = {k for k, _ in model.named_parameters()}
    return model.state_dict(), param_keys


def coverage(name: str, basis: str, seed: int):
    """Measure coverage for one model. Returns a result dict."""
    torch.manual_seed(seed)
    sd0_live, param_keys = load_model(name, basis)
    sd0 = {k: v.detach().clone() for k, v in sd0_live.items()}
    sd1 = permute_model(name, copy.deepcopy(dict(sd0)), inplace=True)
    if sd1 is None:
        raise RuntimeError(f"{name}: permute_model returned None")

    total = changed = 0
    for key, before in sd0.items():
        if key not in param_keys:
            continue
        after = sd1[key]
        n = before.numel()
        total += n
        if before.shape != after.shape:
            # A reshape would mean the permuter restructured the tensor rather
            # than reindexing it; count it whole and let the caller notice.
            changed += n
            continue
        changed += int((before != after).sum().item())

    return {
        "model_name": name,
        "basis": basis,
        "seed": seed,
        "n_parameters": total,
        "n_changed": changed,
        "coverage_pct": round(100.0 * changed / total, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--models", nargs="+", default=None,
                        help="subset to measure (default: all CNNs)")
    parser.add_argument("--include-llms", action="store_true",
                        help="also measure Llama-3.2-1B and Qwen2.5-1.5B")
    parser.add_argument("--basis", choices=["pretrained", "random-init"],
                        default="pretrained",
                        help="weights the measurement runs on (default: pretrained)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", default=OUT_CSV)
    args = parser.parse_args()

    models = args.models if args.models else list(CNN_MODELS)
    if args.include_llms and not args.models:
        models += LLM_MODELS

    rows = []
    for name in models:
        row = coverage(name, args.basis, args.seed)
        rows.append(row)
        print(
            f"{row['model_name']:<20} {row['n_changed']:>13,} / {row['n_parameters']:>13,}"
            f" = {row['coverage_pct']:>6.2f}%"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
