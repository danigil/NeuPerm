"""Resumable, seeded regeneration of the manuscript's Table 3.

Table 3 reports ImageNet12 (ILSVRC-2012 validation split, 50,000 images) top-1
accuracy for nine CNN architectures under each disruption method.  The original
runs behind that table were unseeded and, for vgg11/vgg16, left no artifact at
all, so the published cells cannot be reproduced or adjudicated.  This script
regenerates every cell from recorded seeds.

Design notes
------------
Resumability
    Every (model, condition, repeat) row is appended to the output CSV and
    fsynced the moment it completes, and any row already present is skipped on
    restart.  A multi-hour sweep that is interrupted loses at most the row that
    was in flight.

Seeding
    Seeds are derived deterministically from (seed_base, model, condition id,
    repeat) by BLAKE2b, so the whole sweep reproduces from the four CLI values
    alone and any single row can be regenerated in isolation.  The condition ids
    in ``CONDITIONS`` are part of that derivation: renaming one changes every
    seed downstream of it, so they are frozen.

Replication policy
    A prior audit measured evaluation nondeterminism at exactly zero: PTQ rows,
    a deterministic transform, showed sd = 0.000000 across five seeds on three
    benchmarks.  All observed variance therefore comes from the draw, not from
    the evaluation.  Deterministic conditions (``original`` and the three
    quantization settings) consequently run at n = 1 and are recorded as
    deterministic rather than under-sampled; stochastic conditions
    (``neuperm``, ``noise``, ``prune``) run at n = 5, each with a distinct
    recorded seed.

Transforms
    ``_noise_model``, ``_prune_model`` and the NeuPerm/quantization paths
    reproduce the semantics of ``experiments/exp1.py`` (``noise_model`` with
    ``extract_func='flatten'``, ``prune_model``, ``neuperm_model``) and of
    ``neu_perm.quantization.quantize_model``.  They are restated here rather
    than imported because ``exp1`` pulls in the transformers/datasets LLM stack
    at module scope, which this CNN-only sweep does not need.  Transforms run on
    CPU so that a given seed yields identical weights regardless of the
    evaluation device.

Usage
-----
    python experiments/exp_table3_sweep.py --models mobilenet_v3_small \
        --conditions original,neuperm --repeats 2 --out results/validation.csv
"""

import argparse
import copy
import csv
import hashlib
import os
import random
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neu_perm.perm import permute_model  # noqa: E402
from neu_perm.quantization import quantize_model  # noqa: E402


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

# The nine CNNs of Table 3, in manuscript order.
MODELS = [
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

# The twelve Table 3 columns.  ``cond_id`` is frozen: it is an input to the seed
# derivation and the resume key.  ``deterministic`` marks transforms with no RNG
# draw, which run at n = 1 by policy (see module docstring).
CONDITIONS = [
    {"cond_id": "original", "method": "original", "kwargs": {}, "deterministic": True},
    {"cond_id": "neuperm", "method": "neuperm", "kwargs": {}, "deterministic": False},
    {"cond_id": "noise_eps_1e-04", "method": "noise", "kwargs": {"eps": 1e-4}, "deterministic": False},
    {"cond_id": "noise_eps_1e-03", "method": "noise", "kwargs": {"eps": 1e-3}, "deterministic": False},
    {"cond_id": "noise_eps_1e-02", "method": "noise", "kwargs": {"eps": 1e-2}, "deterministic": False},
    {"cond_id": "noise_eps_1e-01", "method": "noise", "kwargs": {"eps": 1e-1}, "deterministic": False},
    {"cond_id": "prune_amount_0.01", "method": "prune", "kwargs": {"amount": 0.01}, "deterministic": False},
    {"cond_id": "prune_amount_0.05", "method": "prune", "kwargs": {"amount": 0.05}, "deterministic": False},
    {"cond_id": "quant_2bit_perchannel", "method": "quantization", "kwargs": {"n_bits": 2, "per_channel": True}, "deterministic": True},
    {"cond_id": "quant_4bit_perchannel", "method": "quantization", "kwargs": {"n_bits": 4, "per_channel": True}, "deterministic": True},
    {"cond_id": "quant_8bit_perchannel", "method": "quantization", "kwargs": {"n_bits": 8, "per_channel": True}, "deterministic": True},
]

CONDITIONS_BY_ID = {c["cond_id"]: c for c in CONDITIONS}

DATASET_NAME = "imagenet12"

# The local ImageNet-1k validation split: a directory of 1,000 wnid class
# folders. Override the default below with the NEUPERM_IMAGENET12_VAL env var.
DEFAULT_DATA_ROOT = "data/imagenet12/val"
DATA_ROOT_ENV = "NEUPERM_IMAGENET12_VAL"

# Columns of ``densenet121_imagenet12.csv`` (the exp1.py schema), plus the
# seed/repetition/cost fields Table 3 needs.  Additive: every original column
# keeps its original name and meaning.
FIELDNAMES = [
    "model_name",
    "accuracy",
    "time",            # transform wall-clock, as in exp1.py
    "dataset",
    "method",
    "method_kwargs",   # python-repr dict, as in exp1.py
    "cond_id",
    "repeat",
    "seed",
    "deterministic",
    "n_correct",
    "n_total",
    "eval_seconds",
    "batch_size",
    "device",
    "limit_batches",
    "torch_version",
    "git_commit",
    "timestamp",
]


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def derive_seed(seed_base, model_name, cond_id, repeat):
    """Deterministic per-row seed. Stable across machines and python versions."""
    key = f"{seed_base}|{model_name}|{cond_id}|{repeat}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Transforms (semantics of experiments/exp1.py)
# ---------------------------------------------------------------------------


def _extract_weights(model):
    return np.concatenate([w.cpu().detach().numpy().flatten() for w in model.parameters()])


def _load_flat_weights(model, flat):
    torch.nn.utils.vector_to_parameters(torch.from_numpy(flat.copy()), model.parameters())
    return model


def _noise_model(model, eps):
    """Additive Gaussian noise on the flattened parameter vector (exp1 'flatten')."""
    w = _extract_weights(model)
    w += np.random.normal(0, eps, w.shape).astype(w.dtype)
    return _load_flat_weights(model, w)


def _prune_model(model, amount):
    """Random unstructured pruning of every Conv2d/Linear weight (exp1).

    ``prune.remove`` bakes the mask into ``weight`` and drops the ``weight_orig``
    /``weight_mask`` reparametrisation.  This is numerically identical -- the
    resulting weight tensors are bitwise equal to the masked ones, verified
    across all 54 prunable resnet50 layers -- but it cuts the weight memory the
    model occupies on the GPU by roughly 2.3x (vgg16 prune peaked at 1597 MiB
    with the reparametrisation, 689 MiB without).  Without it the vgg prune rows
    are the memory high-water mark of the sweep and are the first thing to OOM
    on a shared GPU.
    """
    prunable = [m for _, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    for module in prunable:
        prune.random_unstructured(module, name="weight", amount=amount)
    for module in prunable:
        prune.remove(module, "weight")
    return model


def _neuperm_model(model, model_name):
    sd = model.state_dict()
    sd_perm = permute_model(model_name=model_name, sd=sd, inplace=True)
    model.load_state_dict(sd_perm)
    return model


def apply_condition(model, model_name, condition):
    """Return (transformed model, transform wall-clock seconds). Runs on CPU."""
    method, kwargs = condition["method"], condition["kwargs"]
    t0 = time.time()
    if method == "original":
        pass
    elif method == "neuperm":
        model = _neuperm_model(model, model_name)
    elif method == "noise":
        model = _noise_model(model, eps=kwargs["eps"])
    elif method == "prune":
        model = _prune_model(model, amount=kwargs["amount"])
    elif method == "quantization":
        model = quantize_model(model, n_bits=kwargs["n_bits"], per_channel=kwargs["per_channel"])
    else:
        raise ValueError(f"Unknown method: {method}")
    return model, time.time() - t0


# ---------------------------------------------------------------------------
# Model and data
# ---------------------------------------------------------------------------


def load_pretrained(model_name):
    """Load torchvision pretrained weights and the matching preprocessing."""
    weights = torchvision.models.get_model_weights(model_name).DEFAULT
    model = torchvision.models.get_model(model_name, weights=weights)
    return model.cpu().eval(), weights.transforms()


def build_loader(data_root, preprocess, batch_size, num_workers):
    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"ImageNet12 validation root not found: {data_root}. "
            f"Pass --data-root or set {DATA_ROOT_ENV}."
        )
    ds = torchvision.datasets.ImageFolder(data_root, transform=preprocess)
    if len(ds.classes) != 1000:
        raise ValueError(
            f"{data_root} has {len(ds.classes)} class directories, expected 1000. "
            "--data-root must point at the validation split itself, not its parent."
        )
    # shuffle=False: every row is evaluated on the identical image sequence, so
    # the only source of variation between repeats is the seeded transform.
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return ds, loader


def evaluate(model, loader, device, limit_batches=None):
    """Top-1 accuracy. Returns (n_correct, n_total, eval_seconds)."""
    model = model.to(device).eval()
    correct = total = 0
    t0 = time.time()
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            predicted = model(images).argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if limit_batches is not None and i + 1 >= limit_batches:
                break
    return correct, total, time.time() - t0


# ---------------------------------------------------------------------------
# Resumable CSV
# ---------------------------------------------------------------------------


def read_done_rows(path):
    """Return the set of (model_name, cond_id, repeat) already recorded."""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = set(FIELDNAMES) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} exists but is missing columns {sorted(missing)}; "
                "it was not written by this script. Use a different --out."
            )
        for row in reader:
            done.add((row["model_name"], row["cond_id"], int(row["repeat"])))
    return done


class RowWriter:
    """Append-and-fsync one row at a time, so an interruption loses at most one."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        is_new = not os.path.exists(path) or os.path.getsize(path) == 0
        self.fh = open(path, "a", newline="")
        self.writer = csv.DictWriter(self.fh, fieldnames=FIELDNAMES)
        if is_new:
            self.writer.writeheader()
            self._flush()

    def _flush(self):
        self.fh.flush()
        os.fsync(self.fh.fileno())

    def write(self, row):
        self.writer.writerow(row)
        self._flush()

    def close(self):
        self.fh.close()


def git_commit():
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Summary: mean, sd, 95% CI
# ---------------------------------------------------------------------------

SUMMARY_FIELDNAMES = [
    "model_name", "cond_id", "method", "method_kwargs", "n", "deterministic",
    "mean", "sd", "ci95_low", "ci95_high", "ci95_halfwidth", "seeds",
    "unit_of_replication", "replication_note",
]

UNIT_OF_REPLICATION = (
    "one full pass over the 50,000-image ImageNet-1k (ILSVRC-2012) validation "
    "split; one replicate = one independent seeded draw of the transform, "
    "evaluated on the identical fixed image sequence"
)


def summarize(rows_path, summary_path):
    """Aggregate per-cell mean/sd/95% CI. CI uses Student's t with df = n - 1."""
    from scipy import stats

    cells = OrderedDict()
    with open(rows_path, newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["model_name"], row["cond_id"])
            cells.setdefault(key, []).append(row)

    out_rows = []
    for (model_name, cond_id), rows in cells.items():
        rows = sorted(rows, key=lambda r: int(r["repeat"]))
        accs = np.array([float(r["accuracy"]) for r in rows], dtype=np.float64)
        n = len(accs)
        deterministic = rows[0]["deterministic"] == "True"
        mean = float(accs.mean())

        if n < 2:
            sd = ci_low = ci_high = half = ""
            note = (
                "deterministic transform; single evaluation is exact, not under-sampled"
                if deterministic
                else "STOCHASTIC CONDITION AT n=1 -- under-sampled, no CI available"
            )
        else:
            sd = float(accs.std(ddof=1))
            half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
            ci_low, ci_high = mean - half, mean + half
            note = (
                "deterministic transform evaluated more than once"
                if deterministic
                else f"{n} independent seeded draws"
            )

        out_rows.append({
            "model_name": model_name,
            "cond_id": cond_id,
            "method": rows[0]["method"],
            "method_kwargs": rows[0]["method_kwargs"],
            "n": n,
            "deterministic": deterministic,
            "mean": mean,
            "sd": sd,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "ci95_halfwidth": half,
            "seeds": ";".join(r["seed"] for r in rows),
            "unit_of_replication": UNIT_OF_REPLICATION,
            "replication_note": note,
        })

    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    log(f"\nPer-cell summary -> {summary_path}")
    log(f"{'model':<20} {'condition':<24} {'n':>2}  {'mean':>9}  {'sd':>9}  95% CI")
    for r in out_rows:
        ci = "n/a" if r["ci95_low"] == "" else f"[{r['ci95_low']:.5f}, {r['ci95_high']:.5f}]"
        sd = "n/a" if r["sd"] == "" else f"{r['sd']:.6f}"
        log(f"{r['model_name']:<20} {r['cond_id']:<24} {r['n']:>2}  {r['mean']:>9.5f}  {sd:>9}  {ci}")
    return out_rows


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run(args):
    conditions = [CONDITIONS_BY_ID[c] for c in args.conditions]
    done = set() if args.no_resume else read_done_rows(args.out)
    if done:
        log(f"Resuming: {len(done)} row(s) already in {args.out} will be skipped.")

    writer = RowWriter(args.out)
    commit = git_commit()
    total_eval_seconds = 0.0
    n_done = 0

    try:
        for model_name in args.models:
            # Plan first, so a fully-completed model costs no model load at all.
            plan = []
            for cond in conditions:
                n_reps = 1 if cond["deterministic"] else args.repeats
                for repeat in range(n_reps):
                    if (model_name, cond["cond_id"], repeat) not in done:
                        plan.append((cond, repeat))
            if not plan:
                log(f"\n{model_name}: all requested rows already present, skipping.")
                continue

            log(f"\n{'=' * 70}\n{model_name}: {len(plan)} row(s) to run\n{'=' * 70}")
            model_orig, preprocess = load_pretrained(model_name)
            sd_orig = copy.deepcopy(model_orig.state_dict())
            _, loader = build_loader(args.data_root, preprocess, args.batch_size, args.num_workers)

            for cond, repeat in plan:
                cond_id = cond["cond_id"]
                seed = derive_seed(args.seed_base, model_name, cond_id, repeat)
                set_all_seeds(seed)

                model = copy.deepcopy(model_orig)
                model.load_state_dict(sd_orig)
                model, transform_seconds = apply_condition(model, model_name, cond)

                correct, total, eval_seconds = evaluate(
                    model, loader, args.device, args.limit_batches
                )
                accuracy = correct / total
                total_eval_seconds += eval_seconds
                n_done += 1

                writer.write({
                    "model_name": model_name,
                    "accuracy": accuracy,
                    "time": transform_seconds,
                    "dataset": DATASET_NAME,
                    "method": cond["method"],
                    "method_kwargs": repr(cond["kwargs"]),
                    "cond_id": cond_id,
                    "repeat": repeat,
                    "seed": seed,
                    "deterministic": cond["deterministic"],
                    "n_correct": correct,
                    "n_total": total,
                    "eval_seconds": eval_seconds,
                    "batch_size": args.batch_size,
                    "device": args.device,
                    "limit_batches": "" if args.limit_batches is None else args.limit_batches,
                    "torch_version": torch.__version__,
                    "git_commit": commit,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })

                log(
                    f"  {cond_id:<24} rep={repeat} seed={seed:<11} "
                    f"acc={accuracy:.5f} ({correct}/{total})  "
                    f"transform={transform_seconds:.2f}s eval={eval_seconds:.1f}s"
                )

                del model
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

            del model_orig, sd_orig, loader
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    finally:
        writer.close()

    if n_done:
        log(
            f"\n{n_done} evaluation(s) this run, {total_eval_seconds / 60:.1f} min of "
            f"evaluation wall-clock, mean {total_eval_seconds / n_done:.1f}s per evaluation."
        )
    return n_done


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=",".join(MODELS),
                   help="Comma-separated model names (default: all nine Table 3 models).")
    p.add_argument("--conditions", default=",".join(c["cond_id"] for c in CONDITIONS),
                   help="Comma-separated condition ids (default: all twelve Table 3 columns).")
    p.add_argument("--repeats", type=int, default=5,
                   help="Repeats for STOCHASTIC conditions (default 5). Deterministic "
                        "conditions always run once, by the replication policy.")
    p.add_argument("--data-root", default=os.environ.get(DATA_ROOT_ENV, DEFAULT_DATA_ROOT),
                   help=f"ImageNet12 validation split (env {DATA_ROOT_ENV}, default {DEFAULT_DATA_ROOT}).")
    p.add_argument("--out", default="results/table3_sweep.csv", help="Row-level output CSV.")
    p.add_argument("--summary-out", default=None,
                   help="Per-cell summary CSV (default: <out> with a _summary suffix).")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore rows already in --out (they are still appended to, not truncated).")
    p.add_argument("--summarize-only", action="store_true", help="Re-derive the summary from --out and exit.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed-base", type=int, default=20260820,
                   help="Global seed base; every row seed derives from it.")
    p.add_argument("--limit-batches", type=int, default=None,
                   help="Evaluate only the first N batches. For timing probes and smoke "
                        "tests only -- the resulting accuracies are partial.")
    p.add_argument("--gpu-mem-fraction", type=float, default=None,
                   help="Cap this process at a fraction of total GPU memory, so an "
                        "over-large batch fails here instead of pressuring a co-tenant job.")
    args = p.parse_args(argv)

    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown_models = [m for m in args.models if m not in MODELS]
    unknown_conds = [c for c in args.conditions if c not in CONDITIONS_BY_ID]
    if unknown_models:
        p.error(f"Unknown model(s): {unknown_models}. Known: {MODELS}")
    if unknown_conds:
        p.error(f"Unknown condition(s): {unknown_conds}. Known: {list(CONDITIONS_BY_ID)}")
    if args.repeats < 1:
        p.error("--repeats must be >= 1")
    if args.limit_batches is not None and args.out == p.get_default("out"):
        p.error("--limit-batches produces partial accuracies; pass an explicit --out "
                "so probe rows cannot contaminate the sweep CSV.")
    if args.summary_out is None:
        base, ext = os.path.splitext(args.out)
        args.summary_out = f"{base}_summary{ext}"
    return args


def main(argv=None):
    args = parse_args(argv)

    if args.summarize_only:
        if not os.path.exists(args.out):
            raise FileNotFoundError(f"No rows to summarize: {args.out}")
        summarize(args.out, args.summary_out)
        return

    if args.gpu_mem_fraction is not None and args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)

    log(f"models     : {args.models}")
    log(f"conditions : {args.conditions}")
    log(f"repeats    : {args.repeats} (stochastic) / 1 (deterministic)")
    log(f"data root  : {args.data_root}")
    log(f"out        : {args.out}")
    log(f"seed base  : {args.seed_base}")
    log(f"device     : {args.device}  batch_size={args.batch_size}  workers={args.num_workers}")
    if args.limit_batches is not None:
        log(f"PROBE MODE : first {args.limit_batches} batch(es) only -- accuracies are partial")

    run(args)
    summarize(args.out, args.summary_out)


if __name__ == "__main__":
    main()
