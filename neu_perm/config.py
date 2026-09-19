import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

DATASETS_DIR = os.environ.get("NEUPERM_DATASETS_DIR", "data")

IMAGENET12_ROOT = os.environ.get("IMAGENET12_ROOT", "data/imagenet12")


def require_imagenet12() -> str:
    """Validate IMAGENET12_ROOT and return it. Call from scripts that read the dataset.

    Deliberately NOT enforced at import time. ``neu_perm.config`` is imported by
    ``neu_perm.data_loaders`` and by ~19 experiments that never touch ImageNet -- the
    overhead scripts, the LLM scripts, the SNR scripts -- and a reproducibility-package
    user re-running the overhead table must not be blocked by a dataset they do not
    need. A constant shared by every experiment is not a self-contained check, so per
    the project's coding-style rule the failure fires at the earliest point that can
    resolve it: the consuming script.

    The previous guard tested only for the literal placeholder ``"<path_to_imagenet12>"``,
    so a path that did not exist passed validation silently and surfaced only when a
    sweep died on it -- the missing-referent defect the coding-style rules name.
    """
    if not os.path.isdir(IMAGENET12_ROOT):
        raise FileNotFoundError(
            f"IMAGENET12_ROOT points at {IMAGENET12_ROOT!r}, which is not a directory. "
            "Set it in neu_perm/config.py to the ImageNet-12 root (the directory "
            "holding 'val/')."
        )
    val_dir = os.path.join(IMAGENET12_ROOT, "val")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"IMAGENET12_ROOT is {IMAGENET12_ROOT!r} but {val_dir!r} is missing. "
            "The validation split must be a directory of per-class subdirectories."
        )
    return IMAGENET12_ROOT
