import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# Set via the NEUPERM_IMAGENET12_ROOT environment variable, or edit the default
# below, to point at a directory holding the ImageNet-12 validation dataset.
# Only the CNN-accuracy experiment needs this; importing config for RESULTS_DIR
# must not fail, so the check is deferred to get_imagenet12_root() at point of use.
IMAGENET12_ROOT = os.environ.get("NEUPERM_IMAGENET12_ROOT", "<path_to_imagenet12>")


def get_imagenet12_root() -> str:
    """Return the ImageNet-12 validation root, failing loudly if unset."""
    if IMAGENET12_ROOT == "<path_to_imagenet12>":
        raise ValueError(
            "IMAGENET12_ROOT is not set. Set the NEUPERM_IMAGENET12_ROOT "
            "environment variable or edit neu_perm/config.py to point at your "
            "ImageNet-12 validation dataset."
        )
    return IMAGENET12_ROOT