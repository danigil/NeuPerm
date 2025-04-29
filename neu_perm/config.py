import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

IMAGENET12_ROOT = "<path_to_imagenet12>"

if IMAGENET12_ROOT == "<path_to_imagenet12>":
    raise ValueError(
        "IMAGENET12_ROOT is not set. Please set it in neu_perm/config.py to the path of your ImageNet12 dataset."
    )