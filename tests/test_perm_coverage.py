"""
Regression test for the Table 1 permutable-coverage series (REPRODUCTION_GUIDE.md:467).

Coverage is the fraction of a model's *learnable parameters* whose value changes
position when ``permute_model`` is applied. The measurement lives in
``experiments/exp_perm_coverage.py``; this test pins its output so a change in the
permuter cannot silently move a published number. That script's docstring carries
the full rationale for the two basis choices, which are:

* **pretrained weights** — the checkpoints a user actually applies NeuPerm to;
* **learnable parameters only** — BatchNorm buffers excluded, because at random
  initialisation they are constant vectors and permuting them changes nothing
  elementwise, which depresses the ratio for a reason unrelated to coverage.

The series below was measured on 2026-08-17 against ``fcca9cf`` at seed 20260817:

    densenet121 77.02 | resnet50 80.86 | resnet101 88.81 | vgg11 100.00
    vgg16 99.99 | efficientnet_b0 67.80 | efficientnet_b4 86.42
    mobilenet_v2 51.47 | mobilenet_v3_small 81.35

These are the values the manuscript's Table 1 should print, rounded. They differ
from the previously published series (76 / 80 / 88 / 100 / 100 / 67 / 86 / 51 / 81)
on four rows, each by one percentage point upward: densenet121, resnet50,
resnet101 and efficientnet_b0. The earlier series was measured on randomly
initialised weights, a basis recorded only in prose in NP2_SNR_INGEST_HANDOFF.md
and reproduced by no committed script — see
DR-20260816-neuperm-table1-correction-retracted.

Tolerance is +/-1 percentage point, as the reproduction guide specifies. Coverage
under this basis is close to deterministic: the spread over permutation seeds is
well inside the tolerance, so a failure here is a real change in the permuter,
not sampling noise.

This test downloads pretrained weights on first run.
"""

import pytest

from experiments.exp_perm_coverage import coverage

# Measured Table 1 series, in percent, pretrained weights, parameters only.
EXPECTED_COVERAGE = {
    "densenet121": 77.02,
    "resnet50": 80.86,
    "resnet101": 88.81,
    "vgg11": 100.00,
    "vgg16": 99.99,
    "efficientnet_b0": 67.80,
    "efficientnet_b4": 86.42,
    "mobilenet_v2": 51.47,
    "mobilenet_v3_small": 81.35,
}

TOLERANCE_PP = 1.0
BASIS = "pretrained"
SEED = 20260817


@pytest.mark.parametrize("model_name,expected", sorted(EXPECTED_COVERAGE.items()))
def test_coverage_matches_table1(model_name: str, expected: float) -> None:
    measured = coverage(model_name, BASIS, SEED)["coverage_pct"]
    assert abs(measured - expected) <= TOLERANCE_PP, (
        f"{model_name}: coverage {measured:.2f}% is more than {TOLERANCE_PP} pp "
        f"from the pinned {expected:.2f}%. If the permuter changed deliberately, "
        f"re-run experiments/exp_perm_coverage.py, update this series, and update "
        f"Table 1 in the manuscript."
    )


def test_mbconv_architectures_are_ring_permuted() -> None:
    """The MBConv architectures must be well above their pre-merge coverage.

    Before the np2 merge (``fcca9cf``) the permuter reordered only the
    squeeze-and-excitation hidden dimension and a few cross-block chains on these
    three architectures, giving roughly 26 / 16 / 20 percent. This guards against
    a regression to that behaviour, which would not be caught by the tolerance
    above alone if the pinned series were ever edited to match.
    """
    for model_name, floor in (
        ("efficientnet_b0", 60.0),
        ("efficientnet_b4", 80.0),
        ("mobilenet_v2", 45.0),
    ):
        measured = coverage(model_name, BASIS, SEED)["coverage_pct"]
        assert measured > floor, (
            f"{model_name}: coverage {measured:.2f}% is below {floor}%, which "
            f"suggests the MBConv expanded-dimension ring is no longer permuted"
        )
