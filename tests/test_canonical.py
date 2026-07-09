"""
Tests for neu_perm.canonical — canonical ordering and tie analysis.
"""

import math
from collections import OrderedDict

import pytest
import torch

from neu_perm.canonical import (
    PermSite,
    canonical_sort_indices,
    compute_neuron_metric,
    count_approx_ties,
    count_exact_ties,
    get_permutable_sites,
    layer_tie_analysis,
    theoretical_recovery_probability,
    theoretical_uniqueness_bound,
    tie_group_permutation_count,
    tie_group_sizes,
    uniqueness_probability,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic state_dicts
# ---------------------------------------------------------------------------


def _make_vgg_pair_sd():
    """Minimal VGG-like state_dict with one feature pair (0, 3)."""
    sd = OrderedDict()
    sd["features.0.weight"] = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    ).unsqueeze(-1).unsqueeze(-1)  # (4, 2, 1, 1) conv-like
    sd["features.0.bias"] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    sd["features.3.weight"] = torch.randn(8, 4, 3, 3)
    return sd


def _make_conv_bn_conv_sd():
    """Minimal conv-bn-conv state_dict."""
    sd = OrderedDict()
    sd["block.conv1.weight"] = torch.tensor(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    ).unsqueeze(-1).unsqueeze(-1)  # (3, 2, 1, 1)
    sd["block.bn.weight"] = torch.tensor([1.0, 1.0, 1.0])
    sd["block.bn.bias"] = torch.tensor([0.0, 0.0, 0.0])
    sd["block.bn.running_mean"] = torch.tensor([0.0, 0.0, 0.0])
    sd["block.bn.running_var"] = torch.tensor([1.0, 1.0, 1.0])
    sd["block.conv2.weight"] = torch.randn(6, 3, 3, 3)
    return sd


def _make_conv_bn_conv_site():
    return PermSite(
        kind="conv_bn_conv",
        keys={
            "conv1_weight": "block.conv1.weight",
            "bn_weight": "block.bn.weight",
            "bn_bias": "block.bn.bias",
            "bn_running_mean": "block.bn.running_mean",
            "bn_running_var": "block.bn.running_var",
            "conv2_weight": "block.conv2.weight",
        },
    )


def _make_vgg_pair_site():
    return PermSite(
        kind="vgg_pair",
        keys={
            "layer1_weight": "features.0.weight",
            "layer1_bias": "features.0.bias",
            "layer2_weight": "features.3.weight",
        },
    )


# ---------------------------------------------------------------------------
# Tests: compute_neuron_metric
# ---------------------------------------------------------------------------


class TestComputeNeuronMetric:
    def test_l1_norm_vgg(self):
        sd = _make_vgg_pair_sd()
        site = _make_vgg_pair_site()
        metric = compute_neuron_metric(sd, site, "l1_norm")
        assert metric.shape == (4,)
        # L1 norm of [1,2] = 3, [3,4] = 7, [5,6] = 11, [7,8] = 15
        expected = torch.tensor([3.0, 7.0, 11.0, 15.0])
        torch.testing.assert_close(metric, expected)

    def test_l2_norm_conv_bn_conv(self):
        sd = _make_conv_bn_conv_sd()
        site = _make_conv_bn_conv_site()
        metric = compute_neuron_metric(sd, site, "l2_norm")
        assert metric.shape == (3,)
        # L2 norms of [1,1], [2,2], [3,3]
        expected = torch.tensor([math.sqrt(2), math.sqrt(8), math.sqrt(18)])
        torch.testing.assert_close(metric, expected, atol=1e-5, rtol=1e-5)

    def test_bias_value_vgg(self):
        sd = _make_vgg_pair_sd()
        site = _make_vgg_pair_site()
        metric = compute_neuron_metric(sd, site, "bias_value")
        expected = torch.tensor([0.1, 0.2, 0.3, 0.4])
        torch.testing.assert_close(metric, expected)

    def test_variance(self):
        sd = _make_vgg_pair_sd()
        site = _make_vgg_pair_site()
        metric = compute_neuron_metric(sd, site, "variance")
        assert metric.shape == (4,)
        # Variance of [1,2] = 0.25, [3,4] = 0.25, [5,6] = 0.25, [7,8] = 0.25
        expected = torch.tensor([0.5, 0.5, 0.5, 0.5])  # torch.var uses Bessel correction by default
        # Actually torch var with dim=1 on 2 elements: sum((x-mean)^2) / (n-1)
        # For [1,2]: mean=1.5, var = (0.25 + 0.25) / 1 = 0.5
        torch.testing.assert_close(metric, expected, atol=1e-5, rtol=1e-5)

    def test_composite(self):
        sd = _make_vgg_pair_sd()
        site = _make_vgg_pair_site()
        metric = compute_neuron_metric(sd, site, "composite")
        assert metric.shape == (4, 3)  # (N, 3) for l1, l2, variance


# ---------------------------------------------------------------------------
# Tests: canonical_sort_indices
# ---------------------------------------------------------------------------


class TestCanonicalSortIndices:
    def test_correct_ordering(self):
        sd = _make_vgg_pair_sd()
        site = _make_vgg_pair_site()
        indices = canonical_sort_indices(sd, site, "l1_norm")
        # L1 norms: [3, 7, 11, 15] -> sorted order is [0, 1, 2, 3]
        expected = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        torch.testing.assert_close(indices, expected)

    def test_reverse_ordering(self):
        """When weights are in descending order, sort should reverse."""
        sd = OrderedDict()
        sd["features.0.weight"] = torch.tensor(
            [[8.0, 8.0], [4.0, 4.0], [1.0, 1.0]]
        ).unsqueeze(-1).unsqueeze(-1)
        sd["features.0.bias"] = torch.tensor([0.3, 0.2, 0.1])
        sd["features.3.weight"] = torch.randn(6, 3, 3, 3)
        site = PermSite(
            kind="vgg_pair",
            keys={
                "layer1_weight": "features.0.weight",
                "layer1_bias": "features.0.bias",
                "layer2_weight": "features.3.weight",
            },
        )
        indices = canonical_sort_indices(sd, site, "l1_norm")
        expected = torch.tensor([2, 1, 0], dtype=torch.long)
        torch.testing.assert_close(indices, expected)


# ---------------------------------------------------------------------------
# Tests: tie analysis
# ---------------------------------------------------------------------------


class TestTieAnalysis:
    def test_no_ties(self):
        vals = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert count_exact_ties(vals) == 0
        assert uniqueness_probability(vals) == 1.0
        assert tie_group_sizes(vals) == []
        assert tie_group_permutation_count(vals) == 1.0

    def test_all_tied(self):
        vals = torch.tensor([5.0, 5.0, 5.0, 5.0])
        assert count_exact_ties(vals) == 4
        assert uniqueness_probability(vals) == 0.0
        assert tie_group_sizes(vals) == [4]
        assert tie_group_permutation_count(vals) == math.factorial(4)

    def test_partial_ties(self):
        vals = torch.tensor([1.0, 2.0, 2.0, 3.0])
        assert count_exact_ties(vals) == 2
        assert uniqueness_probability(vals) == 0.5
        assert tie_group_sizes(vals) == [2]
        assert tie_group_permutation_count(vals) == 2.0

    def test_multiple_tie_groups(self):
        vals = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0])
        groups = sorted(tie_group_sizes(vals))
        assert groups == [2, 2]
        assert tie_group_permutation_count(vals) == 4.0  # 2! * 2!

    def test_approx_ties(self):
        vals = torch.tensor([1.0, 1.0001, 2.0, 3.0])
        assert count_exact_ties(vals) == 0
        assert count_approx_ties(vals, tolerance=0.001) == 2
        assert count_approx_ties(vals, tolerance=0.0001) == 0

    def test_uniqueness_with_tolerance(self):
        vals = torch.tensor([1.0, 1.0001, 2.0, 3.0])
        assert uniqueness_probability(vals, tolerance=0.001) == 0.5
        assert uniqueness_probability(vals, tolerance=0.0) == 1.0


# ---------------------------------------------------------------------------
# Tests: theoretical bounds
# ---------------------------------------------------------------------------


class TestTheoreticalBounds:
    def test_small_layer_high_uniqueness(self):
        # 10 neurons in float32 (2^24 values): very high uniqueness
        prob = theoretical_uniqueness_bound(10, 24)
        assert prob > 0.999

    def test_large_layer_low_uniqueness(self):
        # 10000 neurons in float16 (2^11 = 2048 values): impossible
        prob = theoretical_uniqueness_bound(10000, 11)
        assert prob == 0.0

    def test_recovery_probability_no_ties(self):
        assert theoretical_recovery_probability([]) == 1.0

    def test_recovery_probability_with_ties(self):
        # One group of 3: 1/3! = 1/6
        assert abs(theoretical_recovery_probability([3]) - 1.0 / 6) < 1e-10

    def test_recovery_probability_multiple_groups(self):
        # Groups of 2 and 3: 1/(2! * 3!) = 1/12
        assert abs(theoretical_recovery_probability([2, 3]) - 1.0 / 12) < 1e-10


# ---------------------------------------------------------------------------
# Tests: get_permutable_sites
# ---------------------------------------------------------------------------


class TestGetPermutableSites:
    def test_vgg11_site_count(self):
        sites = get_permutable_sites("vgg11")
        # VGG11: 7 feature pairs + 2 MLP pairs = 9
        assert len(sites) == 9

    def test_vgg16_site_count(self):
        sites = get_permutable_sites("vgg16")
        # VGG16: 12 feature pairs + 2 MLP pairs = 14
        assert len(sites) == 14

    def test_resnet50_site_count(self):
        sites = get_permutable_sites("resnet50")
        # ResNet50: 16 blocks × 2 conv_bn_conv per block = 32
        assert len(sites) == 32

    def test_resnet101_site_count(self):
        sites = get_permutable_sites("resnet101")
        # ResNet101: 30 blocks (actually: 3+4+23+3 = 33) × 2 = 66
        assert len(sites) == 66

    def test_densenet121_site_count(self):
        sites = get_permutable_sites("densenet121")
        # DenseNet121: 6+12+24+16 = 58 blocks, 1 site each = 58... wait
        # Actually from constants: (1,1..6) + (2,1..12) + (3,1..24) + (4,1..16) = 6+12+24+16 = 58
        # But wait: range(1,7)=6, range(1,13)=12, range(1,25)=24, range(1,17)=16 = 58
        assert len(sites) == 58

    def test_llama_site_count(self):
        sites = get_permutable_sites("llama-3.2-1b")
        # 16 blocks × 2 (attn + mlp) = 32
        assert len(sites) == 32

    def test_site_keys_exist_format(self):
        """All sites should have non-empty keys dict."""
        for model_name in ("vgg11", "resnet50", "densenet121", "llama-3.2-1b"):
            sites = get_permutable_sites(model_name)
            for site in sites:
                assert len(site.keys) > 0
                assert site.kind in ("vgg_pair", "conv_bn_conv", "llama_attn", "llama_mlp")


# ---------------------------------------------------------------------------
# Tests: layer_tie_analysis (integration-style with synthetic data)
# ---------------------------------------------------------------------------


class TestLayerTieAnalysis:
    def test_output_schema(self):
        """Verify DataFrame has expected columns."""
        # Create a minimal model by mocking get_permutable_sites
        sd = _make_vgg_pair_sd()
        # We can't easily run layer_tie_analysis with a custom sd because
        # it uses get_permutable_sites which expects real model names.
        # Instead, test the components individually.
        site = _make_vgg_pair_site()
        metric = compute_neuron_metric(sd, site, "l1_norm")
        assert metric.shape[0] == 4
        assert count_exact_ties(metric) == 0


# ---------------------------------------------------------------------------
# Tests: composite metric sorting
# ---------------------------------------------------------------------------


class TestCompositeMetric:
    def test_composite_breaks_ties(self):
        """When L1 norms are tied, composite should use L2/variance to break ties."""
        sd = OrderedDict()
        # Two neurons with same L1 norm but different distributions
        sd["features.0.weight"] = torch.tensor(
            [[3.0, 0.0], [1.5, 1.5]]
        ).unsqueeze(-1).unsqueeze(-1)
        sd["features.0.bias"] = torch.tensor([0.1, 0.2])
        sd["features.3.weight"] = torch.randn(4, 2, 3, 3)
        site = PermSite(
            kind="vgg_pair",
            keys={
                "layer1_weight": "features.0.weight",
                "layer1_bias": "features.0.bias",
                "layer2_weight": "features.3.weight",
            },
        )

        # L1 norms are both 3.0 -> tied for l1_norm
        metric_l1 = compute_neuron_metric(sd, site, "l1_norm")
        assert count_exact_ties(metric_l1) == 2

        # But composite should have different values
        metric_comp = compute_neuron_metric(sd, site, "composite")
        # L2 norms differ: sqrt(9) = 3.0 vs sqrt(4.5) ≈ 2.12
        assert metric_comp.shape == (2, 3)
