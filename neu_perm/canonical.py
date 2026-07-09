"""
Canonical ordering analysis for the adaptive attack on NeuPerm.

Implements functions to compute canonical orderings of neurons/channels
using permutation-invariant metrics (L1-norm, L2-norm, bias, variance),
analyze tie collisions, and compute theoretical uniqueness bounds.
"""

import copy
import math
from collections import OrderedDict
from functools import partial, reduce
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from neu_perm.perm import (
    DENSENET121_ALL_BLOCKS,
    LLAMA3_2_1B_ALL_BLOCKS,
    QWEN2_5_1_5B_ALL_BLOCKS,
    RESNET50_ALL_BLOCKS,
    RESNET101_ALL_BLOCKS,
    VGG_11_ALL_FEATURES,
    VGG_11_ALL_MLPS,
    VGG_16_FEATURES,
    VGG_16_MLPS,
)

# ---------------------------------------------------------------------------
# Metric names
# ---------------------------------------------------------------------------
METRIC_NAMES = ("l1_norm", "l2_norm", "bias_value", "variance", "composite")

MetricName = Literal["l1_norm", "l2_norm", "bias_value", "variance", "composite"]

# ---------------------------------------------------------------------------
# Permutable layer descriptors
#
# Each entry is a tuple describing one permutation site:
#   For CNNs: ("conv_bn_conv", conv1_prefix, bn_prefix|None, conv2_prefix, None)
#   For VGG feature/mlp pairs: ("linear_pair", layer1_prefix, layer2_prefix, None, None)
#   For Llama attention: ("llama_attn", layer_prefix, None, None, {"num_heads": ..., "gqa": True})
#   For Llama MLP: ("llama_mlp", layer_prefix, None, None, None)
# ---------------------------------------------------------------------------

PermSiteKind = Literal[
    "conv_bn_conv",
    "vgg_pair",
    "llama_attn",
    "llama_mlp",
]


class PermSite:
    """Describes a single permutation site in a model."""

    def __init__(
        self,
        kind: PermSiteKind,
        keys: Dict[str, str],
        group_size: int = 1,
        extra: Optional[Dict] = None,
    ):
        self.kind = kind
        self.keys = keys  # relevant state_dict key prefixes
        self.group_size = group_size  # >1 for head-level permutation
        self.extra = extra or {}

    def __repr__(self):
        return f"PermSite(kind={self.kind!r}, keys={self.keys!r}, group_size={self.group_size})"


# ---------------------------------------------------------------------------
# Architecture-specific layer pair extraction
# ---------------------------------------------------------------------------


def _vgg_sites(features_pairs, mlp_pairs, key_prefix=""):
    """Build PermSite list for VGG architectures."""
    sites = []
    for ftr1, ftr2 in features_pairs:
        layer1 = f"{key_prefix}features.{ftr1}"
        layer2 = f"{key_prefix}features.{ftr2}"
        sites.append(
            PermSite(
                kind="vgg_pair",
                keys={
                    "layer1_weight": f"{layer1}.weight",
                    "layer1_bias": f"{layer1}.bias",
                    "layer2_weight": f"{layer2}.weight",
                },
            )
        )
    for mlp1, mlp2 in mlp_pairs:
        layer1 = f"{key_prefix}classifier.{mlp1}"
        layer2 = f"{key_prefix}classifier.{mlp2}"
        sites.append(
            PermSite(
                kind="vgg_pair",
                keys={
                    "layer1_weight": f"{layer1}.weight",
                    "layer1_bias": f"{layer1}.bias",
                    "layer2_weight": f"{layer2}.weight",
                },
            )
        )
    return sites


def _resnet_sites(blocks, key_prefix=""):
    """Build PermSite list for ResNet architectures."""
    sites = []
    for block_id in blocks:
        block = f"{key_prefix}layer{block_id}"
        # conv1 -> bn1 -> conv2
        sites.append(
            PermSite(
                kind="conv_bn_conv",
                keys={
                    "conv1_weight": f"{block}.conv1.weight",
                    "bn_weight": f"{block}.bn1.weight",
                    "bn_bias": f"{block}.bn1.bias",
                    "bn_running_mean": f"{block}.bn1.running_mean",
                    "bn_running_var": f"{block}.bn1.running_var",
                    "conv2_weight": f"{block}.conv2.weight",
                },
            )
        )
        # conv2 -> bn2 -> conv3
        sites.append(
            PermSite(
                kind="conv_bn_conv",
                keys={
                    "conv1_weight": f"{block}.conv2.weight",
                    "bn_weight": f"{block}.bn2.weight",
                    "bn_bias": f"{block}.bn2.bias",
                    "bn_running_mean": f"{block}.bn2.running_mean",
                    "bn_running_var": f"{block}.bn2.running_var",
                    "conv2_weight": f"{block}.conv3.weight",
                },
            )
        )
    return sites


def _densenet121_sites(blocks=None, key_prefix=""):
    """Build PermSite list for DenseNet-121."""
    if blocks is None:
        blocks = DENSENET121_ALL_BLOCKS
    sites = []
    for block_idx, layer_idx in blocks:
        prefix = f"{key_prefix}features.denseblock{block_idx}.denselayer{layer_idx}"
        conv1 = f"{prefix}.conv1"
        bn2 = f"{prefix}.norm2"
        conv2 = f"{prefix}.conv2"
        sites.append(
            PermSite(
                kind="conv_bn_conv",
                keys={
                    "conv1_weight": f"{conv1}.weight",
                    "bn_weight": f"{bn2}.weight",
                    "bn_bias": f"{bn2}.bias",
                    "bn_running_mean": f"{bn2}.running_mean",
                    "bn_running_var": f"{bn2}.running_var",
                    "conv2_weight": f"{conv2}.weight",
                },
            )
        )
    return sites


def _llama_sites(blocks=None, key_prefix="", num_heads=32):
    """Build PermSite list for Llama-style decoder transformers (Llama, Qwen2)."""
    if blocks is None:
        blocks = LLAMA3_2_1B_ALL_BLOCKS
    sites = []
    for i in blocks:
        layer = f"{key_prefix}model.layers.{i}"
        # Attention (GQA head-level permutation)
        sites.append(
            PermSite(
                kind="llama_attn",
                keys={
                    "q_proj_weight": f"{layer}.self_attn.q_proj.weight",
                    "k_proj_weight": f"{layer}.self_attn.k_proj.weight",
                    "v_proj_weight": f"{layer}.self_attn.v_proj.weight",
                    "o_proj_weight": f"{layer}.self_attn.o_proj.weight",
                },
                extra={"num_heads": num_heads, "gqa": True},
            )
        )
        # MLP (hidden-dim permutation)
        sites.append(
            PermSite(
                kind="llama_mlp",
                keys={
                    "gate_weight": f"{layer}.mlp.gate_proj.weight",
                    "up_weight": f"{layer}.mlp.up_proj.weight",
                    "down_weight": f"{layer}.mlp.down_proj.weight",
                },
            )
        )
    return sites


_sites_map = {
    "vgg11": partial(_vgg_sites, VGG_11_ALL_FEATURES, VGG_11_ALL_MLPS),
    "vgg16": partial(_vgg_sites, VGG_16_FEATURES, VGG_16_MLPS),
    "resnet50": partial(_resnet_sites, RESNET50_ALL_BLOCKS),
    "resnet101": partial(_resnet_sites, RESNET101_ALL_BLOCKS),
    "densenet121": partial(_densenet121_sites, DENSENET121_ALL_BLOCKS),
    "llama-3.2-1b": partial(_llama_sites, LLAMA3_2_1B_ALL_BLOCKS, num_heads=32),
    "qwen2.5-1.5b": partial(_llama_sites, QWEN2_5_1_5B_ALL_BLOCKS, num_heads=12),
}


def get_permutable_sites(model_name: str, key_prefix: str = "") -> List[PermSite]:
    """Return all permutation sites for *model_name*."""
    return _sites_map[model_name](key_prefix=key_prefix)


# ---------------------------------------------------------------------------
# Neuron / channel metric computation
# ---------------------------------------------------------------------------


def compute_neuron_metric(
    sd: OrderedDict,
    site: PermSite,
    metric_name: MetricName,
) -> torch.Tensor:
    """Compute a 1-D metric tensor (one value per permutable unit) for *site*.

    For CNN layers the permutable unit is an output channel of the first conv.
    For Llama attention it is a KV head group.
    For Llama MLP it is a hidden-dim neuron of the gate projection.
    """
    if metric_name == "composite":
        parts = []
        for m in ("l1_norm", "l2_norm", "variance"):
            parts.append(compute_neuron_metric(sd, site, m))
        return torch.stack(parts, dim=-1)  # (N, 3)

    if site.kind in ("conv_bn_conv", "vgg_pair"):
        return _metric_for_cnn_site(sd, site, metric_name)
    if site.kind == "llama_attn":
        return _metric_for_llama_attn(sd, site, metric_name)
    if site.kind == "llama_mlp":
        return _metric_for_llama_mlp(sd, site, metric_name)
    raise ValueError(f"Unknown site kind: {site.kind}")


def _scalar_metric(w: torch.Tensor, metric_name: str) -> torch.Tensor:
    """Reduce a weight tensor to one scalar per first-dimension slice."""
    # w shape: (N, ...) -> reduce over dims 1..
    n = w.shape[0]
    flat = w.reshape(n, -1).float()
    if metric_name == "l1_norm":
        return flat.abs().sum(dim=1)
    if metric_name == "l2_norm":
        return flat.norm(p=2, dim=1)
    if metric_name == "variance":
        return flat.var(dim=1)
    if metric_name == "bias_value":
        # bias is already 1-D
        return flat.squeeze(1)
    raise ValueError(f"Unknown scalar metric: {metric_name}")


def _metric_for_cnn_site(
    sd: OrderedDict, site: PermSite, metric_name: str
) -> torch.Tensor:
    if site.kind == "vgg_pair":
        w = sd[site.keys["layer1_weight"]]
        if metric_name == "bias_value":
            return sd[site.keys["layer1_bias"]].float()
        return _scalar_metric(w, metric_name)

    # conv_bn_conv
    w = sd[site.keys["conv1_weight"]]
    if metric_name == "bias_value":
        # Use BN weight (gamma) as a proxy — always exists for conv_bn_conv
        return sd[site.keys["bn_weight"]].float()
    return _scalar_metric(w, metric_name)


def _metric_for_llama_attn(
    sd: OrderedDict, site: PermSite, metric_name: str
) -> torch.Tensor:
    """Metric per KV head group for Llama GQA attention."""
    k_weight = sd[site.keys["k_proj_weight"]]  # (kv_dim, embed_dim)
    v_weight = sd[site.keys["v_proj_weight"]]
    kv_dim = k_weight.shape[0]

    q_weight = sd[site.keys["q_proj_weight"]]  # (embed_dim, embed_dim)
    embed_dim = q_weight.shape[0]
    num_heads = site.extra.get("num_heads", 32)
    head_dim = embed_dim // num_heads
    n_kv_heads = kv_dim // head_dim

    # Reshape K to (n_kv_heads, head_dim, embed_dim), compute metric per head
    k_heads = k_weight.reshape(n_kv_heads, head_dim, -1)

    if metric_name == "bias_value":
        # No biases in Llama projections; fall back to L1 norm
        return _scalar_metric(k_heads.reshape(n_kv_heads, -1), "l1_norm")

    return _scalar_metric(k_heads.reshape(n_kv_heads, -1), metric_name)


def _metric_for_llama_mlp(
    sd: OrderedDict, site: PermSite, metric_name: str
) -> torch.Tensor:
    """Metric per hidden-dim neuron of the Llama MLP gate projection."""
    gate_w = sd[site.keys["gate_weight"]]  # (hidden_dim, embed_dim)
    if metric_name == "bias_value":
        return _scalar_metric(gate_w, "l1_norm")
    return _scalar_metric(gate_w, metric_name)


# ---------------------------------------------------------------------------
# Canonical sort indices
# ---------------------------------------------------------------------------


def canonical_sort_indices(
    sd: OrderedDict,
    site: PermSite,
    metric_name: MetricName,
) -> torch.Tensor:
    """Return argsort indices that canonicalize the permutable units at *site*."""
    metric_vals = compute_neuron_metric(sd, site, metric_name)
    if metric_vals.ndim == 1:
        return torch.argsort(metric_vals, stable=True)
    # composite: lexicographic sort via structured sort
    # Sort by last metric first (variance), then l2, then l1 for stability
    n = metric_vals.shape[0]
    idx = torch.arange(n)
    for col in range(metric_vals.shape[1] - 1, -1, -1):
        order = torch.argsort(metric_vals[idx, col], stable=True)
        idx = idx[order]
    return idx


# ---------------------------------------------------------------------------
# Tie analysis
# ---------------------------------------------------------------------------


def count_exact_ties(metric_values: torch.Tensor) -> int:
    """Count number of neurons that share an exact metric value with at least one other."""
    if metric_values.ndim > 1:
        # For composite, check if any row is duplicated
        _, counts = torch.unique(metric_values, dim=0, return_counts=True)
    else:
        _, counts = torch.unique(metric_values, return_counts=True)
    # neurons in tie groups (groups of size > 1)
    return int((counts[counts > 1]).sum().item())


def count_approx_ties(metric_values: torch.Tensor, tolerance: float) -> int:
    """Count neurons that have at least one neighbor within *tolerance*."""
    if metric_values.ndim > 1:
        # Use L2 distance between composite metric rows
        flat = metric_values.float()
        dists = torch.cdist(flat.unsqueeze(0), flat.unsqueeze(0)).squeeze(0)
    else:
        vals = metric_values.float().unsqueeze(1)
        dists = torch.cdist(vals, vals).squeeze(0)

    # Zero out diagonal
    dists.fill_diagonal_(float("inf"))
    has_neighbor = (dists <= tolerance).any(dim=1)
    return int(has_neighbor.sum().item())


def uniqueness_probability(metric_values: torch.Tensor, tolerance: float = 0.0) -> float:
    """Fraction of neurons with a unique rank (no ties within *tolerance*)."""
    n = metric_values.shape[0]
    if n == 0:
        return 1.0
    if tolerance == 0.0:
        n_tied = count_exact_ties(metric_values)
    else:
        n_tied = count_approx_ties(metric_values, tolerance)
    return 1.0 - n_tied / n


def tie_group_sizes(metric_values: torch.Tensor) -> List[int]:
    """Return list of tie-group sizes (groups of size >= 2)."""
    if metric_values.ndim > 1:
        _, counts = torch.unique(metric_values, dim=0, return_counts=True)
    else:
        _, counts = torch.unique(metric_values, return_counts=True)
    return [int(c.item()) for c in counts if c.item() > 1]


def tie_group_permutation_count(metric_values: torch.Tensor) -> float:
    """Number of equivalent canonical orderings due to ties = product(k_i!)."""
    groups = tie_group_sizes(metric_values)
    if not groups:
        return 1.0
    return reduce(lambda a, b: a * b, (math.factorial(g) for g in groups), 1.0)


# ---------------------------------------------------------------------------
# Per-layer tie analysis across a full model
# ---------------------------------------------------------------------------


def layer_tie_analysis(
    sd: OrderedDict,
    model_name: str,
    metric_name: MetricName,
    tolerances: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Run tie analysis across all permutable sites for a model.

    Returns a DataFrame with columns:
        site_idx, site_kind, n_neurons, metric_name,
        n_exact_ties, uniqueness_prob_exact,
        n_approx_ties_<tol>, uniqueness_prob_<tol>,
        n_tie_groups, equivalent_orderings
    """
    if tolerances is None:
        tolerances = [1e-7, 1e-5, 1e-3]

    sites = get_permutable_sites(model_name)
    rows = []
    for idx, site in enumerate(sites):
        metric_vals = compute_neuron_metric(sd, site, metric_name)
        n = metric_vals.shape[0]
        n_exact = count_exact_ties(metric_vals)
        groups = tie_group_sizes(metric_vals)
        equiv_orderings = tie_group_permutation_count(metric_vals)

        row = {
            "site_idx": idx,
            "site_kind": site.kind,
            "site_keys": str(list(site.keys.values())[:2]),
            "n_neurons": n,
            "metric_name": metric_name,
            "n_exact_ties": n_exact,
            "uniqueness_prob_exact": 1.0 - n_exact / n if n > 0 else 1.0,
            "n_tie_groups": len(groups),
            "equivalent_orderings": equiv_orderings,
        }

        for tol in tolerances:
            n_approx = count_approx_ties(metric_vals, tol)
            row[f"n_approx_ties_{tol}"] = n_approx
            row[f"uniqueness_prob_{tol}"] = 1.0 - n_approx / n if n > 0 else 1.0

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Theoretical uniqueness bound (birthday-problem style)
# ---------------------------------------------------------------------------


def theoretical_uniqueness_bound(n_neurons: int, metric_precision_bits: int = 24) -> float:
    """Probability that all neurons have distinct metric values (birthday bound).

    For float32, the mantissa has 23 bits + 1 implicit = 24 bits of precision.
    In a given exponent range, there are ~2^24 distinct representable values.

    P(all unique) = product_{i=0}^{n-1} (1 - i / 2^b)
    """
    space = 2 ** metric_precision_bits
    if n_neurons > space:
        return 0.0
    log_prob = sum(math.log1p(-i / space) for i in range(n_neurons))
    return math.exp(log_prob)


def theoretical_recovery_probability(tie_groups: List[int]) -> float:
    """Given tie group sizes, probability of correctly guessing the original order.

    The adversary must guess the permutation within each tie group:
    P(correct) = 1 / product(k_i!)
    """
    if not tie_groups:
        return 1.0
    denom = reduce(lambda a, b: a * b, (math.factorial(g) for g in tie_groups), 1.0)
    return 1.0 / denom
