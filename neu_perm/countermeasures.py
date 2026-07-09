"""
Countermeasures against canonical-ordering attacks on NeuPerm.

Provides lightweight post-permutation perturbations that break canonical
ordering ties without affecting model accuracy.
"""

import copy
from collections import OrderedDict
from typing import Optional

import torch

from neu_perm.canonical import (
    MetricName,
    PermSite,
    compute_neuron_metric,
    get_permutable_sites,
)


def tie_breaking_perturbation(
    sd: OrderedDict,
    model_name: str,
    metric_name: MetricName = "l1_norm",
    epsilon: float = 1e-6,
    inplace: bool = False,
) -> OrderedDict:
    """Add minimal noise to tied neurons to break canonical ordering ambiguity.

    After NeuPerm permutation, neurons with identical metric values allow the
    adversary to narrow down possible orderings. This function adds noise of
    scale *epsilon* only to weight rows of neurons that are part of tie groups.

    Parameters
    ----------
    sd : OrderedDict
        State dict (typically after NeuPerm permutation).
    model_name : str
        Architecture name for site dispatch.
    metric_name : MetricName
        Metric used to detect ties.
    epsilon : float
        Scale of perturbation. Values 1e-7 to 1e-5 are typical — orders of
        magnitude smaller than noise baselines in exp1 (1e-4 to 1e-1).
    inplace : bool
        If True, mutate *sd* directly.

    Returns
    -------
    OrderedDict
        Perturbed state dict (new copy unless *inplace*).
    """
    if not inplace:
        sd = copy.deepcopy(sd)

    sites = get_permutable_sites(model_name)
    for site in sites:
        metric_vals = compute_neuron_metric(sd, site, metric_name)
        tied_mask = _find_tied_neurons(metric_vals)
        if tied_mask.any():
            _perturb_site(sd, site, tied_mask, epsilon)

    return sd


def _find_tied_neurons(metric_vals: torch.Tensor) -> torch.Tensor:
    """Return boolean mask of neurons that share a metric value with another."""
    n = metric_vals.shape[0]
    if metric_vals.ndim > 1:
        # composite: check row-wise
        _, inverse, counts = torch.unique(
            metric_vals, dim=0, return_inverse=True, return_counts=True
        )
    else:
        _, inverse, counts = torch.unique(
            metric_vals, return_inverse=True, return_counts=True
        )
    # A neuron is tied if its unique-group has count > 1
    return counts[inverse] > 1


def _perturb_site(
    sd: OrderedDict,
    site: PermSite,
    tied_mask: torch.Tensor,
    epsilon: float,
) -> None:
    """Add Gaussian noise to the weight rows of tied neurons at *site*."""
    if site.kind == "vgg_pair":
        weight_key = site.keys["layer1_weight"]
    elif site.kind == "conv_bn_conv":
        weight_key = site.keys["conv1_weight"]
    elif site.kind == "llama_attn":
        weight_key = site.keys["k_proj_weight"]
    elif site.kind == "llama_mlp":
        weight_key = site.keys["gate_weight"]
    else:
        return

    w = sd[weight_key]

    # For llama_attn, tied_mask is over KV heads, need to expand to weight rows
    if site.kind == "llama_attn":
        n_kv_heads = tied_mask.shape[0]
        head_dim = w.shape[0] // n_kv_heads
        expanded_mask = tied_mask.unsqueeze(1).expand(-1, head_dim).reshape(-1)
    else:
        expanded_mask = tied_mask

    if expanded_mask.any():
        noise = torch.randn_like(w[expanded_mask]) * epsilon
        w[expanded_mask] = w[expanded_mask] + noise
