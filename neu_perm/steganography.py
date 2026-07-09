"""
Simulated steganographic payload embedding and extraction using canonical ordering.

This module does NOT implement a real steganographic scheme. It simulates the
coordinate-dependent embedding that the canonical-ordering adaptive attack exploits:

1. The adversary establishes a canonical order of neurons per layer (e.g., sorted by L1-norm).
2. Payload bits are assigned to neurons in canonical order.
3. After NeuPerm permutes the network, the adversary re-computes the canonical order.
4. Any neuron whose rank changed causes a bit error.

The bit error rate (BER) measures how effectively NeuPerm disrupts recovery.

Two BER computation approaches are provided:

- **Tracked BER**: Monkey-patches `get_perm_idxs` to record the permutation indices
  applied by NeuPerm, then computes exact BER by composing canonical ordering with
  the known permutation. Varies per repeat (random permutation).

- **Analytical BER**: Computes expected BER from the tie-group structure of the
  canonical metric. Deterministic for a given model and metric. For a tie group of
  size k, the expected BER contribution is (k-1)/N (one fixed point in expectation).
"""

import copy
import hashlib
import struct
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import neu_perm.perm as _perm_module
from neu_perm.canonical import (
    MetricName,
    PermSite,
    canonical_sort_indices,
    compute_neuron_metric,
    get_permutable_sites,
    tie_group_sizes,
)


# ---------------------------------------------------------------------------
# Payload generation
# ---------------------------------------------------------------------------


def generate_payload(n_bits: int, seed: int = 42) -> np.ndarray:
    """Generate a random binary payload of *n_bits*."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=n_bits, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Site capacity
# ---------------------------------------------------------------------------


def _get_site_capacity(sd: OrderedDict, site: PermSite) -> int:
    """Number of permutable units (neurons/channels/heads) at a site."""
    if site.kind == "vgg_pair":
        return sd[site.keys["layer1_weight"]].shape[0]
    if site.kind == "conv_bn_conv":
        return sd[site.keys["conv1_weight"]].shape[0]
    if site.kind == "llama_attn":
        k_weight = sd[site.keys["k_proj_weight"]]
        q_weight = sd[site.keys["q_proj_weight"]]
        kv_dim = k_weight.shape[0]
        embed_dim = q_weight.shape[0]
        num_heads = site.extra.get("num_heads", 32)
        head_dim = embed_dim // num_heads
        return kv_dim // head_dim  # n_kv_heads
    if site.kind == "llama_mlp":
        return sd[site.keys["gate_weight"]].shape[0]
    raise ValueError(f"Unknown site kind: {site.kind}")


def total_capacity(sd: OrderedDict, model_name: str) -> int:
    """Total number of permutable units across all sites."""
    sites = get_permutable_sites(model_name)
    return sum(_get_site_capacity(sd, s) for s in sites)


# ---------------------------------------------------------------------------
# Raw BER helpers
# ---------------------------------------------------------------------------


def compute_ber(original: np.ndarray, recovered: np.ndarray) -> float:
    """Bit error rate = Hamming distance / length."""
    if len(original) == 0:
        return 0.0
    return float(np.sum(original != recovered)) / len(original)


# ---------------------------------------------------------------------------
# Steganographic payload embedding (actual weight modification)
#
# Implements three embedding schemes:
#   1. LSB: Replace the least significant bit(s) of float32 weight mantissa
#   2. Sign-LSB: Embed in the sign of small-magnitude weights
#   3. Spread-spectrum: Add a small signal weighted by the payload bit
#
# All schemes embed coordinate-dependently: the payload bit assignment
# depends on which neuron is at which position. This is exactly the type
# of steganography that NeuPerm + canonical ordering attacks target.
# ---------------------------------------------------------------------------


# Map each supported float dtype to a same-width signed int dtype.
# The view must be bit-preserving and MUST NOT upcast: LSB steganography writes
# the lowest mantissa bit, so upcasting fp16->fp32 and casting back would round
# that bit away (fp16 has a 10-bit mantissa vs fp32's 23). float16 weights
# (e.g. the LLMs) therefore need an int16 view, not the old fp32-only path.
_FLOAT_INT_WIDTH = {
    torch.float32: torch.int32,
    torch.float16: torch.int16,
}


def _float_to_int_view(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret a float tensor's bits as a same-width signed int (no upcast).

    fp32->int32, fp16->int16. Raises on any other dtype so a silent precision
    mismatch can't corrupt the payload.
    """
    int_dtype = _FLOAT_INT_WIDTH.get(t.dtype)
    if int_dtype is None:
        raise TypeError(
            f"LSB steganography supports float32/float16 parameters, got {t.dtype}"
        )
    return t.view(int_dtype)


def _int_to_float_view(t: torch.Tensor, float_dtype: torch.dtype) -> torch.Tensor:
    """Inverse of :func:`_float_to_int_view`: reinterpret int bits as ``float_dtype``."""
    return t.view(float_dtype)


def _embeddable_keys(sd: OrderedDict) -> List[str]:
    """Return all state_dict keys that hold float weight/bias parameters.

    Skips non-float tensors (e.g., integer buffers like ``num_batches_tracked``,
    running mean/var in BN layers which are not trained).

    Tied weights that share the same underlying storage (e.g. an LLM's
    ``lm_head.weight`` aliasing ``model.embed_tokens.weight`` under
    ``tie_word_embeddings``) are counted **once**: they are a single physical
    region, so embedding into both keys would have the second clobber the
    first's payload bits. We keep the first key encountered and skip later
    aliases, which keeps embed and extract in agreement and corrects the
    capacity count.
    """
    keys = []
    seen_storage = set()
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            continue
        storage_id = (v.untyped_storage().data_ptr(), v.storage_offset())
        if storage_id in seen_storage:
            continue
        seen_storage.add(storage_id)
        keys.append(k)
    return keys


def total_embeddable_params(sd: OrderedDict) -> int:
    """Total number of embeddable scalar parameters across all float tensors."""
    return sum(sd[k].numel() for k in _embeddable_keys(sd))


def lsb_embed(
    sd: OrderedDict,
    payload: np.ndarray,
    n_lsb_bits: int = 1,
    inplace: bool = False,
) -> OrderedDict:
    """Embed payload by replacing the LSBs of float32 weight mantissas.

    Distributes payload bits sequentially across **all** float parameters
    in the state dict (weights, biases, BN parameters, etc.), in state_dict
    iteration order.  For each parameter tensor every scalar element
    contributes *n_lsb_bits* bits of capacity.

    Parameters
    ----------
    sd : OrderedDict
        Model state dict.
    payload : np.ndarray
        Binary payload (0/1 array).
    n_lsb_bits : int
        Number of least significant mantissa bits to replace (1–8).
    inplace : bool
        If True, modify *sd* directly.

    Returns
    -------
    OrderedDict
        State dict with payload embedded.
    """
    if not inplace:
        sd = copy.deepcopy(sd)

    keys = _embeddable_keys(sd)
    bit_idx = 0

    for key in keys:
        if bit_idx >= len(payload):
            break

        w = sd[key]
        orig_dtype = w.dtype
        n_elements = w.numel()
        bits_for_param = min(n_elements * n_lsb_bits, len(payload) - bit_idx)
        if bits_for_param <= 0:
            continue

        # Bit-preserving int view in the parameter's native width (no upcast).
        flat = w.reshape(-1).contiguous()
        int_view = _float_to_int_view(flat)

        for lsb_pos in range(n_lsb_bits):
            n_this_bit = min(n_elements, bits_for_param - lsb_pos * n_elements)
            if n_this_bit <= 0:
                break
            # Match the int view's dtype so bitwise ops don't upcast (int16 vs int32).
            bits = torch.from_numpy(
                payload[bit_idx : bit_idx + n_this_bit].astype(np.int64)
            ).to(device=int_view.device, dtype=int_view.dtype)
            mask = ~(torch.ones_like(int_view[:n_this_bit]) << lsb_pos)
            int_view[:n_this_bit] = (int_view[:n_this_bit] & mask) | (bits << lsb_pos)
            bit_idx += n_this_bit

        sd[key] = _int_to_float_view(int_view, orig_dtype).reshape(w.shape)

    return sd


def lsb_extract(
    sd: OrderedDict,
    n_bits: int,
    n_lsb_bits: int = 1,
) -> np.ndarray:
    """Extract payload by reading LSBs from all float parameters.

    Mirror of ``lsb_embed``: reads in the same key order.
    """
    keys = _embeddable_keys(sd)
    recovered = []
    bit_idx = 0

    for key in keys:
        if bit_idx >= n_bits:
            break

        w = sd[key]
        n_elements = w.numel()
        flat = w.reshape(-1).contiguous()
        int_view = _float_to_int_view(flat)

        for lsb_pos in range(n_lsb_bits):
            n_this_bit = min(n_elements, n_bits - bit_idx)
            if n_this_bit <= 0:
                break
            bits = ((int_view[:n_this_bit] >> lsb_pos) & 1).cpu().numpy().astype(np.uint8)
            recovered.append(bits)
            bit_idx += n_this_bit

    return np.concatenate(recovered)[:n_bits]


def spread_spectrum_embed(
    sd: OrderedDict,
    model_name: str,
    payload: np.ndarray,
    amplitude: float = 1e-4,
    seed: int = 123,
    inplace: bool = False,
) -> OrderedDict:
    """Embed payload using spread-spectrum encoding (similar to MaleficNet).

    For each payload bit b_i, adds (+amplitude if b_i=1, -amplitude if b_i=0)
    times a pseudo-random spreading code to the weight values. The spreading
    code is deterministic given the seed, so the receiver can correlate to
    extract.

    Parameters
    ----------
    sd : OrderedDict
        Model state dict.
    model_name : str
        Architecture name.
    payload : np.ndarray
        Binary payload (0/1 array).
    amplitude : float
        Embedding strength.
    seed : int
        Seed for the spreading code PRNG.
    inplace : bool
        If True, modify *sd* directly.
    """
    if not inplace:
        sd = copy.deepcopy(sd)

    sites = get_permutable_sites(model_name)
    rng = np.random.RandomState(seed)
    bit_idx = 0

    for site in sites:
        weight_key = _primary_weight_key(site)
        w = sd[weight_key]
        n_elements = w.numel()
        bits_for_layer = min(n_elements, len(payload) - bit_idx)
        if bits_for_layer <= 0:
            break

        flat = w.reshape(-1).float()
        # Each payload bit is spread across one weight element
        for i in range(bits_for_layer):
            spreading_sign = 1.0 if payload[bit_idx] == 1 else -1.0
            # Pseudo-random spreading code element
            code = rng.randn()
            flat[i] = flat[i] + amplitude * spreading_sign * abs(code)
            bit_idx += 1

        sd[weight_key] = flat.reshape(w.shape).to(w.dtype)

    return sd


def spread_spectrum_extract(
    sd: OrderedDict,
    model_name: str,
    sd_original: OrderedDict,
    n_bits: int,
) -> np.ndarray:
    """Extract spread-spectrum payload by comparing to original weights.

    The receiver computes the difference between the stego and original
    weights. Positive difference → bit 1, negative → bit 0.
    """
    sites = get_permutable_sites(model_name)
    recovered = []
    bit_idx = 0

    for site in sites:
        weight_key = _primary_weight_key(site)
        w_stego = sd[weight_key].reshape(-1).float()
        w_orig = sd_original[weight_key].reshape(-1).float()

        diff = w_stego - w_orig
        bits_for_layer = min(len(diff), n_bits - bit_idx)
        if bits_for_layer <= 0:
            break

        bits = (diff[:bits_for_layer] > 0).cpu().numpy().astype(np.uint8)
        recovered.append(bits)
        bit_idx += bits_for_layer

    return np.concatenate(recovered)[:n_bits]


def _primary_weight_key(site: PermSite) -> str:
    """Return the primary weight key for embedding at a site."""
    if site.kind == "vgg_pair":
        return site.keys["layer1_weight"]
    if site.kind == "conv_bn_conv":
        return site.keys["conv1_weight"]
    if site.kind == "llama_attn":
        return site.keys["k_proj_weight"]
    if site.kind == "llama_mlp":
        return site.keys["gate_weight"]
    raise ValueError(f"Unknown site kind: {site.kind}")


# ---------------------------------------------------------------------------
# Tracked permutation: record indices during NeuPerm
# ---------------------------------------------------------------------------


def permute_model_tracked(
    model_name: str,
    sd: OrderedDict,
    inplace: bool = True,
) -> Tuple[OrderedDict, List[torch.Tensor]]:
    """Like ``permute_model`` but also records every permutation index vector.

    Returns ``(sd_permuted, recorded_perms)`` where *recorded_perms* is a list
    of 1-D tensors — one per ``get_perm_idxs`` call inside the permutation
    function.
    """
    recorded: List[torch.Tensor] = []
    original_fn = _perm_module.get_perm_idxs

    def _recording_get_perm_idxs(n_c):
        idxs = original_fn(n_c)
        recorded.append(idxs.clone())
        return idxs

    _perm_module.get_perm_idxs = _recording_get_perm_idxs
    try:
        sd_perm = _perm_module.permute_model(model_name, sd, inplace=inplace)
    finally:
        _perm_module.get_perm_idxs = original_fn

    return sd_perm, recorded


# ---------------------------------------------------------------------------
# Map from perm-index position to PermSite
#
# Each architecture has a known number of get_perm_idxs calls per site.
# VGG pair: 1 call
# conv_bn_conv: 1 call
# llama_attn: 1 call (for kv_head_perm_idxs; the q/o reordering is derived)
#             BUT in the non-GQA branch there's also 1 call
# llama_mlp: 1 call
#
# For ResNet, each block calls permute_conv_bn_conv 2 times (conv1→bn1→conv2,
# conv2→bn2→conv3) plus optionally a 3rd (conv3→bn3→downsample). We track
# 2 sites per block (the first two). The 3rd call's perm indices are NOT
# associated with a canonical site because the downsample layer is constrained
# by skip connections and we don't analyse its canonical ordering.
# ---------------------------------------------------------------------------


def _site_to_perm_idx_count(model_name: str) -> List[int]:
    """Number of get_perm_idxs calls each PermSite consumes.

    For most sites this is 1. For ResNet blocks the 3rd call (downsample)
    is consumed by the second site as an extra (accounted for separately).
    """
    sites = get_permutable_sites(model_name)
    counts = [1] * len(sites)
    return counts


def _map_perm_indices_to_sites(
    model_name: str,
    recorded_perms: List[torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """Map each PermSite index to its permutation index tensor.

    Returns ``{site_idx: perm_tensor}``.

    For ResNet, blocks produce 2 or 3 perm calls. Sites are defined as pairs
    (conv1→bn1→conv2, conv2→bn2→conv3). The optional 3rd call (downsample)
    is skipped.
    """
    sites = get_permutable_sites(model_name)

    # All architectures produce exactly one get_perm_idxs call per PermSite,
    # so the mapping is 1:1 for VGG, ResNet, DenseNet, and Llama.
    assert len(recorded_perms) == len(sites), (
        f"Perm call count ({len(recorded_perms)}) != site count ({len(sites)}) "
        f"for {model_name}"
    )
    return {i: recorded_perms[i] for i in range(len(sites))}


# ---------------------------------------------------------------------------
# Tracked BER computation
# ---------------------------------------------------------------------------


def compute_tracked_site_ber(
    sd_before: OrderedDict,
    site: PermSite,
    site_perm: torch.Tensor,
    metric_name: MetricName,
) -> float:
    """Compute exact BER for a site given the known NeuPerm permutation.

    The adversary's attack:
      1. Sort neurons by metric → rank_to_pos_before = argsort(metrics)
      2. Embed bit r at neuron at position rank_to_pos_before[r]
      3. NeuPerm applies permutation π: neuron at position p moves to π[p]
      4. Adversary re-sorts → rank_to_pos_after = argsort(metrics_after)
      5. Reads bit r from neuron at position rank_to_pos_after[r]

    Bit error at rank r when:
      The neuron at rank_to_pos_after[r] (in the permuted sd) was originally
      at a position with a DIFFERENT original rank than r.

    With unique metrics: the same metric value always gets the same rank,
    so BER = 0 regardless of the permutation.

    With tied metrics: within each tie group, the stable sort uses position
    as tiebreaker, and the permutation shuffles positions → BER > 0.
    """
    metrics_before = compute_neuron_metric(sd_before, site, metric_name)
    n = metrics_before.shape[0]
    if n == 0:
        return 0.0

    # Original canonical rank of each neuron (by position before permutation)
    order_before = canonical_sort_indices(sd_before, site, metric_name)
    rank_before = torch.empty(n, dtype=torch.long)
    rank_before[order_before] = torch.arange(n, dtype=torch.long)

    # After permutation π: neuron originally at position p is now at position π[p].
    # The metrics don't change (permutation-invariant), so metrics_after[π[p]] = metrics_before[p].
    # For the stable sort on sd_after, the tie-breaking is by the NEW position.
    # Neuron originally at position p has new position π[p] and metric metrics_before[p].
    # The adversary sorts by (metric_value, new_position) = (metrics_before[p], π[p]).

    # Build (metric_value, new_position) pairs for sorting
    perm = site_perm  # perm[i] = the original position that ends up at new position i
    # Actually, in NeuPerm, the permutation is applied as:
    #   sd[key] = sd[key][perm_idxs, ...]
    # This means new_position[i] gets the value from original_position[perm_idxs[i]].
    # So neuron originally at position perm_idxs[i] ends up at new position i.
    # Equivalently: new_pos_of_original_p = perm_inv[p]

    # Compute inverse permutation
    perm_inv = torch.empty(n, dtype=torch.long)
    perm_inv[perm] = torch.arange(n, dtype=torch.long)

    # Now: neuron originally at position p is at new position perm_inv[p]
    # The adversary computes canonical sort on sd_after:
    # For each new position q, metric_after[q] = metrics_before[perm[q]]
    # Stable argsort of metrics_after sorts by (metric_value, position_q)

    metrics_after = metrics_before[perm]  # metrics at new positions
    if metrics_after.ndim == 1:
        order_after = torch.argsort(metrics_after, stable=True)
    else:
        idx = torch.arange(n)
        for col in range(metrics_after.shape[1] - 1, -1, -1):
            sub_order = torch.argsort(metrics_after[idx, col], stable=True)
            idx = idx[sub_order]
        order_after = idx

    # rank_after[q] = the rank assigned to new position q by the adversary
    rank_after = torch.empty(n, dtype=torch.long)
    rank_after[order_after] = torch.arange(n, dtype=torch.long)

    # The adversary reads bit r from new position order_after[r].
    # That position holds neuron originally from position perm[order_after[r]].
    # That neuron's original rank was rank_before[perm[order_after[r]]].
    # Bit error if rank_before[perm[order_after[r]]] != r.

    errors = 0
    for r in range(n):
        new_pos = order_after[r].item()
        orig_pos = perm[new_pos].item()
        orig_rank = rank_before[orig_pos].item()
        if orig_rank != r:
            errors += 1

    return errors / n


# ---------------------------------------------------------------------------
# Analytical (expected) BER from tie-group structure
# ---------------------------------------------------------------------------


def compute_analytical_site_ber(
    sd: OrderedDict,
    site: PermSite,
    metric_name: MetricName,
) -> float:
    """Expected BER for a site based on tie-group structure.

    For a random permutation within a tie group of size k, the expected number
    of fixed points (neurons keeping their rank) is 1. So expected errors per
    group = k - 1. Neurons with unique metrics always keep their rank (0 errors).

    Expected BER = sum_over_tie_groups(k_i - 1) / N
    """
    metrics = compute_neuron_metric(sd, site, metric_name)
    n = metrics.shape[0]
    if n == 0:
        return 0.0
    groups = tie_group_sizes(metrics)
    n_errors = sum(k - 1 for k in groups)
    return n_errors / n


def compute_analytical_model_ber(
    sd: OrderedDict,
    model_name: str,
    metric_name: MetricName,
) -> Dict[str, float]:
    """Analytical expected BER per site and aggregate for a model."""
    sites = get_permutable_sites(model_name)
    per_site = {}
    total_units = 0
    weighted_sum = 0.0

    for idx, site in enumerate(sites):
        ber = compute_analytical_site_ber(sd, site, metric_name)
        n_units = _get_site_capacity(sd, site)
        key_desc = list(site.keys.values())[0].rsplit(".", 1)[0]
        per_site[f"site_{idx}_{key_desc}"] = ber
        total_units += n_units
        weighted_sum += ber * n_units

    per_site["aggregate_weighted"] = weighted_sum / total_units if total_units > 0 else 0.0
    per_site["aggregate_mean"] = (
        sum(v for k, v in per_site.items() if k.startswith("site_")) / len(sites)
        if sites
        else 0.0
    )
    return per_site


# ---------------------------------------------------------------------------
# Full model tracked BER (used by experiment)
# ---------------------------------------------------------------------------


def compute_tracked_model_ber(
    sd_before: OrderedDict,
    model_name: str,
    metric_name: MetricName,
    site_perms: Dict[int, torch.Tensor],
) -> Dict[str, float]:
    """Exact BER per site using tracked permutation indices."""
    sites = get_permutable_sites(model_name)
    per_site = {}
    total_units = 0
    weighted_sum = 0.0

    for idx, site in enumerate(sites):
        if idx not in site_perms:
            continue
        ber = compute_tracked_site_ber(sd_before, site, site_perms[idx], metric_name)
        n_units = _get_site_capacity(sd_before, site)
        key_desc = list(site.keys.values())[0].rsplit(".", 1)[0]
        per_site[f"site_{idx}_{key_desc}"] = ber
        total_units += n_units
        weighted_sum += ber * n_units

    per_site["aggregate_weighted"] = weighted_sum / total_units if total_units > 0 else 0.0
    per_site["aggregate_mean"] = (
        sum(v for k, v in per_site.items() if k.startswith("site_")) / len(sites)
        if sites
        else 0.0
    )
    return per_site


def compute_detailed_tracked_ber(
    sd_before: OrderedDict,
    model_name: str,
    metric_name: MetricName,
    site_perms: Dict[int, torch.Tensor],
) -> List[Dict]:
    """Per-site tracked BER details as list of dicts (for DataFrame)."""
    sites = get_permutable_sites(model_name)
    rows = []
    for idx, site in enumerate(sites):
        if idx not in site_perms:
            continue
        ber = compute_tracked_site_ber(sd_before, site, site_perms[idx], metric_name)
        n_units = _get_site_capacity(sd_before, site)
        key_desc = list(site.keys.values())[0].rsplit(".", 1)[0]
        rows.append({
            "site_idx": idx,
            "site_kind": site.kind,
            "site_key": key_desc,
            "n_units": n_units,
            "metric_name": metric_name,
            "ber": ber,
        })
    return rows


# ---------------------------------------------------------------------------
# EvilModel-faithful byte substitution (Wang et al., EvilModel 2.0,
# Computers & Security 120:102807, 2022)
#
# Embeds a real byte payload by overwriting the *low x_bytes* of each float32
# parameter's value while preserving the high (sign + exponent + high-mantissa)
# bytes — exactly EvilModel's "keep the leading byte(s), substitute the trailing
# byte(s)" scheme:
#
#   x_bytes = 2  ->  "half substitution"  (keep first 2 bytes, write last 2; max rate 50%)
#   x_bytes = 3  ->  "MSB reservation"    (keep exponent byte, write last 3; max rate 75%)
#
# Faithful integrity convention: the payload **length (uint32) + SHA-256 digest**
# are stored in the carrier layer's *bias* (EvilModel stores length+hash in the
# bias), enabling an integrity-checked extraction. The payload bytes themselves
# go into the carrier-site *weights*.
#
# Carrier sites are NeuPerm's permutable sites (get_permutable_sites), so the
# embedded coordinates are exactly the ones NeuPerm reorders: extraction is
# byte-exact before NeuPerm and corrupted after.
#
# float32 only — half substitution / MSB reservation need >=32-bit floats; the
# fp16 LLMs have no spare high bytes (this is EvilModel's own CNN target domain).
# ---------------------------------------------------------------------------

_EVILMODEL_LEN_BYTES = 4  # uint32 payload length
_EVILMODEL_HASH_BYTES = 32  # SHA-256 digest
_EVILMODEL_HEADER_BYTES = _EVILMODEL_LEN_BYTES + _EVILMODEL_HASH_BYTES  # 36


def _evilmodel_carrier_weight_keys(model_name: str) -> List[str]:
    """Carrier weight keys = the primary weight of each permutable site."""
    return [_primary_weight_key(s) for s in get_permutable_sites(model_name)]


def _site_bias_key(site: PermSite) -> str:
    """The bias that NeuPerm permutes alongside a site's primary weight.

    vgg_pair -> the conv/fc bias (``layer1_bias``); conv_bn_conv -> the BN bias
    that follows conv1 (``bn_bias``) since those convs carry no bias. This is the
    EvilModel integrity-header carrier and is reordered with the site, so the
    header is intact pre-NeuPerm and scrambled post-NeuPerm."""
    for k in ("layer1_bias", "bn_bias"):
        if k in site.keys:
            return site.keys[k]
    raise KeyError(f"site kind {site.kind!r} has no bias key for the EvilModel header")


def _evilmodel_header_bias_key(model_name: str) -> str:
    """Bias key holding the length+SHA header (first carrier site)."""
    return _site_bias_key(get_permutable_sites(model_name)[0])


def _require_float32(sd: OrderedDict, key: str) -> None:
    if sd[key].dtype != torch.float32:
        raise TypeError(
            f"EvilModel byte substitution requires float32 parameters; "
            f"{key!r} is {sd[key].dtype}"
        )


def _write_bytes_low(sd: OrderedDict, key: str, data: bytes, x_bytes: int) -> None:
    """Overwrite the low ``x_bytes`` of each float32 element of ``sd[key]`` with
    consecutive bytes of ``data`` (little-endian within each element)."""
    _require_float32(sd, key)
    w = sd[key]
    flat = w.reshape(-1).contiguous()
    int_view = _float_to_int_view(flat)  # int32, bit-preserving
    n_params = (len(data) + x_bytes - 1) // x_bytes
    if n_params > int_view.numel():
        raise ValueError(
            f"{key!r} holds {int_view.numel()} params "
            f"({int_view.numel() * x_bytes} bytes) < {len(data)} bytes requested"
        )
    # Pad to a whole number of params, pack x_bytes/param little-endian.
    padded = data + b"\x00" * ((-len(data)) % x_bytes)
    bvals = np.frombuffer(padded, dtype=np.uint8).reshape(-1, x_bytes).astype(np.int64)
    packed = np.zeros(bvals.shape[0], dtype=np.int64)
    for j in range(x_bytes):
        packed |= bvals[:, j] << (8 * j)
    low_mask = (1 << (8 * x_bytes)) - 1
    keep_mask = torch.tensor(~low_mask, dtype=int_view.dtype)
    packed_t = torch.from_numpy(packed).to(device=int_view.device, dtype=int_view.dtype)
    int_view[:n_params] = (int_view[:n_params] & keep_mask) | packed_t
    sd[key] = _int_to_float_view(int_view, torch.float32).reshape(w.shape)


def _read_bytes_low(sd: OrderedDict, key: str, n_bytes: int, x_bytes: int) -> bytes:
    """Inverse of :func:`_write_bytes_low`."""
    _require_float32(sd, key)
    w = sd[key]
    flat = w.reshape(-1).contiguous()
    int_view = _float_to_int_view(flat)
    n_params = (n_bytes + x_bytes - 1) // x_bytes
    if n_params > int_view.numel():
        raise ValueError(f"{key!r} too small to hold {n_bytes} bytes")
    low_mask = (1 << (8 * x_bytes)) - 1
    vals = (int_view[:n_params].cpu().numpy().astype(np.int64)) & low_mask
    out = np.zeros((n_params, x_bytes), dtype=np.uint8)
    for j in range(x_bytes):
        out[:, j] = (vals >> (8 * j)) & 0xFF
    return out.reshape(-1)[:n_bytes].tobytes()


def evilmodel_capacity_bytes(sd: OrderedDict, model_name: str, x_bytes: int = 2) -> int:
    """Total payload capacity (bytes) across carrier-site weights."""
    keys = _evilmodel_carrier_weight_keys(model_name)
    return sum(sd[k].numel() for k in keys) * x_bytes


def generate_payload_bytes(n_bytes: int, seed: int = 42) -> bytes:
    """Benign random byte blob (payload stand-in; not malware)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=n_bytes, dtype=np.uint8).tobytes()


def evilmodel_embed(
    sd: OrderedDict,
    model_name: str,
    payload_bytes: bytes,
    x_bytes: int = 2,
    inplace: bool = False,
) -> OrderedDict:
    """Embed ``payload_bytes`` by EvilModel byte substitution into carrier sites.

    Header (uint32 length + SHA-256) is written into the first carrier's bias;
    payload bytes are written across the carrier-site weights in order.
    """
    if x_bytes not in (2, 3):
        raise ValueError("x_bytes must be 2 (half substitution) or 3 (MSB reservation)")
    if not inplace:
        sd = copy.deepcopy(sd)

    keys = _evilmodel_carrier_weight_keys(model_name)
    if not keys:
        raise ValueError(f"No permutable carrier sites for {model_name!r}")

    cap = evilmodel_capacity_bytes(sd, model_name, x_bytes)
    if len(payload_bytes) > cap:
        raise ValueError(f"payload {len(payload_bytes)} bytes > capacity {cap} bytes")

    # Integrity header in the first carrier's bias (EvilModel: length+hash in bias).
    header = struct.pack("<I", len(payload_bytes)) + hashlib.sha256(payload_bytes).digest()
    bias_key = _evilmodel_header_bias_key(model_name)
    if bias_key not in sd:
        raise KeyError(f"header bias {bias_key!r} missing from state_dict")
    _write_bytes_low(sd, bias_key, header, x_bytes)

    # Payload across carrier weights, in order.
    off = 0
    for key in keys:
        if off >= len(payload_bytes):
            break
        chunk = payload_bytes[off : off + sd[key].numel() * x_bytes]
        _write_bytes_low(sd, key, chunk, x_bytes)
        off += len(chunk)
    return sd


def evilmodel_extract_raw(
    sd: OrderedDict, model_name: str, n_bytes: int, x_bytes: int = 2
) -> bytes:
    """Read ``n_bytes`` payload bytes from carrier weights (ignores the header).

    Used to measure byte/bit error vs a known payload regardless of header
    integrity (the header itself is corrupted once NeuPerm permutes the bias)."""
    keys = _evilmodel_carrier_weight_keys(model_name)
    out = bytearray()
    for key in keys:
        if len(out) >= n_bytes:
            break
        want = min(n_bytes - len(out), sd[key].numel() * x_bytes)
        out += _read_bytes_low(sd, key, want, x_bytes)
    return bytes(out[:n_bytes])


def evilmodel_extract(
    sd: OrderedDict, model_name: str, x_bytes: int = 2
) -> Tuple[bytes, bool]:
    """Integrity-checked extraction. Returns ``(payload_bytes, hash_ok)``.

    Reads length+hash from the first carrier's bias, then that many payload
    bytes from the carrier weights, and verifies the SHA-256. A corrupted
    header (e.g. after NeuPerm scrambles the bias) yields ``("", False)``."""
    bias_key = _evilmodel_header_bias_key(model_name)
    header = _read_bytes_low(sd, bias_key, _EVILMODEL_HEADER_BYTES, x_bytes)
    length = struct.unpack("<I", header[:_EVILMODEL_LEN_BYTES])[0]
    exp_hash = header[_EVILMODEL_LEN_BYTES:_EVILMODEL_HEADER_BYTES]
    cap = evilmodel_capacity_bytes(sd, model_name, x_bytes)
    if length > cap:  # corrupted header -> bogus length; fail without huge alloc
        return b"", False
    payload = evilmodel_extract_raw(sd, model_name, length, x_bytes)
    return payload, hashlib.sha256(payload).digest() == exp_hash


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Unpack bytes to a 0/1 bit array (LSB-first within each byte)."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
