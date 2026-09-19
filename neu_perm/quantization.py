"""Post-training quantization (PTQ) via weight-only quantize-dequantize.

Uses PyTorch's FX-based graph tracing to automatically fuse BatchNorm layers
into preceding Conv/Linear layers where possible (Conv→BN patterns).  For
architectures with unfusable BN layers (e.g. DenseNet's BN→ReLU→Conv
pre-activation pattern), BN parameters are left untouched during
quantize-dequantize — only Conv/Linear weights are quantized.

After quantize-dequantize the weights remain as standard float tensors with
reduced effective precision — this lets us reuse ``lsb_extract`` to measure
how many steganographic LSBs survived the quantization.
"""

import copy
from collections import OrderedDict
from typing import Optional, Set  # noqa: F401

import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_fx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.float64, torch.bfloat16)

# Noop qconfig: triggers BN fusion in prepare_fx without inserting any
# observer or fake-quantize nodes.
_NOOP_QCONFIG = tq.QConfig(
    activation=tq.NoopObserver.with_args(dtype=torch.float32),
    weight=tq.NoopObserver.with_args(dtype=torch.float32),
)


def _is_quantizable(t: torch.Tensor) -> bool:
    """Return True if *t* is a float tensor suitable for quantization."""
    return isinstance(t, torch.Tensor) and t.dtype in _FLOAT_DTYPES


def _get_bn_param_keys(model: nn.Module) -> Set[str]:
    """Return state_dict keys that belong to BatchNorm modules."""
    bn_keys = set()
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                            nn.SyncBatchNorm)):
            prefix = name + "." if name else ""
            for param_name in mod.state_dict():
                bn_keys.add(prefix + param_name)
    return bn_keys


# Patterns that identify BatchNorm parameters in state dict key names.
_BN_KEY_PATTERNS = ('running_mean', 'running_var', 'num_batches_tracked')
_BN_MODULE_PATTERNS = ('.bn', '.norm', 'bn.')


def get_bn_keys_from_sd(sd: OrderedDict) -> Set[str]:
    """Identify BatchNorm parameter keys from a state dict by name patterns.

    This works directly on state dicts without needing an nn.Module instance.
    """
    bn_keys = set()
    for key in sd:
        # Always skip running stats and batch counters
        if any(key.endswith(p) for p in _BN_KEY_PATTERNS):
            bn_keys.add(key)
            continue
        # Skip weight/bias that belong to BN modules (identified by name)
        if any(p in key for p in _BN_MODULE_PATTERNS):
            bn_keys.add(key)
    return bn_keys


def fuse_batchnorm(model: nn.Module) -> nn.Module:
    """Fuse BatchNorm into Conv/Linear using FX graph tracing.

    Returns a new ``GraphModule`` with fusable BN layers folded into the
    preceding Conv/Linear weights.  BN layers that cannot be fused (e.g.
    DenseNet's pre-activation BN→ReLU→Conv pattern) remain unchanged.
    """
    model = copy.deepcopy(model).eval()
    qcm = tq.QConfigMapping().set_global(_NOOP_QCONFIG)
    fused = prepare_fx(model, qcm, example_inputs=(torch.randn(1, 3, 224, 224),))
    return fused


# ---------------------------------------------------------------------------
# Core: tensor-level quantize-dequantize
# ---------------------------------------------------------------------------


def quantize_dequantize_tensor(
    tensor: torch.Tensor,
    n_bits: int = 8,
    per_channel: bool = False,
    channel_axis: int = 0,
) -> torch.Tensor:
    """Quantize *tensor* to *n_bits* and immediately dequantize back to float.

    Parameters
    ----------
    tensor : torch.Tensor
        Input float tensor of any shape.
    n_bits : int
        Target bit-width (e.g. 8 or 4).
    per_channel : bool
        If True, compute scale independently along *channel_axis*.
        Falls back to per-tensor for 1-D or scalar tensors.
    channel_axis : int
        Axis along which to compute per-channel scales (default 0,
        i.e. the output-channel dimension for Conv2d / Linear weights).

    Returns
    -------
    torch.Tensor
        Dequantized tensor with same shape and dtype as input.
    """
    orig_dtype = tensor.dtype
    w = tensor.float()  # work in float32

    qmin = -(1 << (n_bits - 1))
    qmax = (1 << (n_bits - 1)) - 1

    if per_channel and w.ndim >= 2:
        # Compute per-channel scale along *channel_axis*
        n_channels = w.shape[channel_axis]
        perm = [channel_axis] + [i for i in range(w.ndim) if i != channel_axis]
        w_perm = w.permute(perm).reshape(n_channels, -1)
        amax = w_perm.abs().amax(dim=1)  # (n_channels,)
        scale = amax / qmax
        scale = scale.clamp(min=1e-8)
        shape = [1] * w.ndim
        shape[channel_axis] = n_channels
        scale = scale.reshape(shape)
    else:
        # Per-tensor scale
        amax = w.abs().amax()
        scale = amax / qmax
        scale = scale.clamp(min=1e-8)

    q = (w / scale).round().clamp(qmin, qmax)
    w_deq = q * scale

    return w_deq.to(orig_dtype)


# ---------------------------------------------------------------------------
# State-dict level
# ---------------------------------------------------------------------------


def quantize_dequantize_sd(
    sd: OrderedDict,
    n_bits: int = 8,
    per_channel: bool = False,
    inplace: bool = False,
    skip_keys: Set[str] = frozenset(),
) -> OrderedDict:
    """Apply quantize-dequantize to every float tensor in *sd*.

    Non-float tensors (e.g. ``num_batches_tracked``) and keys listed in
    *skip_keys* pass through unchanged.

    Parameters
    ----------
    sd : OrderedDict
        Model state dict.
    n_bits : int
        Target bit-width.
    per_channel : bool
        Use per-channel quantization for tensors with ndim >= 2.
    inplace : bool
        If True, modify *sd* directly.
    skip_keys : set of str
        State dict keys to leave unchanged (e.g. BatchNorm parameters).

    Returns
    -------
    OrderedDict
        Quantize-dequantized state dict.
    """
    if not inplace:
        sd = copy.deepcopy(sd)

    for key in list(sd.keys()):
        if key in skip_keys:
            continue
        v = sd[key]
        if not _is_quantizable(v):
            continue
        sd[key] = quantize_dequantize_tensor(
            v, n_bits=n_bits, per_channel=per_channel,
        )

    return sd


# ---------------------------------------------------------------------------
# Model-level convenience
# ---------------------------------------------------------------------------


def quantize_model(
    model: nn.Module,
    n_bits: int = 8,
    per_channel: bool = False,
) -> nn.Module:
    """Return a model with BN fused (where possible) and weights fake-quantized.

    Steps:
    1. Fuse BatchNorm into Conv/Linear via FX tracing (handles Conv→BN patterns).
    2. Identify any remaining (unfused) BatchNorm parameters.
    3. Apply quantize-dequantize to all float weight tensors except BN params.

    The input *model* is never mutated.
    """
    fused = fuse_batchnorm(model)
    bn_keys = _get_bn_param_keys(fused)
    sd = fused.state_dict()
    sd_q = quantize_dequantize_sd(sd, n_bits=n_bits, per_channel=per_channel,
                                  inplace=True, skip_keys=bn_keys)
    fused.load_state_dict(sd_q)
    return fused
