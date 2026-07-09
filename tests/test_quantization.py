"""Tests for neu_perm.quantization — weight-only PTQ via quantize-dequantize."""

import copy
from collections import OrderedDict

import numpy as np
import pytest
import torch

from neu_perm.quantization import (
    quantize_dequantize_sd,
    quantize_dequantize_tensor,
    quantize_model,
)


# ---------------------------------------------------------------------------
# Tensor-level tests
# ---------------------------------------------------------------------------


class TestQuantizeDequantizeTensor:
    def test_output_shape_matches_input(self):
        t = torch.randn(64, 32, 3, 3)
        out = quantize_dequantize_tensor(t, n_bits=8)
        assert out.shape == t.shape

    def test_8bit_values_close_but_not_identical(self):
        torch.manual_seed(0)
        t = torch.randn(128, 64)
        out = quantize_dequantize_tensor(t, n_bits=8)
        # Should be close
        assert torch.allclose(t, out, atol=0.05)
        # But not identical (quantization introduces rounding)
        assert not torch.equal(t, out)

    def test_4bit_larger_error_than_8bit(self):
        torch.manual_seed(0)
        t = torch.randn(128, 64)
        out_8 = quantize_dequantize_tensor(t, n_bits=8)
        out_4 = quantize_dequantize_tensor(t, n_bits=4)
        err_8 = (t - out_8).abs().mean().item()
        err_4 = (t - out_4).abs().mean().item()
        assert err_4 > err_8

    def test_zero_tensor_stays_zero(self):
        t = torch.zeros(32, 16)
        out = quantize_dequantize_tensor(t, n_bits=8)
        assert torch.equal(out, t)

    def test_scalar_tensor(self):
        t = torch.tensor(3.14)
        out = quantize_dequantize_tensor(t, n_bits=8)
        assert out.shape == t.shape
        # Should be close to original
        assert abs(out.item() - t.item()) < 0.1

    def test_per_channel_different_scales(self):
        """Per-channel should give different quantization per output channel."""
        torch.manual_seed(42)
        # Channel 0 has small values, channel 1 has large values
        t = torch.zeros(2, 100)
        t[0] = torch.randn(100) * 0.01
        t[1] = torch.randn(100) * 100.0

        out_pt = quantize_dequantize_tensor(t, n_bits=4, per_channel=False)
        out_pc = quantize_dequantize_tensor(t, n_bits=4, per_channel=True)

        # Per-channel should preserve small channel better
        err_pt_ch0 = (t[0] - out_pt[0]).abs().mean().item()
        err_pc_ch0 = (t[0] - out_pc[0]).abs().mean().item()
        assert err_pc_ch0 < err_pt_ch0

    def test_float16_roundtrip(self):
        t = torch.randn(32, 16).half()
        out = quantize_dequantize_tensor(t, n_bits=8)
        assert out.dtype == torch.float16
        assert out.shape == t.shape

    def test_bfloat16_roundtrip(self):
        t = torch.randn(32, 16).bfloat16()
        out = quantize_dequantize_tensor(t, n_bits=8)
        assert out.dtype == torch.bfloat16
        assert out.shape == t.shape

    def test_1d_tensor_per_channel_fallback(self):
        """Per-channel on 1D tensor should fall back to per-tensor."""
        t = torch.randn(64)
        out_pt = quantize_dequantize_tensor(t, n_bits=8, per_channel=False)
        out_pc = quantize_dequantize_tensor(t, n_bits=8, per_channel=True)
        assert torch.equal(out_pt, out_pc)


# ---------------------------------------------------------------------------
# State-dict level tests
# ---------------------------------------------------------------------------


class TestQuantizeDequantizeSd:
    @pytest.fixture
    def sample_sd(self):
        return OrderedDict([
            ("conv.weight", torch.randn(16, 3, 3, 3)),
            ("conv.bias", torch.randn(16)),
            ("bn.weight", torch.randn(16)),
            ("bn.bias", torch.randn(16)),
            ("bn.running_mean", torch.randn(16)),
            ("bn.running_var", torch.randn(16).abs()),
            ("bn.num_batches_tracked", torch.tensor(100, dtype=torch.long)),
        ])

    def test_all_float_keys_quantized(self, sample_sd):
        sd_q = quantize_dequantize_sd(sample_sd, n_bits=8)
        for key in ["conv.weight", "conv.bias", "bn.weight", "bn.bias",
                     "bn.running_mean", "bn.running_var"]:
            assert not torch.equal(sd_q[key], sample_sd[key]), f"{key} was not quantized"

    def test_non_float_keys_unchanged(self, sample_sd):
        sd_q = quantize_dequantize_sd(sample_sd, n_bits=8)
        assert torch.equal(
            sd_q["bn.num_batches_tracked"],
            sample_sd["bn.num_batches_tracked"],
        )

    def test_immutable_by_default(self, sample_sd):
        orig_weight = sample_sd["conv.weight"].clone()
        quantize_dequantize_sd(sample_sd, n_bits=8, inplace=False)
        assert torch.equal(sample_sd["conv.weight"], orig_weight)

    def test_inplace_modifies_original(self, sample_sd):
        orig_weight = sample_sd["conv.weight"].clone()
        quantize_dequantize_sd(sample_sd, n_bits=8, inplace=True)
        assert not torch.equal(sample_sd["conv.weight"], orig_weight)


# ---------------------------------------------------------------------------
# LSB destruction tests
# ---------------------------------------------------------------------------


class TestLsbDestruction:
    """Verify that PTQ destroys LSB steganographic payloads."""

    def _make_stego_sd(self, n_lsb_bits: int = 1):
        """Create a small state_dict with embedded LSB payload."""
        from neu_perm.steganography import (
            generate_payload,
            lsb_embed,
            total_embeddable_params,
        )

        sd = OrderedDict([
            ("w1", torch.randn(64, 32)),
            ("w2", torch.randn(32, 16)),
            ("b1", torch.randn(64)),
        ])
        n_params = total_embeddable_params(sd)
        n_bits = n_params * n_lsb_bits
        payload = generate_payload(n_bits, seed=42)
        sd_stego = lsb_embed(sd, payload, n_lsb_bits=n_lsb_bits, inplace=False)
        return sd_stego, payload, n_bits

    def test_lsb_1bit_destroyed_after_8bit_ptq(self):
        from neu_perm.steganography import compute_ber, lsb_extract

        sd_stego, payload, n_bits = self._make_stego_sd(n_lsb_bits=1)
        sd_q = quantize_dequantize_sd(sd_stego, n_bits=8)
        recovered = lsb_extract(sd_q, n_bits, n_lsb_bits=1)
        ber = compute_ber(payload, recovered)
        # Should be near 0.5 (random)
        assert 0.3 < ber < 0.7, f"BER={ber}, expected near 0.5"

    def test_lsb_1bit_destroyed_after_4bit_ptq(self):
        from neu_perm.steganography import compute_ber, lsb_extract

        sd_stego, payload, n_bits = self._make_stego_sd(n_lsb_bits=1)
        sd_q = quantize_dequantize_sd(sd_stego, n_bits=4)
        recovered = lsb_extract(sd_q, n_bits, n_lsb_bits=1)
        ber = compute_ber(payload, recovered)
        assert 0.3 < ber < 0.7, f"BER={ber}, expected near 0.5"

    def test_lsb_8bit_destroyed_after_8bit_ptq(self):
        from neu_perm.steganography import compute_ber, lsb_extract

        sd_stego, payload, n_bits = self._make_stego_sd(n_lsb_bits=8)
        sd_q = quantize_dequantize_sd(sd_stego, n_bits=8)
        recovered = lsb_extract(sd_q, n_bits, n_lsb_bits=8)
        ber = compute_ber(payload, recovered)
        assert 0.3 < ber < 0.7, f"BER={ber}, expected near 0.5"


# ---------------------------------------------------------------------------
# Model-level tests
# ---------------------------------------------------------------------------


class TestQuantizeModel:
    def test_model_produces_valid_output(self):
        """Quantized model should produce non-NaN outputs."""
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 10),
        )
        model_q = quantize_model(model, n_bits=8)
        x = torch.randn(1, 3, 32, 32)
        out = model_q(x)
        assert not torch.isnan(out).any()
        assert out.shape == (1, 10)

    def test_4bit_model_produces_valid_output(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10),
        )
        model_q = quantize_model(model, n_bits=4)
        x = torch.randn(1, 64)
        out = model_q(x)
        assert not torch.isnan(out).any()

    def test_original_model_unchanged(self):
        model = torch.nn.Linear(32, 16)
        orig_weight = model.weight.data.clone()
        quantize_model(model, n_bits=8)
        assert torch.equal(model.weight.data, orig_weight)
