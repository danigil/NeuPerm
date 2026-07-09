"""
Tests for neu_perm.steganography — payload simulation and BER computation.
"""

from collections import OrderedDict

import numpy as np
import pytest
import torch

from neu_perm.canonical import PermSite, compute_neuron_metric
from neu_perm.steganography import (
    _read_bytes_low,
    _write_bytes_low,
    compute_analytical_site_ber,
    compute_ber,
    compute_tracked_site_ber,
    evilmodel_capacity_bytes,
    evilmodel_embed,
    evilmodel_extract,
    evilmodel_extract_raw,
    generate_payload,
    generate_payload_bytes,
    lsb_embed,
    lsb_extract,
)


# ---------------------------------------------------------------------------
# Tests: generate_payload
# ---------------------------------------------------------------------------


class TestGeneratePayload:
    def test_correct_length(self):
        payload = generate_payload(100, seed=0)
        assert len(payload) == 100

    def test_binary_values(self):
        payload = generate_payload(1000, seed=42)
        assert set(np.unique(payload)).issubset({0, 1})

    def test_deterministic(self):
        p1 = generate_payload(100, seed=7)
        p2 = generate_payload(100, seed=7)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds(self):
        p1 = generate_payload(100, seed=1)
        p2 = generate_payload(100, seed=2)
        assert not np.array_equal(p1, p2)


# ---------------------------------------------------------------------------
# Tests: compute_ber
# ---------------------------------------------------------------------------


class TestComputeBER:
    def test_identical(self):
        p = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
        assert compute_ber(p, p) == 0.0

    def test_all_flipped(self):
        original = np.array([0, 0, 0, 0], dtype=np.uint8)
        recovered = np.array([1, 1, 1, 1], dtype=np.uint8)
        assert compute_ber(original, recovered) == 1.0

    def test_half_flipped(self):
        original = np.array([0, 0, 1, 1], dtype=np.uint8)
        recovered = np.array([1, 1, 1, 1], dtype=np.uint8)
        assert compute_ber(original, recovered) == 0.5

    def test_empty(self):
        assert compute_ber(np.array([], dtype=np.uint8), np.array([], dtype=np.uint8)) == 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sd_and_site(weights):
    """Create a VGG-pair-like sd and site from a weight matrix."""
    sd = OrderedDict()
    sd["layer1.weight"] = weights.unsqueeze(-1).unsqueeze(-1)
    sd["layer1.bias"] = torch.arange(weights.shape[0], dtype=torch.float32)
    sd["layer2.weight"] = torch.randn(8, weights.shape[0], 3, 3)
    site = PermSite(
        kind="vgg_pair",
        keys={
            "layer1_weight": "layer1.weight",
            "layer1_bias": "layer1.bias",
            "layer2_weight": "layer2.weight",
        },
    )
    return sd, site


# ---------------------------------------------------------------------------
# Tests: analytical BER
# ---------------------------------------------------------------------------


class TestAnalyticalBER:
    def test_unique_metrics_ber_zero(self):
        """All unique metrics → expected BER = 0."""
        w = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        sd, site = _make_sd_and_site(w)
        ber = compute_analytical_site_ber(sd, site, "l1_norm")
        assert ber == 0.0

    def test_all_tied_metrics_high_ber(self):
        """All tied metrics → expected BER = (k-1)/k."""
        w = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        sd, site = _make_sd_and_site(w)
        ber = compute_analytical_site_ber(sd, site, "l1_norm")
        # One group of 4: expected errors = 3, BER = 3/4 = 0.75
        assert abs(ber - 0.75) < 1e-10

    def test_partial_ties(self):
        """Two neurons tied, two unique → expected BER = 1/4."""
        w = torch.tensor([[1.0, 1.0], [1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
        sd, site = _make_sd_and_site(w)
        ber = compute_analytical_site_ber(sd, site, "l1_norm")
        # L1 norms: [2, 2, 6, 10]. Tie group of size 2: expected errors = 1
        # BER = 1/4 = 0.25
        assert abs(ber - 0.25) < 1e-10


# ---------------------------------------------------------------------------
# Tests: tracked BER with known permutation
# ---------------------------------------------------------------------------


class TestTrackedBER:
    def test_identity_permutation_ber_zero(self):
        """Identity permutation → BER = 0 regardless of ties."""
        w = torch.tensor([[1.0, 1.0], [1.0, 1.0], [3.0, 4.0]])
        sd, site = _make_sd_and_site(w)
        identity_perm = torch.tensor([0, 1, 2])
        ber = compute_tracked_site_ber(sd, site, identity_perm, "l1_norm")
        assert ber == 0.0

    def test_unique_metrics_any_perm_ber_zero(self):
        """Unique metrics → BER = 0 for ANY permutation."""
        w = torch.tensor([[1.0, 0.0], [0.0, 3.0], [5.0, 0.0], [0.0, 8.0]])
        sd, site = _make_sd_and_site(w)
        perm = torch.tensor([2, 0, 3, 1])
        ber = compute_tracked_site_ber(sd, site, perm, "l1_norm")
        assert ber == 0.0

    def test_tied_metrics_swap_ber_nonzero(self):
        """Two tied neurons that get swapped → BER > 0."""
        # L1 norms: [2, 2, 5, 8] — first two are tied
        w = torch.tensor([[1.0, 1.0], [1.0, 1.0], [2.0, 3.0], [4.0, 4.0]])
        sd, site = _make_sd_and_site(w)
        # Permutation that swaps first two neurons (the tied ones)
        perm = torch.tensor([1, 0, 2, 3])
        ber = compute_tracked_site_ber(sd, site, perm, "l1_norm")
        # Neurons 0 and 1 swapped, both tied → both get wrong rank
        # BER = 2/4 = 0.5
        assert abs(ber - 0.5) < 1e-10

    def test_tied_metrics_no_swap_ber_zero(self):
        """Two tied neurons that DON'T swap → BER = 0."""
        w = torch.tensor([[1.0, 1.0], [1.0, 1.0], [2.0, 3.0], [4.0, 4.0]])
        sd, site = _make_sd_and_site(w)
        # Permutation that keeps first two in same relative order
        perm = torch.tensor([0, 1, 3, 2])  # only swap last two (unique metrics)
        ber = compute_tracked_site_ber(sd, site, perm, "l1_norm")
        assert ber == 0.0

    def test_full_reversal_with_unique_ber_zero(self):
        """Fully reversed permutation with unique metrics → BER = 0."""
        w = torch.tensor([[1.0, 0.0], [0.0, 3.0], [5.0, 0.0]])
        sd, site = _make_sd_and_site(w)
        perm = torch.tensor([2, 1, 0])
        ber = compute_tracked_site_ber(sd, site, perm, "l1_norm")
        assert ber == 0.0


# ---------------------------------------------------------------------------
# Tests: LSB embed/extract round-trip across float widths
#
# Regression guard for the fp16 path: the old helpers upcast every parameter to
# float32 before writing the mantissa LSB, then cast back, which rounded the bit
# away for float16 weights (the LLMs). The dtype-aware helpers must round-trip a
# payload exactly in both float32 and float16.
# ---------------------------------------------------------------------------


def _random_float_sd(dtype, seed=0):
    """Small multi-tensor state dict with mixed shapes in a given float dtype."""
    g = torch.Generator().manual_seed(seed)
    sd = OrderedDict()
    sd["a.weight"] = torch.randn(6, 4, 3, 3, generator=g).to(dtype)
    sd["a.bias"] = torch.randn(6, generator=g).to(dtype)
    sd["b.weight"] = torch.randn(10, 6, generator=g).to(dtype)
    return sd


class TestLSBRoundTrip:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_clean_roundtrip_ber_zero(self, dtype):
        """Embed then extract on the same model → BER exactly 0 in fp32 and fp16."""
        sd = _random_float_sd(dtype)
        n_params = sum(v.numel() for v in sd.values())
        payload = generate_payload(n_params, seed=7)

        sd_stego = lsb_embed(sd, payload, n_lsb_bits=1, inplace=False)
        recovered = lsb_extract(sd_stego, n_params, n_lsb_bits=1)

        assert compute_ber(payload, recovered) == 0.0

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_embed_preserves_dtype(self, dtype):
        """Embedding must not change parameter dtypes (no silent upcast)."""
        sd = _random_float_sd(dtype)
        n_params = sum(v.numel() for v in sd.values())
        payload = generate_payload(n_params, seed=1)
        sd_stego = lsb_embed(sd, payload, n_lsb_bits=1, inplace=False)
        assert all(v.dtype == dtype for v in sd_stego.values())

    def test_fp16_embed_actually_survives(self):
        """Targeted regression: at least one fp16 weight's LSB changes and is
        recovered — guards against the upcast-then-cast bug that silently
        discarded the payload (clean BER would have been ~0.5)."""
        sd = _random_float_sd(torch.float16, seed=3)
        n_params = sum(v.numel() for v in sd.values())
        # All-ones payload forces every mantissa LSB to 1.
        payload = np.ones(n_params, dtype=np.uint8)
        sd_stego = lsb_embed(sd, payload, n_lsb_bits=1, inplace=False)
        recovered = lsb_extract(sd_stego, n_params, n_lsb_bits=1)
        assert np.array_equal(recovered, payload)

    def test_unsupported_dtype_raises(self):
        """bfloat16 (or any unlisted dtype) must fail loudly, not corrupt silently."""
        sd = OrderedDict()
        sd["a.weight"] = torch.randn(4, 4).to(torch.bfloat16)
        payload = generate_payload(16, seed=0)
        with pytest.raises(TypeError):
            lsb_embed(sd, payload, n_lsb_bits=1, inplace=False)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_tied_weights_counted_once(self, dtype):
        """Tied weights sharing storage (LLM lm_head <-> embed_tokens) must be
        embedded once. Counting both keys would have the second embed clobber
        the first's bits, giving a nonzero clean round-trip BER."""
        from neu_perm.steganography import _embeddable_keys, total_embeddable_params

        shared = torch.randn(8, 4).to(dtype)
        sd = OrderedDict()
        sd["model.embed_tokens.weight"] = shared
        sd["model.layer.weight"] = torch.randn(5, 4).to(dtype)
        sd["lm_head.weight"] = shared  # tied: same storage as embed_tokens

        keys = _embeddable_keys(sd)
        assert "lm_head.weight" not in keys  # alias skipped
        assert total_embeddable_params(sd) == shared.numel() + 5 * 4

        n = total_embeddable_params(sd)
        payload = generate_payload(n, seed=11)
        sd_stego = lsb_embed(sd, payload, n_lsb_bits=1, inplace=False)
        recovered = lsb_extract(sd_stego, n, n_lsb_bits=1)
        assert compute_ber(payload, recovered) == 0.0


# ---------------------------------------------------------------------------
# Tests: EvilModel byte substitution (half-sub / MSB reservation)
# ---------------------------------------------------------------------------


class TestEvilModelBytes:
    @pytest.mark.parametrize("x_bytes", [2, 3])
    def test_write_read_roundtrip(self, x_bytes):
        sd = OrderedDict({"w": torch.randn(200).float()})
        data = generate_payload_bytes(50, seed=3)
        _write_bytes_low(sd, "w", data, x_bytes)
        out = _read_bytes_low(sd, "w", len(data), x_bytes)
        assert out == data

    def test_half_sub_preserves_high_bytes(self):
        orig = torch.randn(100).float()
        sd = OrderedDict({"w": orig.clone()})
        _write_bytes_low(sd, "w", generate_payload_bytes(100, seed=1), x_bytes=2)
        hi_o = (orig.view(torch.int32) >> 16) & 0xFFFF
        hi_n = (sd["w"].view(torch.int32) >> 16) & 0xFFFF
        torch.testing.assert_close(hi_o, hi_n)  # top 2 bytes untouched

    def test_write_rejects_non_float32(self):
        sd = OrderedDict({"w": torch.randn(10).half()})
        with pytest.raises(TypeError):
            _write_bytes_low(sd, "w", b"\x01\x02", x_bytes=2)


class TestEvilModelVGG:
    """Full embed -> extract -> NeuPerm-destroys on a real (random-init) vgg11."""

    @pytest.fixture(scope="class")
    def vgg11_sd(self):
        import torchvision
        sd = torchvision.models.vgg11(weights=None).state_dict()
        return {k: (v.float() if torch.is_tensor(v) and v.dtype != torch.float32 else v)
                for k, v in sd.items()}

    def test_capacity_positive(self, vgg11_sd):
        assert evilmodel_capacity_bytes(vgg11_sd, "vgg11", x_bytes=2) > 0

    def test_overflow_raises(self, vgg11_sd):
        cap = evilmodel_capacity_bytes(vgg11_sd, "vgg11", x_bytes=2)
        with pytest.raises(ValueError):
            evilmodel_embed(vgg11_sd, "vgg11", b"\x00" * (cap + 1), x_bytes=2)

    def test_roundtrip_byte_exact(self, vgg11_sd):
        payload = generate_payload_bytes(20_000, seed=5)
        stego = evilmodel_embed(vgg11_sd, "vgg11", payload, x_bytes=2, inplace=False)
        recovered, ok = evilmodel_extract(stego, "vgg11", x_bytes=2)
        assert ok and recovered == payload

    def test_neuperm_destroys_payload(self, vgg11_sd):
        import copy
        import neu_perm.perm as perm
        payload = generate_payload_bytes(20_000, seed=6)
        stego = evilmodel_embed(vgg11_sd, "vgg11", payload, x_bytes=2, inplace=False)
        torch.manual_seed(0)
        permd = perm.permute_model("vgg11", copy.deepcopy(stego), inplace=True)
        _, ok = evilmodel_extract(permd, "vgg11", x_bytes=2)
        raw = evilmodel_extract_raw(permd, "vgg11", len(payload), x_bytes=2)
        from neu_perm.steganography import bytes_to_bits
        ber = float(np.mean(bytes_to_bits(raw) != bytes_to_bits(payload)))
        assert not ok and ber > 0.0
