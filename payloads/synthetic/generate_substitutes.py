#!/usr/bin/env python3
"""Generate safe synthetic substitutes for the nine MaleficNet malware payloads.

Why this exists
---------------
The SNR evaluation was computed on nine XOR-obfuscated *live* malware
samples. Those live samples are not distributed with this artifact; they are
replaced by the deterministic, inert substitutes this script produces. See
`payloads/MANIFEST.sha256` for the SHA-256 provenance of the originals, so an
authorised researcher can obtain the exact bytes by hash.

Design goals (each substitute)
------------------------------
1. Deterministic + seeded  -> the artifact is reproducible byte-for-byte.
2. Length matched exactly   -> the steganographic pipeline allocates the same
   message length / spreading gain, so SNR/BER behave comparably.
3. Byte-entropy matched      -> the target Shannon entropy is the measured
   entropy of the original payload (constant-XOR preserves entropy, so the
   stored .xor and the decoded plaintext share one entropy value).
4. Obviously inert           -> each decoded payload opens with an ASCII banner
   that names it a synthetic substitute; no MZ/PE or ELF header appears in
   either the stored or the decoded form.

Storage format
--------------
The original consumers read `<name>.xor` and decode with `byte ^ 0xFF`. To stay
a drop-in, each substitute is written the same way: we build the synthetic
plaintext (banner + entropy-shaped body) and store `plaintext ^ 0xFF` as
`<name>.xor`. Decoding it with the existing `b ^ 0xFF` step yields the readable
synthetic plaintext.

Usage
-----
    python generate_substitutes.py            # writes ./<name>.xor
    python generate_substitutes.py --verify   # regenerate in memory, print
                                              # realized length + entropy, no write
"""
from __future__ import annotations

import argparse
import collections
import math
from pathlib import Path

import numpy as np

# --- Target profile of the nine originals -----------------------------------
# (name, exact_byte_length, measured_shannon_entropy_bits_per_byte)
# Lengths and entropies were measured in memory from the original payload
# archive members; the malware bytes were never written to disk. The
# decoded-plaintext entropy equals the stored-.xor entropy because
# XOR with a constant (0xFF) is a byte-wise bijection and leaves the histogram,
# hence the Shannon entropy, unchanged.
TARGETS = [
    ("asprox",       94208, 6.2614),
    ("bladabindi",  107520, 6.3721),
    ("cerber",      619008, 5.0924),
    ("destover",     91888, 5.7354),
    ("eq.drug",     380928, 7.6325),
    ("kovter",      431884, 7.4827),
    ("stuxnet",      24960, 5.9975),
    ("stuxnet_t",     2000, 5.2128),
    ("stuxnet_t16",  16000, 5.9365),
]

GLOBAL_SEED = 42
XOR_KEY = 0xFF


def shannon_entropy(b: bytes) -> float:
    if not b:
        return 0.0
    counts = collections.Counter(b)
    n = len(b)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _dist_entropy(r: float) -> float:
    """Shannon entropy (bits) of the geometric-shaped distribution p_i ~ r**i
    over the 256 byte values, for r in (0, 1]."""
    idx = np.arange(256)
    w = r ** idx
    p = w / w.sum()
    return float(-(p * np.log2(p)).sum())


def _solve_r(target_H: float) -> float:
    """Bisect r in (0,1] so the geometric byte distribution has entropy target_H.
    r -> 1 gives the uniform distribution (H = 8); smaller r concentrates mass
    on low symbols and lowers H monotonically."""
    lo, hi = 1e-6, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _dist_entropy(mid) < target_H:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_plaintext(name: str, length: int, target_H: float, index: int) -> bytes:
    """Deterministic inert plaintext of exactly `length` bytes whose empirical
    entropy is close to `target_H`. Starts with a synthetic-identifying banner."""
    banner = (
        f"SYNTHETIC-NEUPERM-SUBSTITUTE::{name}::NOT-MALWARE::"
        f"seed={GLOBAL_SEED}::inert-entropy-shaped-noise\n"
    ).encode("ascii")
    if len(banner) > length:
        banner = banner[:length]
    body_len = length - len(banner)

    rng = np.random.default_rng(GLOBAL_SEED + index)
    if body_len > 0:
        r = _solve_r(target_H)
        idx = np.arange(256)
        w = r ** idx
        p = w / w.sum()
        # Deterministic permutation of which byte value carries which mass, so
        # the body is not a trivial ramp yet still has the target histogram.
        perm = rng.permutation(256)
        symbols = perm[rng.choice(256, size=body_len, p=p)]
        body = bytes(symbols.astype(np.uint8).tolist())
    else:
        body = b""

    out = banner + body
    assert len(out) == length, (name, len(out), length)
    # Guarantee no PE/ELF magic at the start (the ASCII banner already ensures
    # this, but assert it so a future banner edit cannot regress the property).
    assert out[:2] != b"MZ" and out[:4] != b"\x7fELF"
    return out


def encode_xor(plaintext: bytes) -> bytes:
    return bytes(b ^ XOR_KEY for b in plaintext)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="regenerate in memory and print realized stats; no files written")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    print(f"{'name':14} {'length':>8} {'target_H':>9} {'realized_H':>11}  {'banner_ok'}")
    for i, (name, length, target_H) in enumerate(TARGETS):
        plaintext = build_plaintext(name, length, target_H, i)
        realized_H = shannon_entropy(plaintext)
        banner_ok = plaintext.startswith(b"SYNTHETIC-NEUPERM-SUBSTITUTE")
        print(f"{name:14} {length:>8} {target_H:>9.4f} {realized_H:>11.4f}  {banner_ok}")
        if not args.verify:
            (here / f"{name}.xor").write_bytes(encode_xor(plaintext))
    if not args.verify:
        print(f"\nWrote {len(TARGETS)} substitute .xor files to {here}")


if __name__ == "__main__":
    main()
