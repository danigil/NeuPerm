"""Generate synthetic canary payloads for the NeuPerm SNR experiments.

The public NeuPerm repo ships **no real malware**. To let the Experiment 2
(MaleficNet SNR) pipeline run out of the box, this script writes random-byte
"canary" payloads of the exact sizes of the real samples listed in MANIFEST.csv.

Real-payload reproduction: obtain the samples named in MANIFEST.csv from theZoo
(https://github.com/ytisf/theZoo), verify each against its ``sha256_xor`` after
applying the documented XOR-0xFF obfuscation, and drop the ``.xor`` files here.
See README.md.

Deterministic: each payload's bytes are seeded from a fixed base seed and the
payload name, so canaries regenerate identically across machines.

Usage:
    python payloads/make_canary.py            # writes <name>.bin for every row
    python payloads/make_canary.py --seed 123
"""
import argparse
import csv
import os
import random
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "MANIFEST.csv")
BASE_SEED = 20260709  # fixed for reproducibility


def canary_bytes(name: str, size: int, base_seed: int) -> bytes:
    seed = (base_seed ^ zlib.crc32(name.encode())) & 0xFFFFFFFF
    rng = random.Random(seed)
    return bytes(rng.getrandbits(8) for _ in range(size))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()

    with open(MANIFEST, newline="") as fh:
        rows = list(csv.DictReader(fh))

    for row in rows:
        name = row["name"]
        size = int(row["size_bytes"])
        # canary output mirrors the real filename but with a .bin extension
        stem = name[:-4] if name.endswith(".xor") else name
        out = os.path.join(args.out_dir, f"{stem}.bin")
        with open(out, "wb") as fh:
            fh.write(canary_bytes(name, size, args.seed))
        print(f"wrote canary {out} ({size} bytes)")


if __name__ == "__main__":
    main()
