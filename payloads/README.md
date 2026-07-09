# Payloads

Experiment 2 (MaleficNet SNR, Table 5 / Fig 3) and Experiment 3 (EvilModel)
embed a real-malware payload into model parameters and measure whether NeuPerm
disrupts its recovery.

**This public repository ships no malware.** It contains only:

- `MANIFEST.csv` — the 9 payloads used in the paper: filename,
  malware family, byte size, `sha256_xor` (SHA-256 of the XOR-0xFF-obfuscated
  blob), obfuscation scheme, and source.
- `make_canary.py` — writes synthetic random-byte `*.bin` canaries of the exact
  same sizes so the SNR pipeline runs end to end without any real sample.

`*.xor`, `*.bin`, `*.zip`, and `decoded/` are git-ignored so no payload bytes are
ever committed.

## Run with canaries (default, safe)

```bash
python payloads/make_canary.py       # writes stuxnet.bin, cerber.bin, ...
```

Canaries are deterministic (seeded per name), so SNR numbers are reproducible
run to run. They are random noise, **not** the paper's real-malware numbers —
use them to exercise the pipeline, not to reproduce the exact Table 5 values.

## Reproduce the real-malware numbers

1. Obtain each sample named in `MANIFEST.csv` from
   [theZoo](https://github.com/ytisf/theZoo) (the paper's source; families:
   Stuxnet, Destover, Asprox, Bladabindi, EquationDrug, Kovter, Cerber, plus two
   truncated Stuxnet variants sized for small models). Plaintext SHA-256 / VT
   references are available on theZoo / VirusTotal by family name.
2. Apply the XOR-0xFF obfuscation (byte-wise `b ^ 0xFF`) and save as
   `<name>.xor` in this directory.
3. Verify each against its `sha256_xor` in `MANIFEST.csv`.
4. Point the Experiment 2 script at the real payloads (see the experiment's
   `--payload-dir` / config option) instead of the canaries.

> The `MANIFEST.csv` byte sizes are of the obfuscated `.xor` blobs, matching what
> the SNR scripts read; canaries are generated to those sizes.
