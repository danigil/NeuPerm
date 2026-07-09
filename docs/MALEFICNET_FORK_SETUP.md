# MaleficNet fork setup (Experiment 2, fork mode)

The default `spread_spectrum_llm` mode of `experiments/exp2_maleficnet_snr.py`
is self-contained and needs nothing here. This document is only for the optional
`maleficnet_fork` mode, which reproduces the exact fork-based MaleficNet SNR
numbers (Table 5 / Fig 3) by injecting and extracting real payloads with the
original MaleficNet code.

You need two external ingredients that this repository does not ship:

1. **The MaleficNet fork** — the patched MaleficNet injector/extractor used in
   the paper (MIT-licensed). Clone the upstream MaleficNet project and apply your
   local patches, or use your existing fork checkout. It must contain the
   injector/extractor/`maleficnet` modules and `message_lengths.csv` (per-model
   payload bit-lengths, consumed by the extractor at runtime).

2. **Real payloads** — obtain the 9 theZoo samples listed in
   `payloads/MANIFEST.csv`, apply the XOR-0xFF obfuscation, and place the
   resulting `<name>.xor` files in a payload directory. See `payloads/README.md`.
   (For a dry run of the pipeline, the canary `.bin` files from
   `payloads/make_canary.py` work in place of real payloads.)

## Point the script at them via environment variables

```bash
export NEUPERM_MALEFICNET_DIR=/path/to/your/maleficnet/fork   # added to sys.path
export NEUPERM_PAYLOAD_DIR=/path/to/payloads                  # default: ./payloads
```

Then set `MODE = "maleficnet_fork"` in the config block of
`experiments/exp2_maleficnet_snr.py` and run it. If `NEUPERM_MALEFICNET_DIR` is
unset or missing, the script prints guidance and exits cleanly instead of
crashing.

No absolute paths are baked into the code — every external location comes from
the environment variables above.
