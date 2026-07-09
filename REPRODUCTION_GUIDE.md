# Reproduction guide

Step-by-step reproduction of the NeuPerm paper's results. Every table and figure
can be rebuilt from the committed CSVs in `results/` (no GPU needed); each can
also be regenerated from scratch by running the matching experiment script.

## 0. Environment

```bash
conda env create -n neuperm -f environment.yaml
conda activate neuperm
export NEUPERM_IMAGENET12_ROOT=/path/to/imagenet/val   # CNN experiment only
```

Sanity check the ported library:

```bash
pytest tests -q          # module round-trip / invariant tests
```

## 1. Rebuild tables and figures from shipped data (fast path)

```bash
jupyter nbconvert --to notebook --execute notebooks/exp1_results.ipynb   # Tables 3, 4
jupyter nbconvert --to notebook --execute notebooks/exp2_results.ipynb   # Table 5, Fig 3
```

The notebooks read only `results/*.csv`. See `results/README.md` for each file's
schema.

## 2. Regenerate a result from scratch

Each script has an in-file config block (`if __name__ == "__main__":`); edit the
model list / grids / seeds / device there and run it. The script overwrites its
canonical CSV in `results/`.

| Paper artifact | Script | Result CSV(s) |
| --- | --- | --- |
| Table 3 — CNN accuracy | `experiments/exp1_cnn_accuracy.py` | `<model>_imagenet12.csv` |
| Table 4 — LLM SQuAD/BoolQ/WikiText-2 | `experiments/exp1_llm_eval.py` | `llm_eval.csv` |
| Table 5 + Fig 3 — MaleficNet SNR | `experiments/exp2_maleficnet_snr.py` | `exp2_snr.csv` |
| EvilModel table | `experiments/exp3_evilmodel.py` | `evilmodel_neuperm.csv` |
| Unshuffle attack + Lemma V | `experiments/exp_adaptive_canonical.py` | `canonical_*.csv` |
| Overhead + memory | `experiments/exp_overhead.py` | `neuperm_overhead.csv` |
| Supplementary BER | `experiments/exp_quantization_ber.py` | `quantization_ber_<model>.csv` |

### Experiment 1 — NeuPerm does not degrade performance (Tables 3, 4)

- CNNs: `python experiments/exp1_cnn_accuracy.py` — 9 CNNs on ImageNet-12 under
  baseline / NeuPerm / noise / prune / PTQ. Expect the NeuPerm row within noise of
  the baseline (±0.5% top-1); 4-/2-bit PTQ degrades sharply.
- LLMs: `python experiments/exp1_llm_eval.py` — Llama-3.2-1B-Instruct and
  Qwen2.5-1.5B-Instruct on SQuAD (F1), BoolQ (accuracy), and WikiText-2
  (perplexity), 5 seeds. Expect the NeuPerm row within ±0.5 F1 / ±1% accuracy of
  the baseline, and a WikiText-2 perplexity change below the 1e-4 noise floor.

### Experiment 2 — NeuPerm mitigates MaleficNet (Table 5, Fig 3)

- Default self-contained mode: `python experiments/exp2_maleficnet_snr.py`
  (`MODE = "spread_spectrum_llm"`) embeds a spread-spectrum payload into an LLM
  and measures BER clean vs after NeuPerm — runs with no external dependency.
- Exact fork-based SNR across CNNs + LLMs: set `MODE = "maleficnet_fork"` and
  follow `docs/MALEFICNET_FORK_SETUP.md` (needs the external MaleficNet fork and
  real payloads). Expect post-NeuPerm SNR to fall below the extraction threshold
  on architectures with ≥50% permutable coverage.

### Experiment 3 — Disrupting a byte-exact attack (EvilModel)

`python experiments/exp3_evilmodel.py` — embed a benign byte payload via EvilModel
half-substitution, then extract before and after NeuPerm. Expect byte-exact
recovery before NeuPerm and destroyed recovery (BER ≈ 0.5) after.

### Adaptive adversary (unshuffle attack, Lemma V)

`python experiments/exp_adaptive_canonical.py` — select the analysis via the
`ANALYSIS` config var (`canonical_attack`, `tie_corpus`, `tie_lsb`). Reproduces
the canonical-ordering adaptive attack and the tie-uniqueness analyses.

### Overhead and deployability

`python experiments/exp_overhead.py` — wall-clock and peak CPU/GPU memory of
applying NeuPerm across the model suite.

### Supplementary: quantization BER

`python experiments/exp_quantization_ber.py` — payload bit-error-rate under
post-training quantization vs NeuPerm.

## Payloads

No malware ships in this repo. Generate canaries with
`python payloads/make_canary.py`; reproduce real-payload numbers per
`payloads/README.md`.
