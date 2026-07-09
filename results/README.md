# Results

Result CSVs backing the paper's tables and figures. Each is written (and
regenerated) by the matching script in `experiments/`; the committed copies let
every table/figure rebuild from CSV without a GPU re-run.

| CSV | Script | Paper artifact |
| --- | --- | --- |
| `<model>_imagenet12.csv` (9) | `exp1_cnn_accuracy.py` | Table 3 — CNN accuracy under mitigations |
| `llm_eval.csv` | `exp1_llm_eval.py` | Table 4 — Llama/Qwen SQuAD, BoolQ, WikiText-2 |
| `exp2_snr.csv` | `exp2_maleficnet_snr.py` | Table 5 + Fig 3 — MaleficNet SNR / spread-spectrum BER |
| `evilmodel_neuperm.csv` | `exp3_evilmodel.py` | EvilModel byte-substitution |
| `canonical_*.csv` | `exp_adaptive_canonical.py` | Unshuffle attack + Lemma V |
| `neuperm_overhead.csv` | `exp_overhead.py` | Overhead + peak memory |
| `quantization_ber_<model>.csv` | `exp_quantization_ber.py` | Supplementary quantization BER |
| `neuperm_cnns_table.tex`, `quantization_table.tex` | — | Prebuilt LaTeX tables |

## Schemas

- **`<model>_imagenet12.csv`** — `model_name, accuracy, time, dataset, method,
  method_kwargs`. `method` ∈ {`original`, `neuperm`, `noise`, `prune`,
  `quantization`}; `method_kwargs` holds the per-method parameters (e.g.
  `{'eps': 0.001}`, `{'n_bits': 8, 'per_channel': False}`).
- **`llm_eval.csv`** — `model, benchmark, method, method_kwargs, seed, score`.
  `benchmark` ∈ {`squad` (F1), `boolq` (accuracy %), `wikitext2` (perplexity)}.
- **`exp2_snr.csv`** — `mode, model, payload, condition, metric, value, repeat,
  seed, gamma, error`. `condition` ∈ {`clean`, `after_neuperm`, quantization
  variants}; `metric` is `snr` (fork mode) or `ber` (spread-spectrum mode); the
  `payload` column names the malware family.
- The remaining CSVs use the columns their generating script writes; see the
  script header for details.

Re-running a script overwrites its CSV(s) in place.
