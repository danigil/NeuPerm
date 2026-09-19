# `experiments/` — paper artifact map

Each script reproduces a specific table, figure, or numerical claim in the
paper. Run them from the repository root with the `neuperm` environment
activated; CSV outputs land in `results/`, where reference copies are already
committed. `REPRODUCTION.md` (repository root) is the full paper-artifact →
script/CSV index and lists the external data, models, and payloads you must
supply.

| Paper artifact | Script(s) | Result CSV(s) |
|---|---|---|
| Table 1 — permutable-parameter coverage | `exp_perm_coverage.py` | `results/perm_coverage.csv` |
| Table 3 — CNN top-1 accuracy under each disruption method | `exp_table3_sweep.py` | `results/table3_sweep.csv`, `results/{model}_imagenet12.csv` |
| CNN accuracy — EfficientNet / MobileNet | `exp_new_models_accuracy.py`, `exp_new_models_full.py` | `results/accuracy_{model}.csv` |
| Quantization accuracy / payload BER vs. NeuPerm | `exp_quantization.py`, `exp_quant_accuracy_only.py`, `exp_quant_accuracy_ptq2.py` | `results/quantization_ber_{model}.csv`, `results/quant_accuracy_{model}.csv` |
| LLM benchmark under the mitigation sweep — Llama-3.2-1B | `exp_llm_squad_x3.py`, `exp_llm_squad_ptq.py`, `exp_llm_boolq.py`, `exp_llm_wikitext_ppl.py` | `results/llama_*.csv`, `results/llm_squad_ptq.csv` |
| LLM benchmark under the mitigation sweep — Qwen2.5-1.5B | `exp_qwen_squad.py`, `exp_qwen_boolq.py`, `exp_qwen_wikitext_ppl.py` | `results/qwen_*.csv` |
| Adaptive unshuffling (canonical-ordering) attack + tie analysis | `exp_canonical.py`, `exp_canonical_tie_corpus.py`, `exp_canonical_tie_lsb.py` | `results/canonical_*.csv` |
| EvilModel (byte-substitution) generalization | `exp_evilmodel_neuperm.py` | `results/evilmodel_neuperm.csv` |
| Overhead — wall-clock and memory | `exp_overhead.py`, `exp_overhead_baselines.py`, `exp_overhead_memory.py` | `results/neuperm_overhead.csv`, `results/baselines_overhead.csv`, `results/overhead_memory.csv` |
| Section VIII-B — MaleficNet SNR (Table 5 / Figure 3) | `exp_vgg16_multi_payload_snr.py`, `exp_efficientnet_b0_stuxnet_snr.py`, `exp_efficientnet_b4_multi_payload_snr.py`, `exp_mobilenet_stuxnet_snr.py`, `exp_mobilenet_v3_small_stuxnet_snr.py`, `exp_ptq_snr_injected.py`, `exp_maleficnet_baseline_snr.py`, `exp_maleficnet_quant_sd.py`, `exp_maleficnet_extract_snr.py`, `exp_maleficnet_qwen_snr.py` | `results/*_snr*.csv`, `results/maleficnet_*_snr*.csv` |
| Spread-spectrum payload destruction (MaleficNet-class ablation), LLMs | `exp_spreadspectrum_llm.py` | `results/spread_spectrum_llm.csv` |

## Common environment variables

- `IMAGENET12_ROOT` — directory holding the ImageNet val split (required for the
  CNN accuracy runs); may instead be set in `neu_perm/config.py`.
  `exp_table3_sweep.py` also honours `NEUPERM_IMAGENET12_VAL`.
- `NEUPERM_PAYLOAD_DIR` / `MALWARE_DIR` — directory of `<name>.xor` payloads.
  Defaults to the inert substitutes in `payloads/synthetic/`; set it to point at
  the real originals (see `REPRODUCTION.md` and `payloads/MANIFEST.sha256`).
- `MALEFICNET_DIR` — clone of the MaleficNet third-party code (defaults to
  `./external_code/maleficnet`).
- `OVERHEAD_REPEATS`, `OVERHEAD_TAG` / `OVERHEAD_OUT`, `OVERHEAD_WARMUP_MODEL` —
  overrides for the overhead scripts (see their docstrings).

A requested payload that is absent is a hard error, never a silent skip: the SNR
tables have one row per (model, payload), so a dropped row would corrupt the
table.
