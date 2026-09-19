# Reproducing the paper

This document maps each table / figure / numerical claim in the paper to the
script that generates it and the CSV that materializes it, and lists the
external data, models, and payloads you must supply. Run every script from the
repository root with the `neuperm` environment activated; each writes its CSV
into `results/`, where reference copies are already committed.

## Environment

```bash
conda env create -n neuperm -f environment.yaml
conda activate neuperm
pip install -e .
```

The pinned, verified environment (see `environment.yaml` / `requirements-lock.txt`):

| Package | Version |
|---|---|
| Python | 3.9.18 |
| PyTorch | 2.1.0 (CUDA 11.8) |
| torchvision | 0.16.0 |
| numpy | 1.25.0 |
| transformers | 4.45.2 |
| tokenizers | 0.20.3 |
| safetensors | 0.7.0 |

The coverage, CNN-accuracy, EvilModel, adaptive-unshuffling, LLM-benchmark and
SNR experiments were run in this environment (Xeon Silver 4310 CPU / A30
GPU-class hardware). The overhead and memory tables were measured on a second
machine (Python 3.11, PyTorch 2.5.1, CUDA 12.4); those wall-clock and memory
numbers are hardware-specific and are not expected to be bit-reproducible on
other machines.

## Data and models you must obtain

Some inputs cannot be redistributed with this artifact. Obtain them locally and
point the code at them.

- **ImageNet-1k (ILSVRC-2012) validation split (50,000 images)** — not
  redistributable. Download from <https://image-net.org/> and arrange it so that
  `$IMAGENET12_ROOT/val/<class_id>/<image>.JPEG` is browsable by
  `torchvision.datasets.ImageFolder`. Set it with `export
  IMAGENET12_ROOT=/path/to/imagenet12`, or edit `IMAGENET12_ROOT` in
  `neu_perm/config.py`. `exp_table3_sweep.py` also honours
  `NEUPERM_IMAGENET12_VAL`.
- **HuggingFace models** — accept each model's license on HuggingFace, then pull
  locally. Pin the exact checkpoint commits the paper used:
  - `meta-llama/Llama-3.2-1B-Instruct` @ `9213176726f574b556790deb65791e0c5aa438b6`
  - `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`

  CNN weights come from torchvision's `DEFAULT` aliases; pin torchvision
  `0.16.0` so `DEFAULT` resolves to the same checkpoints (the EvilModel accuracy
  check specifically uses `IMAGENET1K_V1`).
- **MaleficNet third-party code** — the MaleficNet SNR experiments reproduce
  stego models using the MaleficNet authors' released code, which is **not
  vendored here**. Clone it yourself and expose it with `export
  MALEFICNET_DIR=/path/to/maleficnet` (the SNR scripts default to
  `./external_code/maleficnet`). The MaleficNet stego checkpoints are trained on
  CIFAR-10 through that pipeline. CIFAR-10, SQuAD v1.1, BoolQ, and WikiText-2 are
  downloaded automatically on first use.

## Payloads — synthetic substitutes vs. originals

The SNR experiments were computed on nine real malware samples. Those
live samples are **not** included. Instead:

- `payloads/synthetic/` ships **inert, deterministic substitutes** (seed=42):
  length-matched and entropy-matched to the originals, banner-tagged
  `SYNTHETIC-NEUPERM`, with no PE/ELF header. Default runs use these, so the
  pipeline runs end-to-end out of the box.
- `payloads/MANIFEST.sha256` records the SHA-256 of each original (and of each
  substitute). A hash is a one-way fingerprint and distributes no malware; an
  authorized researcher can use it to re-source the exact originals from a public
  malware corpus (e.g. [theZoo](https://github.com/ytisf/theZoo), archive
  password `infected`; VirusShare; MalwareBazaar) and reproduce the published
  numbers to the decimal. Point the code at real originals with `export
  MALWARE_DIR=...` (or `NEUPERM_PAYLOAD_DIR=...`).
- **`ardamax` and `zeus-bank` are not provided in any form** — no bytes and no
  synthetic substitute. A few SNR cells (VGG16 Zeus-Bank / Ardamax, Qwen Cerber
  / Ardamax) depend on them, so **those specific cells are not reproducible from
  this artifact**; they require the user's own original payloads.

## Master index — paper artifact → script → result CSV(s)

| Paper artifact | Script | Result CSV(s) |
|---|---|---|
| Table 1 — permutable-parameter coverage (`tab:neuperm_cnns`) | `experiments/exp_perm_coverage.py` | `perm_coverage.csv` (`neuperm_cnns_table.tex`) |
| Table 3 — CNN top-1 accuracy on ImageNet under each disruption method (9 CNNs) | `experiments/exp_table3_sweep.py` | `table3_sweep.csv`, `table3_sweep_summary.csv`, `table3_validation.csv`, `table3_validation_summary.csv`, `{model}_imagenet12.csv` |
| CNN accuracy — new architectures (EfficientNet-B0/B4, MobileNet-V2/V3-Small) | `experiments/exp_new_models_accuracy.py`, `experiments/exp_new_models_full.py` | `accuracy_efficientnet_b0.csv`, `accuracy_efficientnet_b4.csv`, `accuracy_mobilenet_v2.csv`, `accuracy_mobilenet_v3_small.csv` |
| Quantization accuracy / payload BER vs. NeuPerm (`quantization_table.tex`) | `experiments/exp_quantization.py`, `experiments/exp_quant_accuracy_only.py`, `experiments/exp_quant_accuracy_ptq2.py` | `quantization_ber_{densenet121,resnet50,resnet101,vgg11,vgg16}.csv`, `quant_accuracy_{densenet121,resnet50,resnet101,vgg11,vgg16}.csv`, `quant_accuracy_ptq2.csv` |
| LLM benchmark under the mitigation sweep — Llama-3.2-1B | `experiments/exp_llm_squad_x3.py`, `experiments/exp_llm_squad_ptq.py`, `experiments/exp_llm_boolq.py`, `experiments/exp_llm_wikitext_ppl.py` | `llama_squad_x3.csv`, `llama-3.2-1b_squad.csv`, `llm_squad_ptq.csv`, `llama_boolq.csv`, `llama_wikitext_ppl.csv` |
| LLM benchmark under the mitigation sweep — Qwen2.5-1.5B | `experiments/exp_qwen_squad.py`, `experiments/exp_qwen_boolq.py`, `experiments/exp_qwen_wikitext_ppl.py` | `qwen_squad_x2.csv`, `qwen_boolq.csv`, `qwen_wikitext_ppl.csv` |
| Adaptive unshuffling (canonical-ordering) attack + tie analysis | `experiments/exp_canonical.py`, `experiments/exp_canonical_tie_corpus.py`, `experiments/exp_canonical_tie_lsb.py` | `canonical_attack_{model}.csv`, `canonical_stego_{model}.csv`, `canonical_theory_{model}.csv`, `canonical_ties_{model}.csv`, `canonical_ties_corpus.csv`, `canonical_ties_lsb.csv` |
| EvilModel (byte-substitution) generalization — inject / extract / verify | `experiments/exp_evilmodel_neuperm.py` | `evilmodel_neuperm.csv` |
| Overhead — wall-clock (NeuPerm and baselines) | `experiments/exp_overhead.py`, `experiments/exp_overhead_baselines.py` | `neuperm_overhead.csv`, `baselines_overhead.csv` |
| Overhead — peak memory | `experiments/exp_overhead_memory.py` | `overhead_memory.csv` |
| Section VIII-B — MaleficNet SNR, modern-CNN rows (Table 5 / Figure 3) | `experiments/exp_vgg16_multi_payload_snr.py`, `experiments/exp_efficientnet_b0_stuxnet_snr.py`, `experiments/exp_efficientnet_b4_multi_payload_snr.py`, `experiments/exp_mobilenet_stuxnet_snr.py`, `experiments/exp_mobilenet_v3_small_stuxnet_snr.py`, `experiments/exp_ptq_snr_injected.py` | `vgg16_snr_{payload}.csv`, `efficientnetb0_stuxnet_snr.csv`, `efficientnet_b4_multi_payload_snr.csv`, `mobilenetv2_stuxnet_snr.csv`, `mobilenetv2_stuxnet_t_snr.csv`, `mobilenetv3s_stuxnet_t16_snr.csv`, `ptq_snr_injected.csv`, `neuperm_snr_np2.csv` |
| Section VIII-B — MaleficNet SNR, CIFAR-10 stegomodel rows (Table 5 / Figure 3) | `experiments/exp_maleficnet_baseline_snr.py`, `experiments/exp_maleficnet_quant_sd.py`, `experiments/exp_maleficnet_extract_snr.py` | `maleficnet_baseline_snr.csv`, `maleficnet_ptq_snr.csv` |
| Section VIII-B — MaleficNet SNR, Qwen carrier | `experiments/exp_maleficnet_qwen_snr.py`, `experiments/exp_maleficnet_qwen_one.py` | `maleficnet_qwen_baseline_snr{,_*}.csv`, `maleficnet_qwen_neuperm_snr{,_*}.csv` |
| Spread-spectrum payload destruction (MaleficNet-class ablation), LLMs | `experiments/exp_spreadspectrum_llm.py` | `spread_spectrum_llm.csv` |
| Functional-equivalence checks | `pytest tests/` | — |

## Reproduction order

1. **Bootstrap.** Create the conda environment, install in editable mode, and set
   `IMAGENET12_ROOT` (plus `MALEFICNET_DIR` if you are reproducing the SNR
   corpus).
2. **Sanity-check.** `pytest tests/` confirms the library behaves as expected —
   permutation correctness/equivalence, coverage, quantization, and steganography.
3. **Table 1.** `python experiments/exp_perm_coverage.py` — fast, no payloads
   required.
4. **Table 3 and the CNN/LLM benchmarks.** Run `exp_table3_sweep.py` and the
   `exp_new_models_*`, `exp_quant*`, `exp_llm_*`, and `exp_qwen_*` scripts; these
   need `IMAGENET12_ROOT` and, for the LLMs, the HuggingFace weights (downloaded
   on first use).
5. **Adaptive unshuffling.** Run the `exp_canonical*` scripts — pretrained
   weights only, no malware.
6. **EvilModel and overhead.** Run `exp_evilmodel_neuperm.py`, `exp_overhead.py`,
   `exp_overhead_baselines.py`, and `exp_overhead_memory.py`.
7. **MaleficNet SNR corpus.** With MaleficNet code and CIFAR-10 stego
   checkpoints in place, run the `exp_*_snr.py` and `exp_maleficnet_*` scripts.
   These default to the inert substitutes in `payloads/synthetic/`; set
   `NEUPERM_PAYLOAD_DIR` (or `MALWARE_DIR`) to reproduce the exact published
   numbers from the original payloads.

Example:

```bash
export IMAGENET12_ROOT=/path/to/imagenet12
python experiments/exp_perm_coverage.py
python experiments/exp_table3_sweep.py
python experiments/exp_evilmodel_neuperm.py

# SNR corpus needs MaleficNet code + checkpoints (see "Data and models"):
export MALEFICNET_DIR=/path/to/maleficnet
python experiments/exp_maleficnet_extract_snr.py --all
```

The overhead scripts accept `OVERHEAD_REPEATS`, `OVERHEAD_TAG` / `OVERHEAD_OUT`,
and `OVERHEAD_WARMUP_MODEL` environment overrides (see their docstrings).

## Notes on the committed results

- `results/archive/neuperm_snr_np2_projected.csv` holds **estimated** (not
  measured) SNR values for a few low-coverage architecture / payload
  combinations; its companion `README.md` explains the projection and its error
  bounds. Do not cite any `source=projected` row as a measured result.
- `results/prev_run/` retains earlier CSVs for a small number of cells, kept for
  comparison only.
