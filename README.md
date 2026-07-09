# NeuPerm

NeuPerm is a zero-retraining, zero-accuracy-cost model-sanitization primitive
that disrupts steganographic malware hidden in neural-network parameters by
applying function-preserving random permutations to a model's units/channels.
This repository reproduces the experiments in the NeuPerm paper.

## Repository layout

```
neu_perm/            core library
  perm.py            function-preserving permutations (CNNs + Llama/Qwen)
  canonical.py       canonical (permutation-invariant) ordering + adaptive attack
  countermeasures.py tie-breaking countermeasure
  quantization.py    post-training quantization + BER helpers
  steganography.py   LSB / EvilModel / spread-spectrum embed & extract
  models.py, data_loaders.py, utils.py, config.py
experiments/         one config-driven script per paper experiment (see below)
results/             canonical result CSVs — every table/figure rebuilds from these
notebooks/           results notebooks (rebuild tables + Fig 3 from results/)
payloads/            payload manifest + canary generator (NO real malware)
tests/               unit / smoke tests for the core modules
```

## Setup

```bash
conda env create -n neuperm -f environment.yaml
conda activate neuperm
```

Set the ImageNet-12 validation path (only needed for the CNN-accuracy
experiment) via an environment variable or by editing `neu_perm/config.py`:

```bash
export NEUPERM_IMAGENET12_ROOT=/path/to/imagenet/val
```

## Experiments

Each script has an in-file configuration block at the bottom (`if __name__ ==
"__main__":`) — edit the model list, grids, seeds, and device there, then run it.

| Script | Paper artifact | Notes |
| --- | --- | --- |
| `experiments/exp1_cnn_accuracy.py` | Table 3 | 9 CNNs × {baseline, NeuPerm, noise, prune, 8/4/2-bit PTQ} on ImageNet-12 |
| `experiments/exp1_llm_eval.py` | Table 4 | Llama-3.2-1B-Instruct + Qwen2.5-1.5B-Instruct on SQuAD, BoolQ, WikiText-2 |
| `experiments/exp2_maleficnet_snr.py` | Table 5 + Fig 3 | MaleficNet SNR; self-contained spread-spectrum mode by default, fork mode optional (see Payloads) |
| `experiments/exp3_evilmodel.py` | EvilModel table | byte-substitution embed → NeuPerm → extract |
| `experiments/exp_adaptive_canonical.py` | Unshuffle attack + Lemma V | canonical-ordering adaptive adversary + tie analyses |
| `experiments/exp_overhead.py` | Overhead + memory tables | wall-clock + peak CPU/GPU memory |
| `experiments/exp_quantization_ber.py` | Supplementary BER | payload bit-error-rate under PTQ vs NeuPerm |

Example:

```bash
python experiments/exp1_cnn_accuracy.py
```

## Reproducing tables and figures

The committed CSVs in `results/` are the paper's data, so the tables and figures
rebuild **without** re-running the experiments. `notebooks/exp1_results.ipynb`
and `notebooks/exp2_results.ipynb` read those CSVs to regenerate the accuracy
tables and the SNR figure (Fig 3). Re-running an experiment script overwrites the
corresponding CSV. See `results/README.md` for the schema of each file, and
`REPRODUCTION_GUIDE.md` for a step-by-step guide.

## Payloads and safety

This repository ships **no malware**. `payloads/` contains only a manifest of the
9 theZoo samples used in the paper (family, size, hash, source) and
`make_canary.py`, which writes synthetic random-byte payloads of matching sizes
so the SNR pipeline runs out of the box:

```bash
python payloads/make_canary.py
```

The default (`spread_spectrum_llm`) mode of `exp2_maleficnet_snr.py` is fully
self-contained. Reproducing the exact fork-based MaleficNet SNR numbers requires
the external (MIT-licensed) MaleficNet fork and real payloads; see
`payloads/README.md`.

## License

See [LICENSE](LICENSE).
