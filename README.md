# NeuPerm

Code for the paper *NeuPerm: Disrupting the Extraction of Malware Hidden in
Neural Network Parameters by Leveraging Permutation Symmetry*.

NeuPerm is a zero-cost, zero-retraining sanitization step that disrupts malware
hidden in pretrained model weights by reordering permutation-equivalent units.
The permutation leaves the model-level computation unchanged while destroying
payloads embedded by weight-steganography attacks such as MaleficNet and
EvilModel, at negligible cost to model accuracy.

## Quickstart — apply NeuPerm

```python
from neu_perm.perm import permute_model

# any supported CNN (vgg11/16, resnet50/101, densenet121,
# efficientnet_b0/b4, mobilenet_v2/v3_small) or LLM (llama-3.2-1b, qwen2.5-1.5b)
sanitized_sd = permute_model("vgg16", model.state_dict())
model.load_state_dict(sanitized_sd)
```

## License

Patent-pending. Licensed for non-commercial, non-derivative use only (Creative
Commons Attribution-NonCommercial-NoDerivatives 4.0 International) — see
[`LICENSE`](LICENSE).

## Reproduce the paper

The repository provides one script per paper experiment, and the CSVs that back
the paper's tables and figures are committed under `results/`. See
[`REPRODUCTION.md`](REPRODUCTION.md) for the paper-artifact → script/CSV index
and for the datasets, models, and payloads you must supply yourself.

### Environment

```bash
conda env create -n neuperm -f environment.yaml
conda activate neuperm
pip install -e .
```

Dependencies are pinned for Python 3.9.18, PyTorch 2.1.0 (CUDA 11.8) and
torchvision 0.16.0, with HuggingFace `transformers` for the two LLMs; the exact
set is in `environment.yaml` and `requirements-lock.txt`.

> **Note:** if you have `pyenv` installed, its shims may stay ahead of conda's
> bin on `PATH` even after `conda activate`. Verify with `which python` — it
> should print `.../envs/neuperm/bin/python`. If it returns a `~/.pyenv/shims/`
> path, run `pyenv shell system` first, or invoke the env's binary directly.

## Citing

See [`CITATION.cff`](CITATION.cff).
