# NeuPerm

# Setup
- Create a `conda` Python environment: `conda create --channel-priority flexible -n neuperm -f environment.yaml`.
- Set `IMAGENET12_ROOT` in [neu_perm/config.py] to a directory that contains the files of the ImageNet12 validation dataset.

# Running Experiments
## Experiment 1: Model Performance
- Configure run parameters in [experiments/exp1.py]:
    - `model_names`: List of model names; options are: `['densenet121','resnet50','resnet101','vgg11','llama-3.2-1b']`
    - `prune_amounts`: List of floats between 0 and 1; what fraction of the parameters to prune.
    - `epsilons`: List of floats to use as the scale of the normal random noise.
    - `n_repeats`: Positive integer. number of times to repeat experiments.
    - `device`: Either `'cpu'` or `'cuda'`. What device to run models on.
    - `batch_size`: Batch size for inference.
- Run the script:
    - (Recommended) run detached from terminal: `nohup python -u experiments/exp1.py > out_exp1.txt`