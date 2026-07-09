"""Experiment 1 (CNNs): NeuPerm does not degrade performance.

Reproduces Table 3 (`tab:exp1_acc`): ImageNet-12 top-1 accuracy of the nine
CNNs under each steganography-mitigation technique — the unmodified baseline,
NeuPerm (ours), additive Gaussian noise, random pruning, and post-training
quantization (8/4/2-bit, per-tensor and per-channel).

Each model writes one CSV `<RESULTS_DIR>/<model>_imagenet12.csv` with a single
schema, so Table 3 rebuilds directly from the CSVs.

Configure the run in the `__main__` block at the bottom (model list, mitigation
grids, device, batch size), then:

    python experiments/exp1_cnn_accuracy.py

Set the ImageNet-12 validation path via `NEUPERM_IMAGENET12_ROOT` or in
`neu_perm/config.py`. The LLMs (Llama/Qwen, Table 4) live in
`experiments/exp1_llm_eval.py`.
"""
import copy
import functools
import os
import time
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import tqdm

from neu_perm.config import RESULTS_DIR, get_imagenet12_root
from neu_perm.perm import permute_model
from neu_perm.quantization import quantize_model

# The nine CNNs evaluated in the paper (Table 3).
CNN_MODELS = [
    "densenet121",
    "resnet50",
    "resnet101",
    "vgg11",
    "vgg16",
    "efficientnet_b0",
    "efficientnet_b4",
    "mobilenet_v2",
    "mobilenet_v3_small",
]


def extract_weights_pytorch(model: torch.nn.Module) -> np.ndarray:
    ws = [w.cpu().detach().numpy().flatten() for w in model.parameters()]
    return np.concatenate(ws)


def load_weights_from_flattened_vector_torch(model, model_weights: np.ndarray, inplace: bool = False):
    model_curr = model if inplace else copy.deepcopy(model)
    params = model_curr.parameters()
    torch.nn.utils.vector_to_parameters(torch.from_numpy(model_weights.copy()), params)
    return model_curr


def torch_eval_cnn(model, testloader, device="cuda", verbose=False):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm.tqdm(testloader, disable=not verbose):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def prune_model(model, amount=0.2):
    for _, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.random_unstructured(module, name="weight", amount=amount)


def noise_model(model, eps=1e-3):
    """Add zero-mean Gaussian noise (std ``eps``) to every parameter."""
    w = extract_weights_pytorch(model)
    w += np.random.normal(0, eps, w.shape).astype(w.dtype)
    return load_weights_from_flattened_vector_torch(model, w)


def neuperm_model(model, model_name, device="cuda"):
    sd = model.state_dict()
    sd_perm = permute_model(model_name=model_name, sd=sd, inplace=True)
    model.load_state_dict(sd_perm)
    return model.to(device)


def single_exp(
    model_name: str,
    device: str = "cuda",
    imagenet12_root: Optional[str] = None,
    batch_size: int = 128,
    prune_amounts: Optional[List[float]] = None,
    epsilons: Optional[List[float]] = None,
    quant_configs: Optional[List[tuple]] = None,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Evaluate one CNN under every mitigation and write its results CSV."""
    if prune_amounts is None:
        prune_amounts = [0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.9, 0.99]
    if epsilons is None:
        epsilons = [1e-4, 1e-3, 1e-2, 1e-1]
    if quant_configs is None:
        # (n_bits, per_channel)
        quant_configs = [(8, False), (8, True), (4, False), (4, True), (2, False), (2, True)]
    if imagenet12_root is None:
        imagenet12_root = get_imagenet12_root()

    dataset_name = "imagenet12"

    weights = torchvision.models.get_model_weights(model_name).DEFAULT
    preprocess = weights.transforms()
    model_orig = torchvision.models.get_model(model_name, weights=weights).to(device)

    imagenet12_ds = torchvision.datasets.ImageNet(imagenet12_root, split="val", transform=preprocess)
    imagenet12_dl = torch.utils.data.DataLoader(
        imagenet12_ds, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=False
    )
    torch_eval_curr = functools.partial(torch_eval_cnn, testloader=imagenet12_dl, device=device)

    sd_orig = copy.deepcopy(model_orig.to("cpu").state_dict())

    def fresh_model():
        m = copy.deepcopy(model_orig.to("cpu")).to(device)
        m.load_state_dict(sd_orig)
        return m

    def record(results, method, accuracy, time_diff, method_kwargs):
        results.append({
            "model_name": model_name,
            "accuracy": accuracy,
            "time": time_diff,
            "dataset": dataset_name,
            "method": method,
            "method_kwargs": method_kwargs,
        })

    results = []

    acc = torch_eval_curr(fresh_model())
    print(f"\tOriginal accuracy: {acc}")
    record(results, "original", acc, 0.0, {})

    for i in range(n_repeats):
        for eps in epsilons:
            model = fresh_model()
            t0 = time.time()
            model_noise = noise_model(model, eps=eps)
            dt = time.time() - t0
            acc_noise = torch_eval_curr(model_noise)
            print(f"\tNoise accuracy ({i}) (eps={eps}): {acc_noise}")
            record(results, "noise", acc_noise, dt, {"eps": eps})

    for i in range(n_repeats):
        for prune_amount in prune_amounts:
            model = fresh_model()
            t0 = time.time()
            prune_model(model, amount=prune_amount)
            dt = time.time() - t0
            acc_pruned = torch_eval_curr(model)
            print(f"\tPruned accuracy ({i}) (amount={prune_amount}): {acc_pruned}")
            record(results, "prune", acc_pruned, dt, {"amount": prune_amount})

    for n_bits, per_channel in quant_configs:
        model = fresh_model()
        t0 = time.time()
        model_quant = quantize_model(model, n_bits=n_bits, per_channel=per_channel).to(device)
        dt = time.time() - t0
        acc_quant = torch_eval_curr(model_quant)
        pc_str = "perchannel" if per_channel else "pertensor"
        print(f"\tQuantization accuracy ({n_bits}bit {pc_str}): {acc_quant}")
        record(results, "quantization", acc_quant, dt, {"n_bits": n_bits, "per_channel": per_channel})

    for i in range(n_repeats):
        model = fresh_model()
        t0 = time.time()
        model_neuperm = neuperm_model(model, model_name, device=device)
        dt = time.time() - t0
        acc_neuperm = torch_eval_curr(model_neuperm)
        print(f"\tNeuPerm accuracy ({i}): {acc_neuperm}")
        record(results, "neuperm", acc_neuperm, dt, {})

    df = pd.DataFrame(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = f"{RESULTS_DIR}/{model_name}_imagenet12.csv"
    df.to_csv(result_path, index=False)
    print(f"Results saved to {result_path}")
    return df


if __name__ == "__main__":
    # --- run configuration ---
    model_names = CNN_MODELS
    prune_amounts = [0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.9, 0.99]
    epsilons = [1e-4, 1e-3, 1e-2, 1e-1]
    quant_configs = [(8, False), (8, True), (4, False), (4, True), (2, False), (2, True)]
    n_repeats = 10
    device: Literal["cuda", "cpu"] = "cuda"
    batch_size = 128
    # -------------------------

    for model_name in model_names:
        print(f"Evaluating {model_name}")
        single_exp(
            model_name,
            device=device,
            batch_size=batch_size,
            prune_amounts=prune_amounts,
            epsilons=epsilons,
            quant_configs=quant_configs,
            n_repeats=n_repeats,
        )
