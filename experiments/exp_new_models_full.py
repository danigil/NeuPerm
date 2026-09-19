"""Full evaluation for new architectures: pruning, noise, quantization, NeuPerm."""
import copy, functools, time, os, sys, gc
from typing import Literal
import pandas as pd
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.utils.prune as prune

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neu_perm.config import IMAGENET12_ROOT, RESULTS_DIR
from neu_perm.config import require_imagenet12
from neu_perm.perm import permute_model
from neu_perm.quantization import quantize_model

def log(msg):
    print(msg, flush=True)

def extract_weights_pytorch(model):
    ws = [w.cpu().detach().numpy().flatten() for w in model.parameters()]
    return np.concatenate(ws)

def load_weights_from_flattened_vector_torch(model, model_weights, inplace=False):
    if inplace:
        model_curr = model
    else:
        model_curr = copy.deepcopy(model)
    torch.nn.utils.vector_to_parameters(torch.from_numpy(model_weights.copy()), model_curr.parameters())
    return model_curr

def torch_eval_cnn(model, testloader, device='cuda'):
    model = model.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def prune_model(model, amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.random_unstructured(module, name='weight', amount=amount)

def noise_model(model, eps=1e-3):
    w = extract_weights_pytorch(model)
    w += np.random.normal(0, eps, w.shape).astype(w.dtype)
    return load_weights_from_flattened_vector_torch(model, w)

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def load_pretrained(model_name):
    weights = torchvision.models.get_model_weights(model_name).DEFAULT
    preprocess = weights.transforms()
    try:
        model = torchvision.models.get_model(model_name, weights=weights)
    except RuntimeError:
        import urllib.parse
        url = weights.url
        filename = os.path.basename(urllib.parse.urlparse(url).path)
        cached = os.path.join(torch.hub.get_dir(), 'checkpoints', filename)
        if not os.path.exists(cached):
            torch.hub.download_url_to_file(url, cached)
        sd = torch.load(cached, map_location='cpu')
        model = torchvision.models.get_model(model_name, weights=None)
        model.load_state_dict(sd)
    return model, preprocess

MODEL_NAMES = ['efficientnet_b0', 'efficientnet_b4', 'mobilenet_v2', 'mobilenet_v3_small']
PRUNE_AMOUNTS = [0.01, 0.05]
EPSILONS = [1e-4, 1e-3, 1e-2, 1e-1]
QUANT_BITS = [8, 4, 2]
N_REPEATS = 3
DEVICE = 'cuda'
BATCH_SIZE = 64

os.makedirs(RESULTS_DIR, exist_ok=True)

for model_name in MODEL_NAMES:
    log(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")

    model_orig, preprocess = load_pretrained(model_name)
    log(f"  Model loaded")

    require_imagenet12()  # fail loudly here, not at config import
    val_dir = os.path.join(IMAGENET12_ROOT, 'val')
    imagenet_ds = torchvision.datasets.ImageFolder(val_dir, transform=preprocess)
    imagenet_dl = torch.utils.data.DataLoader(imagenet_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    eval_fn = functools.partial(torch_eval_cnn, testloader=imagenet_dl, device=DEVICE)

    sd_orig = copy.deepcopy(model_orig.cpu().state_dict())
    results = []

    # Original
    model = copy.deepcopy(model_orig).to(DEVICE)
    acc_orig = eval_fn(model)
    log(f"  Original: {acc_orig:.4f}")
    results.append({'model_name': model_name, 'method': 'original', 'accuracy': acc_orig,
                    'method_kwargs': '{}'})
    del model; cleanup()

    # Pruning
    for amount in PRUNE_AMOUNTS:
        for i in range(N_REPEATS):
            model = copy.deepcopy(model_orig).to(DEVICE)
            model.load_state_dict(sd_orig)
            prune_model(model, amount=amount)
            acc = eval_fn(model)
            log(f"  Prune {amount} ({i}): {acc:.4f} (delta={acc - acc_orig:+.4f})")
            results.append({'model_name': model_name, 'method': 'prune',
                            'accuracy': acc, 'method_kwargs': str({'amount': amount})})
            del model; cleanup()

    # Noise
    for eps in EPSILONS:
        for i in range(N_REPEATS):
            model = copy.deepcopy(model_orig).to(DEVICE)
            model.load_state_dict(sd_orig)
            model = noise_model(model, eps=eps)
            model = model.to(DEVICE)
            acc = eval_fn(model)
            log(f"  Noise {eps} ({i}): {acc:.4f} (delta={acc - acc_orig:+.4f})")
            results.append({'model_name': model_name, 'method': 'noise',
                            'accuracy': acc, 'method_kwargs': str({'eps': eps})})
            del model; cleanup()

    # Quantization (per-channel only)
    for n_bits in QUANT_BITS:
        model_q = quantize_model(model_orig, n_bits=n_bits, per_channel=True).to(DEVICE)
        acc = eval_fn(model_q)
        log(f"  PTQ-{n_bits}bit perchannel: {acc:.4f} (delta={acc - acc_orig:+.4f})")
        results.append({'model_name': model_name, 'method': 'quantization',
                        'accuracy': acc, 'method_kwargs': str({'n_bits': n_bits, 'per_channel': True})})
        del model_q; cleanup()

    # NeuPerm
    for i in range(N_REPEATS):
        sd = copy.deepcopy(sd_orig)
        sd_perm = permute_model(model_name, sd, inplace=True)
        model = copy.deepcopy(model_orig)
        model.load_state_dict(sd_perm)
        model = model.to(DEVICE)
        del sd, sd_perm
        acc = eval_fn(model)
        log(f"  NeuPerm ({i}): {acc:.4f} (delta={acc - acc_orig:+.4f})")
        results.append({'model_name': model_name, 'method': 'neuperm',
                        'accuracy': acc, 'method_kwargs': '{}'})
        del model; cleanup()

    # Save per model
    df = pd.DataFrame(results)
    path = f"{RESULTS_DIR}/{model_name}_imagenet12.csv"
    df.to_csv(path, index=False)
    log(f"  Saved to {path}")

    del model_orig, sd_orig; cleanup()

log("\nDone.")
