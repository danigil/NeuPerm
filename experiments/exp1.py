import copy
import functools
import itertools
import multiprocessing
import pandas as pd
import torchvision.models as models
import torch

# torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torchvision
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import time

import torchvision.transforms as transforms

from neu_perm.data_loaders import *
from neu_perm.models import *
from neu_perm.perm import *
from neu_perm.config import *
from neu_perm.utils import *

from collections.abc import MutableMapping
import tqdm

import torch.nn.utils.prune as prune

def extract_weights_pytorch(model: torch.nn.Module) -> np.ndarray:
    ws = [w.cpu().detach().numpy().flatten() for w in model.parameters()]
    w = np.concatenate(ws)

    return w

def load_weights_from_flattened_vector_torch(model, model_weights: np.ndarray, inplace: bool = False) -> torch.nn.Module:
    import torch
    if inplace:
        model_curr = model
    else:
        model_curr = copy.deepcopy(model)

    params = model_curr.parameters()
    torch.nn.utils.vector_to_parameters(torch.from_numpy(model_weights.copy()), params)
    return model_curr

def torch_eval_cnn(model, testloader, device='cuda', verbose=False):
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm.tqdm(testloader, disable=not verbose):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def prune_model(model, amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.random_unstructured(module, name='weight', amount=amount)
        elif isinstance(module, nn.Linear):
            prune.random_unstructured(module, name='weight', amount=amount)

def noise_model(model, eps=1e-3):
    w = extract_weights_pytorch(model)
    w += np.random.normal(0, eps, w.shape).astype(w.dtype)
    model = load_weights_from_flattened_vector_torch(model, w)

    # torch_to_numpy = {
    #     torch.float16: np.float16,
    #     torch.float32: np.float32,
    #     torch.float64: np.float64,
    #     torch.int8: np.int8,
    #     torch.int16: np.int16,
    #     torch.int32: np.int32,
    #     torch.int64: np.int64,
    # }

    # sd = model.state_dict()
    # for k, v in sd.items():
    #     curr_shape = v.shape
    #     v += torch.from_numpy(np.random.normal(0, eps, curr_shape).astype(torch_to_numpy[v.dtype])).to(v.device)

    # model.load_state_dict(sd)


    return model

def neuperm_model(model, model_name, device='cuda'):
    sd = model.state_dict()
    sd_perm = permute_model(
        model_name=model_name,
        sd=sd,
        inplace=True,
    )
    model.load_state_dict(sd_perm)
    model = model.to(device)
    return model

def neuperm_exp(model, model_name, torch_eval_f, device='cuda', dataset_name='imagenet12'):
    model_cp = copy.deepcopy(model.to('cpu'))

    time_start = time.time()
    model_neuperm = neuperm_model(model_cp, model_name, device=device)
    time_end = time.time()
    time_diff = time_end - time_start
    acc_neuperm = torch_eval_f(model_neuperm)
    res_neuperm = {
        'model_name': model_name,
        'accuracy': acc_neuperm,
        'time': time_diff,
        'dataset': dataset_name,
        'method': 'neuperm',
        'method_kwargs': {},
    }
    return res_neuperm

def single_exp(
    model_name,
    device='cuda',
    imagenet12_root:Optional[str] = None,

    batch_size:int=128,

    dataset_name:Optional[str]=None,

    prune_amounts=[0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.9, 0.99],
    epsilons=[1e-4, 1e-3, 1e-2, 1e-1],

    n_repeats=10,
):
    if model_name in ['densenet121', 'resnet50', 'resnet101', 'vgg11', 'vgg16']:
        if dataset_name is None:
            dataset_name = 'imagenet12'

        if imagenet12_root is None:
            imagenet12_root = IMAGENET12_ROOT

        weights = torchvision.models.get_model_weights(model_name).DEFAULT
        preprocess = weights.transforms()

        model_orig = torchvision.models.get_model(model_name, weights=weights)
        model_orig = model_orig.to(device)

        imagenet12_ds = torchvision.datasets.ImageNet(
            imagenet12_root, split='val',
            transform=preprocess,
        )
        imagenet12_dl = torch.utils.data.DataLoader(
            imagenet12_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            # pin_memory=True,
            persistent_workers=False,
        )

        torch_eval_curr = functools.partial(
            torch_eval_cnn,

            testloader=imagenet12_dl,
            device=device,
        )
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if dataset_name is None:
            dataset_name = 'squad'

        if model_name == 'llama-3.2-1b':
            model_id = "meta-llama/Llama-3.2-1B"

        model_orig = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        torch_eval_curr = functools.partial(
            eval_on_sqad_ds,
            tok=tokenizer,
            stop_after=100,
            ret_f1=True,
        )

    sd_orig = copy.deepcopy(model_orig.to('cpu').state_dict()) 
            
    results = []
    
    model = copy.deepcopy(model_orig.to('cpu')).to(device)
    model.load_state_dict(sd_orig)
    acc = torch_eval_curr(model)

    res_orig = {
        'model_name': model_name,
        'accuracy': acc,
        'time': 0,
        'dataset': dataset_name,
        'method': 'original',
        'method_kwargs': {},
    }

    print(f"\tOriginal accuracy: {res_orig['accuracy']}")
    results.append(res_orig)

    for i in range(n_repeats):
        for eps in epsilons:
            model = copy.deepcopy(model_orig.to('cpu')).to(device)
            model.load_state_dict(sd_orig)

            time_start = time.time()
            model_noise = noise_model(model, eps=eps)
            time_end = time.time()
            time_diff = time_end - time_start

            acc_noise = torch_eval_curr(model_noise)

            print(f"\tNoise accuracy ({i}) (eps={eps}): {acc_noise}")

            res_noise = {
                'model_name': model_name,
                'accuracy': acc_noise,
                'time': time_diff,
                'dataset': dataset_name,
                'method': 'noise',
                'method_kwargs': {'eps': eps},
            }
            results.append(res_noise)
    
    for i in range(n_repeats):
        for prune_amount in prune_amounts:
            model = copy.deepcopy(model_orig.to('cpu')).to(device)
            model.load_state_dict(sd_orig)

            time_start = time.time()
            model_pruned = copy.deepcopy(model)
            prune_model(model_pruned, amount=prune_amount)
            time_end = time.time()
            time_diff = time_end - time_start

            acc_pruned = torch_eval_curr(model_pruned)
            print(f"\tPruned accuracy ({i}) (amount={prune_amount}): {acc_pruned}")

            res_pruned = {
                'model_name': model_name,
                'accuracy': acc_pruned,
                'time': time_diff,
                'dataset': dataset_name,
                'method': 'prune',
                'method_kwargs': {'amount': prune_amount},
            }
            results.append(res_pruned)

    for i in range(n_repeats):
        model = copy.deepcopy(model_orig.to('cpu')).to(device)
        model.load_state_dict(sd_orig)
        
        res_neuperm = neuperm_exp(model, model_name, torch_eval_f=torch_eval_curr, device=device, dataset_name=dataset_name)
        print(f"\tNeuPerm accuracy ({i}): {res_neuperm['accuracy']}")
        results.append(res_neuperm)
    
    df = pd.DataFrame(results)
    result_path = f"{RESULTS_DIR}/{model_name}_{dataset_name}.csv"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    df.to_csv(result_path, index=False)
    print(f"Results saved to {result_path}")

    return df

if __name__ == "__main__":
    model_names = [
        'densenet121',
        'resnet50',
        'resnet101',
        'vgg11',

        'llama-3.2-1b',
    ]

    prune_amounts = [
        0.01,
        0.05,
    ]
    epsilons = [
        1e-4,
        1e-3,
        1e-2,
        1e-1,
    ]
    n_repeats = 10

    device:Literal['cuda', 'cpu'] = 'cuda'
    batch_size: int = 512
    
    for model_name in model_names:
        print(f"Evaluating {model_name}")
        df = single_exp(
            model_name,
            device=device,
            batch_size=batch_size,
            prune_amounts=prune_amounts,
            epsilons=epsilons,

            n_repeats=n_repeats,
        )
        # print(f'df:\n{df}')