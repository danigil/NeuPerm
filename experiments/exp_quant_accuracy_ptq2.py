"""Evaluate PTQ2 per-channel accuracy on ImageNet for all models."""
import copy, functools, time, os, sys, gc
import pandas as pd
import torch, torchvision

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neu_perm.config import IMAGENET12_ROOT, RESULTS_DIR
from neu_perm.config import require_imagenet12
from neu_perm.quantization import quantize_model

def log(msg):
    print(msg, flush=True)

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

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

MODEL_NAMES = ['vgg11', 'vgg16', 'resnet50', 'resnet101', 'densenet121']
DEVICE = 'cuda'
BATCH_SIZE = 64

os.makedirs(RESULTS_DIR, exist_ok=True)

results = []

for model_name in MODEL_NAMES:
    log(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")

    weights = torchvision.models.get_model_weights(model_name).DEFAULT
    preprocess = weights.transforms()
    model_orig = torchvision.models.get_model(model_name, weights=weights)

    require_imagenet12()  # fail loudly here, not at config import
    val_dir = os.path.join(IMAGENET12_ROOT, 'val')
    imagenet_ds = torchvision.datasets.ImageFolder(val_dir, transform=preprocess)
    imagenet_dl = torch.utils.data.DataLoader(imagenet_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    eval_fn = functools.partial(torch_eval_cnn, testloader=imagenet_dl, device=DEVICE)

    # Original accuracy (for delta)
    model = copy.deepcopy(model_orig).to(DEVICE).eval()
    acc_orig = eval_fn(model)
    log(f"  Original: {acc_orig:.4f}")
    del model; cleanup()

    # PTQ2 per-channel
    t0 = time.time()
    model_q = quantize_model(model_orig, n_bits=2, per_channel=True).to(DEVICE)
    elapsed = time.time() - t0
    acc = eval_fn(model_q)
    log(f"  PTQ-2bit perchannel: {acc:.4f} (delta={acc - acc_orig:+.4f})")
    results.append({'model_name': model_name, 'method': 'ptq_2bit_perchannel',
                    'accuracy': acc, 'n_bits': 2, 'per_channel': True, 'time': elapsed})
    del model_q; cleanup()

    del model_orig; cleanup()

df = pd.DataFrame(results)
path = f"{RESULTS_DIR}/quant_accuracy_ptq2.csv"
df.to_csv(path, index=False)
log(f"\nSaved to {path}")
log("Done.")
