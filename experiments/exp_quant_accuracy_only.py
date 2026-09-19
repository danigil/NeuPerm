import copy, functools, time, os, sys, gc
import pandas as pd
import torch, torchvision

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neu_perm.config import IMAGENET12_ROOT, RESULTS_DIR
from neu_perm.config import require_imagenet12
from neu_perm.perm import permute_model
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
QUANT_CONFIGS = [(8, False), (8, True), (4, False), (4, True)]
DEVICE = 'cuda'
BATCH_SIZE = 64

os.makedirs(RESULTS_DIR, exist_ok=True)

SKIP_IF_EXISTS = False

for model_name in MODEL_NAMES:
    csv_path = f"{RESULTS_DIR}/quant_accuracy_{model_name}.csv"
    if SKIP_IF_EXISTS and os.path.exists(csv_path):
        log(f"Skipping {model_name} (already exists: {csv_path})")
        continue
    log(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")

    weights = torchvision.models.get_model_weights(model_name).DEFAULT
    preprocess = weights.transforms()
    model_orig = torchvision.models.get_model(model_name, weights=weights)

    # Use ImageFolder with organized val/ directory instead of torchvision.datasets.ImageNet
    require_imagenet12()  # fail loudly here, not at config import
    val_dir = os.path.join(IMAGENET12_ROOT, 'val')
    imagenet_ds = torchvision.datasets.ImageFolder(val_dir, transform=preprocess)
    imagenet_dl = torch.utils.data.DataLoader(imagenet_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    eval_fn = functools.partial(torch_eval_cnn, testloader=imagenet_dl, device=DEVICE)
    sd_orig = copy.deepcopy(model_orig.cpu().state_dict())

    results = []

    # Original
    model = copy.deepcopy(model_orig).to(DEVICE)
    model.load_state_dict(sd_orig)
    acc_orig = eval_fn(model)
    log(f"  Original: {acc_orig:.4f}")
    results.append({'model_name': model_name, 'method': 'original', 'accuracy': acc_orig,
                    'n_bits': None, 'per_channel': None, 'time': 0.0})
    del model; cleanup()

    # Quantization configs
    for n_bits, per_channel in QUANT_CONFIGS:
        t0 = time.time()
        model_q = quantize_model(model_orig, n_bits=n_bits, per_channel=per_channel).to(DEVICE)
        elapsed = time.time() - t0
        acc = eval_fn(model_q)
        pc = "perchannel" if per_channel else "pertensor"
        log(f"  PTQ-{n_bits}bit {pc}: {acc:.4f} (delta={acc - acc_orig:+.4f})")
        results.append({'model_name': model_name, 'method': f'ptq_{n_bits}bit_{pc}',
                        'accuracy': acc, 'n_bits': n_bits, 'per_channel': per_channel,
                        'time': elapsed})
        del model_q; cleanup()

    # NeuPerm (3 repeats)
    for i in range(3):
        model = copy.deepcopy(model_orig).to(DEVICE)
        model.load_state_dict(sd_orig)
        sd = model.cpu().state_dict()
        t0 = time.time()
        sd_perm = permute_model(model_name, sd, inplace=True)
        elapsed = time.time() - t0
        model.load_state_dict(sd_perm)
        model = model.to(DEVICE)
        del sd, sd_perm; cleanup()
        acc = eval_fn(model)
        log(f"  NeuPerm ({i}): {acc:.4f} (delta={acc - acc_orig:+.4f})")
        results.append({'model_name': model_name, 'method': 'neuperm',
                        'accuracy': acc, 'n_bits': None, 'per_channel': None,
                        'time': elapsed})
        del model; cleanup()

    df = pd.DataFrame(results)
    path = f"{RESULTS_DIR}/quant_accuracy_{model_name}.csv"
    df.to_csv(path, index=False)
    log(f"  Saved to {path}")

    del model_orig, sd_orig; cleanup()

log("\nDone.")
