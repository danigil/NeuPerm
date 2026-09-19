"""Accuracy evaluation for new NeuPerm architectures: EfficientNet-B0/B4, MobileNetV2/V3-Small."""
import copy, functools, time, os, sys, gc
import pandas as pd
import torch, torchvision

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neu_perm.config import IMAGENET12_ROOT, RESULTS_DIR
from neu_perm.config import require_imagenet12
from neu_perm.perm import permute_model

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

def load_pretrained(model_name):
    """Load pretrained model, downloading weights if needed."""
    weights = torchvision.models.get_model_weights(model_name).DEFAULT
    preprocess = weights.transforms()
    # First try loading with weights directly
    try:
        model = torchvision.models.get_model(model_name, weights=weights)
    except RuntimeError:
        # Hash mismatch — load from cached file
        log(f"  Direct load failed, trying cached file...")
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

MODEL_NAMES = ['mobilenet_v2', 'mobilenet_v3_small']
DEVICE = 'cuda'
BATCH_SIZE = 64
N_PERM_REPEATS = 3

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

    # Use cached original accuracy if available, otherwise evaluate
    existing_csv = f"{RESULTS_DIR}/accuracy_{model_name}.csv"
    if os.path.exists(existing_csv):
        prev = pd.read_csv(existing_csv)
        acc_orig = prev[prev['method'] == 'original']['accuracy'].values[0]
        log(f"  Original (cached): {acc_orig:.4f}")
    else:
        model = copy.deepcopy(model_orig).to(DEVICE)
        acc_orig = eval_fn(model)
        log(f"  Original: {acc_orig:.4f}")
        del model; cleanup()
    results.append({'model_name': model_name, 'method': 'original', 'accuracy': acc_orig, 'time': 0.0})

    # NeuPerm (N repeats)
    for i in range(N_PERM_REPEATS):
        sd = copy.deepcopy(sd_orig)
        t0 = time.time()
        sd_perm = permute_model(model_name, sd, inplace=True)
        elapsed = time.time() - t0
        model = copy.deepcopy(model_orig)
        model.load_state_dict(sd_perm)
        model = model.to(DEVICE)
        del sd, sd_perm; cleanup()
        acc = eval_fn(model)
        log(f"  NeuPerm ({i}): {acc:.4f} (delta={acc - acc_orig:+.4f})")
        results.append({'model_name': model_name, 'method': 'neuperm', 'accuracy': acc, 'time': elapsed})
        del model; cleanup()

    df = pd.DataFrame(results)
    path = f"{RESULTS_DIR}/accuracy_{model_name}.csv"
    df.to_csv(path, index=False)
    log(f"  Saved to {path}")

    del model_orig, sd_orig; cleanup()

log("\nDone.")
