"""X3: Re-evaluate Llama on SQuAD with fixed harness (Instruct + chat template).
Configs: original, NeuPerm, noise (eps=1e-4, 1e-3), PTQ 8-bit. Seeds: 5.
"""
import copy, os, sys, time, gc
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neu_perm.config import RESULTS_DIR
import torch.nn.utils.prune as prune_utils
from neu_perm.perm import permute_model
from neu_perm.quantization import quantize_dequantize_sd
from neu_perm.utils import load_llama_3b, load_squad_ds, eval_on_sqad_ds

STOP_AFTER = 200
SEEDS = list(range(5))
DEVICE = "cuda"
CSV_PATH = os.path.join(RESULTS_DIR, "llama_squad_x3.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

def log(m): print(m, flush=True)

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

if os.path.exists(CSV_PATH):
    results = pd.read_csv(CSV_PATH).to_dict("records")
    log(f"Loaded {len(results)} existing results")
else:
    results = []

def save():
    pd.DataFrame(results).to_csv(CSV_PATH, index=False)

def done(method, seed, kw_repr):
    return any(
        r.get("method") == method and r.get("seed") == seed and r.get("kwargs") == kw_repr
        for r in results
    )

log("Loading SQuAD...")
ds, metric = load_squad_ds()
log("Loading Llama-3.2-1B-Instruct...")
model_orig, tok = load_llama_3b()
sd_orig = copy.deepcopy(model_orig.cpu().state_dict())

def eval_f1(model):
    return eval_on_sqad_ds(model, tok=tok, stop_after=STOP_AFTER,
                           ds=ds, metric=metric, ret_f1=True)

def run_config(method, seed, sd_fn, kw_repr):
    if done(method, seed, kw_repr):
        return
    torch.manual_seed(seed)
    sd = sd_fn()
    model = copy.deepcopy(model_orig)
    model.load_state_dict(sd)
    model = model.to(DEVICE).eval()
    del sd
    t0 = time.time()
    f1 = eval_f1(model)
    elapsed = time.time() - t0
    log(f"  [{method} seed={seed} {kw_repr}] F1={f1:.4f} ({elapsed:.1f}s)")
    results.append({"method": method, "seed": seed, "kwargs": kw_repr,
                    "f1": f1, "time": elapsed})
    save()
    del model
    cleanup()

# Original baseline once
if not done("original", 0, "{}"):
    log("Evaluating original...")
    model = copy.deepcopy(model_orig).to(DEVICE).eval()
    t0 = time.time(); f1 = eval_f1(model); elapsed = time.time() - t0
    log(f"  Original F1: {f1:.4f} ({elapsed:.1f}s)")
    results.append({"method": "original", "seed": 0, "kwargs": "{}",
                    "f1": f1, "time": elapsed})
    save()
    del model; cleanup()

# NeuPerm x seeds
for seed in SEEDS:
    def sd_fn(s=seed):
        torch.manual_seed(s)
        return permute_model("llama-3.2-1b", copy.deepcopy(sd_orig), inplace=True)
    run_config("neuperm", seed, sd_fn, "{}")

# Noise eps in {1e-4, 1e-3} x seeds
for eps in (1e-4, 1e-3):
    for seed in SEEDS:
        def sd_fn(s=seed, e=eps):
            torch.manual_seed(s)
            sd = copy.deepcopy(sd_orig)
            for k, v in sd.items():
                if v.dtype.is_floating_point:
                    sd[k] = v + torch.randn_like(v) * e
            return sd
        run_config("noise", seed, sd_fn, f"{{'eps': {eps}}}")

# Noise heavy (destroyed configs)
for eps in (1e-2, 1e-1):
    for seed in SEEDS:
        def sd_fn(s=seed, e=eps):
            torch.manual_seed(s)
            sd = copy.deepcopy(sd_orig)
            for k, v in sd.items():
                if v.dtype.is_floating_point:
                    sd[k] = v + torch.randn_like(v) * e
            return sd
        run_config("noise", seed, sd_fn, f"{{'eps': {eps}}}")

# PTQ 8/4/2 x seeds
for nb in (8, 4, 2):
    for seed in SEEDS:
        def sd_fn(s=seed, n=nb):
            torch.manual_seed(s)
            return quantize_dequantize_sd(copy.deepcopy(sd_orig), n_bits=n,
                                          per_channel=True, inplace=False)
        run_config("ptq", seed, sd_fn, f"{{'n_bits': {nb}}}")

# Random unstructured pruning at 1% / 5%
for ratio in (0.01, 0.05):
    for seed in SEEDS:
        def sd_fn(s=seed, r=ratio):
            torch.manual_seed(s)
            sd = copy.deepcopy(sd_orig)
            for k, v in sd.items():
                if v.dtype.is_floating_point and v.dim() >= 2:
                    mask = (torch.rand_like(v, dtype=torch.float32) >= r).to(v.dtype)
                    sd[k] = v * mask
            return sd
        run_config("prune", seed, sd_fn, f"{{'ratio': {ratio}}}")

log("Done.")
