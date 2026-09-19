"""Evaluate Llama-3.2-1B on SQuAD F1 with PTQ 8/4/2 quantization.

Runs 10 repeats per config, saves results incrementally to a CSV so partial
results are available as the experiment runs.
"""
import copy, os, sys, time, gc
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neu_perm.config import RESULTS_DIR
from neu_perm.quantization import quantize_dequantize_sd, get_bn_keys_from_sd
from neu_perm.utils import load_squad_ds, eval_on_sqad_ds

from transformers import AutoTokenizer, AutoModelForCausalLM

def log(msg):
    print(msg, flush=True)

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

MODEL_ID = "meta-llama/Llama-3.2-1B"
DEVICE = 'cuda'
STOP_AFTER = 100   # SQuAD eval samples per run
N_REPEATS = 10
QUANT_BITS = [8, 4, 2]

CSV_PATH = os.path.join(RESULTS_DIR, "llm_squad_ptq.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load existing results (for resumability)
if os.path.exists(CSV_PATH):
    results = pd.read_csv(CSV_PATH).to_dict('records')
    log(f"Loaded {len(results)} existing results")
else:
    results = []

def save():
    pd.DataFrame(results).to_csv(CSV_PATH, index=False)

def done(method, n_bits, repeat):
    return any(
        r.get('method') == method
        and (pd.isna(r.get('n_bits')) if n_bits is None else r.get('n_bits') == n_bits)
        and r.get('repeat') == repeat
        for r in results
    )

log("Loading SQuAD dataset...")
ds, metric = load_squad_ds()

log(f"Loading Llama-3.2-1B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model_orig = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
sd_orig = copy.deepcopy(model_orig.cpu().state_dict())

def eval_f1(model):
    return eval_on_sqad_ds(model, tok=tokenizer, stop_after=STOP_AFTER,
                           ds=ds, metric=metric, ret_f1=True)

# Original baseline (once)
if not done('original', None, 0):
    log("\nEvaluating original...")
    model = copy.deepcopy(model_orig).to(DEVICE).eval()
    t0 = time.time()
    f1 = eval_f1(model)
    elapsed = time.time() - t0
    log(f"  Original F1: {f1:.4f} ({elapsed:.1f}s)")
    results.append({'method': 'original', 'n_bits': None, 'repeat': 0,
                    'f1': f1, 'time': elapsed})
    save()
    del model; cleanup()

# PTQ configs x 10 repeats
for n_bits in QUANT_BITS:
    for i in range(N_REPEATS):
        if done('ptq', n_bits, i):
            continue
        log(f"\nPTQ-{n_bits}bit perchannel (repeat {i})...")
        t0 = time.time()
        sd_q = quantize_dequantize_sd(sd_orig, n_bits=n_bits,
                                      per_channel=True, inplace=False)
        quant_time = time.time() - t0
        model = copy.deepcopy(model_orig)
        model.load_state_dict(sd_q)
        model = model.to(DEVICE).eval()
        del sd_q
        t0 = time.time()
        f1 = eval_f1(model)
        eval_elapsed = time.time() - t0
        log(f"  F1: {f1:.4f} (quant={quant_time:.1f}s, eval={eval_elapsed:.1f}s)")
        results.append({'method': 'ptq', 'n_bits': n_bits, 'repeat': i,
                        'f1': f1, 'time': quant_time + eval_elapsed})
        save()
        del model; cleanup()

log("\nDone.")
