"""Experiment 1 (LLMs): NeuPerm does not degrade language-model performance.

Reproduces Table 4 (`tab:exp1_llm`): benchmark scores of the two instruction-
tuned LLMs under each steganography-mitigation technique — the unmodified
baseline, NeuPerm (ours), additive Gaussian noise, random pruning, and
post-training quantization (8/4/2-bit).

Models (bfloat16):
    meta-llama/Llama-3.2-1B-Instruct   (perm key ``llama-3.2-1b``)
    Qwen/Qwen2.5-1.5B-Instruct         (perm key ``qwen2.5-1.5b``)

Benchmarks:
    squad      -> SQuAD F1
    boolq      -> BoolQ accuracy (%)
    wikitext2  -> WikiText-2 perplexity (lower is better)

Every result row is written to `<RESULTS_DIR>/llm_eval.csv` as
``(model, benchmark, method, method_kwargs, seed, score)``, so Table 4 rebuilds
directly from that one CSV. Writes incrementally and resumes: rows already in
the CSV are skipped, so a re-run continues where a previous one stopped.

Configure the run in the `__main__` block at the bottom (model list, benchmark
list, mitigation grids, seeds, device, eval sample count), then:

    python experiments/exp1_llm_eval.py

The CNNs (Table 3) live in `experiments/exp1_cnn_accuracy.py`.
"""
import copy
import gc
import json
import os
from typing import Callable, Dict, List, Literal, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neu_perm.config import RESULTS_DIR
from neu_perm.perm import permute_model
from neu_perm.quantization import quantize_dequantize_sd
from neu_perm.utils import (
    eval_on_boolq_ds,
    eval_on_sqad_ds,
    eval_on_wikitext2,
    load_boolq_ds,
    load_squad_ds,
    load_wikitext2_test,
)

# The two LLMs evaluated in the paper (Table 4): HuggingFace id -> perm key.
LLM_MODELS = [
    ("meta-llama/Llama-3.2-1B-Instruct", "llama-3.2-1b"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5-1.5b"),
]

CSV_NAME = "llm_eval.csv"


def log(msg: str) -> None:
    print(msg, flush=True)


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(model_id: str, dtype: torch.dtype):
    """Load an instruction-tuned causal LM + tokenizer (eos as pad token)."""
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    return model, tok


# ---------------------------------------------------------------------------
# State-dict mitigations (each returns the modified sd; mutates its argument)
# ---------------------------------------------------------------------------


def apply_noise(sd, eps: float):
    """Add zero-mean Gaussian noise (std ``eps``) to every float tensor."""
    for k, v in sd.items():
        if v.dtype.is_floating_point:
            sd[k] = v + torch.randn_like(v) * eps
    return sd


def apply_prune(sd, ratio: float):
    """Randomly zero a fraction ``ratio`` of entries in each >=2-D float tensor."""
    for k, v in sd.items():
        if v.dtype.is_floating_point and v.dim() >= 2:
            mask = (torch.rand_like(v, dtype=torch.float32) >= ratio).to(v.dtype)
            sd[k] = v * mask
    return sd


# Transform signature: (sd) -> sd. `sd` is a fresh deepcopy of the original.
Method = Tuple[str, Dict, Callable, bool]  # (name, kwargs, transform, per_seed)


def build_methods(
    perm_key: str,
    epsilons: List[float],
    prune_ratios: List[float],
    quant_bits: List[int],
) -> List[Method]:
    """Assemble the mitigation grid for one model.

    ``base`` is deterministic (greedy decoding) so it runs at a single seed;
    every other method is seeded and repeated across all SEEDS.
    """
    methods: List[Method] = []
    methods.append(("base", {}, lambda sd: sd, False))
    methods.append(("neuperm", {}, lambda sd: permute_model(perm_key, sd, inplace=True), True))
    for eps in epsilons:
        methods.append(("noise", {"eps": eps}, lambda sd, e=eps: apply_noise(sd, e), True))
    for ratio in prune_ratios:
        methods.append(("prune", {"ratio": ratio}, lambda sd, r=ratio: apply_prune(sd, r), True))
    for nb in quant_bits:
        methods.append((
            "quantization",
            {"n_bits": nb, "per_channel": True},
            lambda sd, n=nb: quantize_dequantize_sd(sd, n_bits=n, per_channel=True, inplace=True),
            True,
        ))
    return methods


# ---------------------------------------------------------------------------
# Evaluation dispatch
# ---------------------------------------------------------------------------


def eval_score(benchmark: str, model, tok, squad, boolq, wikitext, stop_after: int) -> float:
    """Return the scalar score for `benchmark` (F1 / accuracy / perplexity)."""
    if benchmark == "squad":
        ds, metric = squad
        return eval_on_sqad_ds(model, tok=tok, stop_after=stop_after, ds=ds, metric=metric, ret_f1=True)
    if benchmark == "boolq":
        return eval_on_boolq_ds(model, tok=tok, stop_after=stop_after, ds=boolq, ret_acc=True)
    if benchmark == "wikitext2":
        # Full-split perplexity; stop_after (a sample cap for SQuAD/BoolQ) does not apply.
        return eval_on_wikitext2(model, tok=tok, ds=wikitext, ret_ppl=True)
    raise ValueError(f"unknown benchmark: {benchmark}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(
    model_ids: List[Tuple[str, str]],
    benchmarks: List[str],
    seeds: List[int],
    epsilons: List[float],
    prune_ratios: List[float],
    quant_bits: List[int],
    dtype: torch.dtype,
    device: str,
    stop_after: int,
) -> pd.DataFrame:
    """Evaluate every (model, benchmark, method, seed) cell into one CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, CSV_NAME)

    if os.path.exists(csv_path):
        results = pd.read_csv(csv_path).to_dict("records")
        log(f"Loaded {len(results)} existing rows from {csv_path}")
    else:
        results = []

    def save():
        pd.DataFrame(results).to_csv(csv_path, index=False)

    def done(model_id, benchmark, method, kw_repr, seed) -> bool:
        return any(
            r.get("model") == model_id
            and r.get("benchmark") == benchmark
            and r.get("method") == method
            and r.get("method_kwargs") == kw_repr
            and r.get("seed") == seed
            for r in results
        )

    # Load benchmark datasets once (shared across models).
    squad = load_squad_ds() if "squad" in benchmarks else None
    boolq = load_boolq_ds() if "boolq" in benchmarks else None
    wikitext = load_wikitext2_test() if "wikitext2" in benchmarks else None

    for model_id, perm_key in model_ids:
        log(f"\n=== Loading {model_id} ===")
        model_orig, tok = load_model(model_id, dtype)
        sd_orig = copy.deepcopy(model_orig.cpu().state_dict())
        methods = build_methods(perm_key, epsilons, prune_ratios, quant_bits)

        for benchmark in benchmarks:
            for method, kwargs, transform, per_seed in methods:
                kw_repr = json.dumps(kwargs, sort_keys=True)
                method_seeds = seeds if per_seed else seeds[:1]
                for seed in method_seeds:
                    if done(model_id, benchmark, method, kw_repr, seed):
                        continue
                    torch.manual_seed(seed)
                    sd = transform(copy.deepcopy(sd_orig))
                    model = copy.deepcopy(model_orig)
                    model.load_state_dict(sd)
                    model = model.to(device).eval()
                    del sd
                    score = eval_score(benchmark, model, tok, squad, boolq, wikitext, stop_after)
                    log(f"  [{model_id} {benchmark} {method} seed={seed} {kw_repr}] score={score:.4f}")
                    results.append({
                        "model": model_id,
                        "benchmark": benchmark,
                        "method": method,
                        "method_kwargs": kw_repr,
                        "seed": seed,
                        "score": score,
                    })
                    save()
                    del model
                    cleanup()

        del model_orig, sd_orig
        cleanup()

    log(f"\nDone. Results in {csv_path}")
    return pd.DataFrame(results)


if __name__ == "__main__":
    # --- run configuration ---
    model_ids = LLM_MODELS                        # (hf_id, perm_key) pairs
    benchmarks = ["squad", "boolq", "wikitext2"]  # SQuAD F1, BoolQ acc, WikiText-2 ppl
    seeds = list(range(5))                        # 5 fixed seeds
    epsilons = [1e-4, 1e-3, 1e-2, 1e-1]           # additive Gaussian noise std grid
    prune_ratios = [0.01, 0.05]                   # random pruning fractions
    quant_bits = [8, 4, 2]                        # PTQ bit-widths (per-channel)
    dtype = torch.bfloat16
    device: Literal["cuda", "cpu"] = "cuda"
    stop_after = 200                              # eval samples per (model, SQuAD/BoolQ)
    # -------------------------

    run(
        model_ids=model_ids,
        benchmarks=benchmarks,
        seeds=seeds,
        epsilons=epsilons,
        prune_ratios=prune_ratios,
        quant_bits=quant_bits,
        dtype=dtype,
        device=device,
        stop_after=stop_after,
    )
