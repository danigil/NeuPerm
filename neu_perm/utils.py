import torch
import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset
from evaluate import load as load_metric


def load_squad_ds():
    ds = load_dataset("squad")          # or "squad_v2"
    metric = load_metric("squad")       # "squad_v2" for v2

    return ds, metric

def load_llama_3b(dtype=torch.bfloat16):
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tok  = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        # device_map="auto"
    )

    return model, tok

def eval_on_sqad_ds(model, tok, stop_after=5, start_after=0, ds=None, metric=None, ret_f1=False):
    if ds is None or metric is None:
        ds, metric = load_squad_ds()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # model.reset_past_key_values()

    preds, refs = [], []

    ds_full_len = len(ds["validation"])

    if start_after is None:
        start_after = 0

    if stop_after is None:
        stop_after = ds_full_len

    interval_len = stop_after - start_after

    # if stop_after:
    ds_full_len = min(ds_full_len, interval_len)

    for i, ex in enumerate(tqdm.tqdm(ds["validation"], total=ds_full_len)):
        if i < start_after:
            continue

        if i >= stop_after:
            break

        messages = [
            {"role": "system", "content": "You are a question-answering assistant. Given a context and a question, output ONLY the shortest exact answer span from the context. No explanations."},
            {"role": "user", "content": f"Context: {ex['context']}\n\nQuestion: {ex['question']}"},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs,
                                max_new_tokens=32,
                                temperature=None,
                                do_sample=False,
                                top_p=None,
                                pad_token_id=tok.eos_token_id,
                                eos_token_id=tok.eos_token_id,
                                )
        ans = tok.decode(out[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True).strip()

        preds.append({"id": ex["id"], "prediction_text": ans})
        refs.append({"id": ex["id"],
                    "answers": ex["answers"]})          # expects dict with lists

    results = metric.compute(predictions=preds, references=refs)

    if ret_f1:
        return results["f1"]

    return results


def load_boolq_ds():
    """Load BoolQ dataset. Unlike SQuAD, no metric ships in `evaluate` for BoolQ,
    so accuracy is computed inline in eval_on_boolq_ds.
    """
    ds = load_dataset("google/boolq")
    return ds


def eval_on_boolq_ds(model, tok, ds=None, stop_after=200, start_after=0, ret_acc=False):
    """Evaluate `model` on the BoolQ validation split.

    Each example has fields: `passage` (str), `question` (str), `answer` (bool).
    Returns accuracy in [0, 100] (multiplied by 100 to match the F1 scale of
    `eval_on_sqad_ds`).
    """
    if ds is None:
        ds = load_boolq_ds()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    ds_full_len = len(ds["validation"])

    if start_after is None:
        start_after = 0

    if stop_after is None:
        stop_after = ds_full_len

    interval_len = stop_after - start_after
    ds_full_len = min(ds_full_len, interval_len)

    correct = 0
    total = 0

    for i, ex in enumerate(tqdm.tqdm(ds["validation"], total=ds_full_len)):
        if i < start_after:
            continue

        if i >= stop_after:
            break

        messages = [
            {"role": "system", "content": "You are a yes/no question answering assistant. Read the passage and answer with EXACTLY 'yes' or 'no'."},
            {"role": "user", "content": f"Passage: {ex['passage']}\n\nQuestion: {ex['question']}\nAnswer (yes or no):"},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs,
                                 max_new_tokens=4,
                                 temperature=None,
                                 do_sample=False,
                                 top_p=None,
                                 pad_token_id=tok.eos_token_id,
                                 eos_token_id=tok.eos_token_id,
                                 )
        ans = tok.decode(out[0][inputs['input_ids'].shape[1]:],
                         skip_special_tokens=True).strip().lower()

        # Classify the first whitespace-stripped token.
        first = ans.split()[0] if ans.split() else ""
        if first.startswith("yes"):
            pred = True
        elif first.startswith("no"):
            pred = False
        else:
            pred = None  # neither — counts as wrong below

        ref = bool(ex["answer"])
        if pred is not None and pred == ref:
            correct += 1
        total += 1

    accuracy = (correct / total) * 100.0 if total > 0 else 0.0

    if ret_acc:
        return accuracy

    return {"accuracy": accuracy}


def load_wikitext2_ds():
    """Load the WikiText-2 (raw) test split for perplexity evaluation.

    Unlike SQuAD/BoolQ this is a raw language-modeling corpus — we score the
    model's next-token likelihood over natural text, with no chat template and
    no generation.
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return ds


def eval_ppl(model, tok, ds=None, max_length=1024, stride=512,
             max_tokens=None, ret_ppl=False):
    """Sliding-window perplexity of `model` over WikiText-2-raw test.

    This is the most direct probe of the input->output function NeuPerm claims to
    preserve: it reads the full next-token distribution (the loss over the logits)
    rather than a downstream task decision. NO chat template, NO `model.generate`.

    Method (HF-canonical strided perplexity): concatenate the test rows, tokenize
    once, then slide a window of `max_length` tokens advancing by `stride`, masking
    all but the newly-revealed `trg_len` target tokens of each window so every
    token is scored exactly once. Negative log-likelihood is accumulated in float32
    (model weights are bf16/fp16; accumulating loss in low precision biases the
    perplexity). Returns perplexity = exp(sum_nll / n_tokens); LOWER is better.

    `max_tokens` optionally caps the number of evaluated tokens (logged loudly when
    set, so a runtime cap is never silent). Default None evaluates the full split.
    """
    if ds is None:
        ds = load_wikitext2_ds()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    text = "\n\n".join(t for t in ds["text"] if t)
    enc = tok(text, return_tensors="pt")
    input_ids_all = enc.input_ids
    seq_len = input_ids_all.size(1)

    if max_tokens is not None and max_tokens < seq_len:
        print(f"[eval_ppl] WARNING: capping perplexity eval to "
              f"{max_tokens}/{seq_len} tokens (max_tokens set)", flush=True)
        seq_len = max_tokens

    nll_sum = 0.0          # python float (float64) accumulation
    n_tokens = 0
    prev_end = 0

    for begin in tqdm.tqdm(range(0, seq_len, stride)):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end           # tokens newly scored in this window
        input_ids = input_ids_all[:, begin:end].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            out = model(input_ids, labels=target_ids)

        # out.loss is the mean NLL over the scored label positions. The causal
        # shift drops one position per sequence, so the number of tokens actually
        # contributing to the loss is (num_unmasked - batch_size).
        num_valid = int((target_ids != -100).sum().item())
        num_loss_tokens = num_valid - target_ids.size(0)
        nll_sum += float(out.loss.float()) * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end = end
        if end == seq_len:
            break

    ppl = float(torch.exp(torch.tensor(nll_sum / n_tokens)))

    if ret_ppl:
        return ppl

    return {"perplexity": ppl}