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


def load_wikitext2_test():
    """Load the raw WikiText-2 test split."""
    return load_dataset("wikitext", "wikitext-2-raw-v1", split="test")


def eval_on_wikitext2(model, tok, ds=None, max_length=1024, stride=512, ret_ppl=False):
    """WikiText-2 raw language-model perplexity (lower is better).

    Scores the full test split with a sliding window (``max_length`` tokens,
    ``stride`` step), accumulating per-token negative log-likelihoods in float32
    and returning ``exp(mean NLL)``. No chat template and no generation — this
    reads the raw next-token distribution directly.
    """
    if ds is None:
        ds = load_wikitext2_test()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    text = "\n\n".join(ds["text"])
    enc = tok(text, return_tensors="pt")
    input_ids = enc["input_ids"]
    seq_len = input_ids.size(1)

    nll_sum = torch.zeros((), dtype=torch.float32)
    n_tokens = 0
    prev_end = 0
    for begin in tqdm.tqdm(range(0, seq_len, stride)):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end  # tokens newly scored this window (avoids double counting)
        ids = input_ids[:, begin:end].to(device)
        target = ids.clone()
        target[:, :-trg_len] = -100  # ignore already-scored context

        with torch.no_grad():
            # mean CE over the (trg_len - 1) scored positions in this window
            out = model(ids, labels=target)
            n_valid = (target[:, 1:] != -100).sum().item()
            nll_sum += out.loss.float().cpu() * n_valid
            n_tokens += n_valid

        prev_end = end
        if end == seq_len:
            break

    ppl = float(torch.exp(nll_sum / n_tokens))

    if ret_ppl:
        return ppl

    return {"perplexity": ppl}