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
    model_id = "meta-llama/Llama-3.2-1B"
    tok  = AutoTokenizer.from_pretrained(model_id)
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

        prompt = (
            f"<s>{ex['context']}\n\n"
            f"Question: {ex['question']}\n\n"
            "Answer:"
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs,
                                max_new_tokens=32,
                                temperature=None,
                                do_sample=False,
                                top_p=None,
                                pad_token_id=tok.eos_token_id
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