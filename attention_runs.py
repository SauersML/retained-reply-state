"""Attention to the reply, run by run, with the reply's cache kept from turn 1 (the introspection condition).

For every run of a hidden_choice.py result: turn 1 (prompt, the run's thinking, the reply) is computed once, the thinking
is removed and the reply's keys are re-rotated to close the gap, and then each turn-2 question is asked over that same
cache.  Saved per run, per question and per query head of the chosen layers: the attention mass from the last position
before the answer to the four reply tokens.  Two questions over the same cache give a paired comparison per run.
--question NAME=TEXT adds any turn-2 question (e.g. a document before the recall question); --questions-file takes
NAME=TEXT pairs from a JSON object.
usage: attention_runs.py RESULT.json --questions A,B --layers 21 --out OUT.npz
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hidden_choice import task, task_of
from kvtools import layers_of, make_cache, shift_keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--questions", default="A,B", help="recall wordings to ask, comma separated")
    ap.add_argument("--question", action="append", default=[], help="NAME=TEXT: a further turn-2 question")
    ap.add_argument("--questions-file", default=None, help="JSON object NAME -> TEXT of further turn-2 questions")
    ap.add_argument("--layers", default="21")
    ap.add_argument("--max-runs", type=int, default=600)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.result))
    T = task_of(d)
    tok = AutoTokenizer.from_pretrained(d["model"])
    model = AutoModelForCausalLM.from_pretrained(d["model"], dtype=getattr(torch, a.dtype), device_map=a.device,
                                                 attn_implementation="eager").eval()
    inv_freq = model.model.rotary_emb.inv_freq.detach().cpu()
    layers = [int(x) for x in a.layers.split(",")]
    chat = lambda msgs: tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                           enable_thinking=True), add_special_tokens=False)
    prompt = chat([{"role": "user", "content": T["turn1"]}])
    reply = tok.encode("I understand.<|im_end|>", add_special_tokens=False)
    P, R = len(prompt), len(reply)
    answer = tok.encode(f"<think>\n\n</think>\n\n{T['label']}:", add_special_tokens=False)
    texts = {w: task(w, d.get("items", "animals"))["recall"] for w in a.questions.split(",") if w}
    texts.update(dict(q.split("=", 1) for q in a.question))
    if a.questions_file:
        texts.update(json.load(open(a.questions_file)))
    suffixes = {}
    for w, text in texts.items():
        full = chat([{"role": "user", "content": T["turn1"]}, {"role": "assistant", "content": "I understand."},
                     {"role": "user", "content": text}])
        assert full[:P + R] == prompt + reply
        suffixes[w] = full[P + R:] + answer
    n = min(a.max_runs, len(d["chosen"]))
    out = {w: np.zeros((n, len(layers), model.config.num_attention_heads), np.float32) for w in suffixes}
    with torch.no_grad():
        clean = layers_of(model(torch.tensor([prompt], device=a.device), use_cache=True).past_key_values)
        prompt_kv = [(k.cpu(), v.cpu()) for k, v in clean]
        for i in range(n):
            think = tok.encode("<think>\n" + d["thinking"][i] + "\n</think>\n\n", add_special_tokens=False)
            lay = layers_of(model(torch.tensor([prompt + think + reply], device=a.device), use_cache=True).past_key_values)
            kept = [(shift_keys(k[:, :, -R:].cpu(), -len(think), inv_freq), v[:, :, -R:].cpu()) for k, v in lay]
            for w, suffix in suffixes.items():
                o = model(torch.tensor([suffix], device=a.device), past_key_values=make_cache([prompt_kv, kept], a.device),
                          use_cache=True, output_attentions=True)
                for j, l in enumerate(layers):
                    out[w][i, j] = o.attentions[l][0, :, -1, P:P + R].float().sum(-1).cpu().numpy()
            if i % 100 == 0:
                print(f"{i} runs", flush=True)
    np.savez_compressed(a.out, chosen=np.array(d["chosen"][:n]), layers=np.array(layers),
                        group=model.config.num_attention_heads // model.config.num_key_value_heads, **out)


if __name__ == "__main__":
    main()
