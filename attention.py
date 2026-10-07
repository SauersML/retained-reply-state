"""Attention of every query head from the last recall position ("Animal:") to the reply tokens, in the stripped
context (identical for every run), with the reply's tokens and the rest of the context reported separately.
Output: per query head the attention mass on each reply token, and per key/value head the mean over its query heads."""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hidden_choice import WORDINGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--wording", default="A")
    ap.add_argument("--recall-wording", default=None, help="the recall question's wording, if different from turn 1's")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    TURN1, RECALL = WORDINGS[a.wording][0], WORDINGS[a.recall_wording or a.wording][1]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, device_map=a.device,
                                                 attn_implementation="eager").eval()
    cfg = model.config
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    chat = lambda msgs: tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                           enable_thinking=True), add_special_tokens=False)
    prompt = chat([{"role": "user", "content": TURN1}])
    reply = tok.encode("I understand.<|im_end|>", add_special_tokens=False)
    P, R = len(prompt), len(reply)
    full = chat([{"role": "user", "content": TURN1}, {"role": "assistant", "content": "I understand."},
                 {"role": "user", "content": RECALL}]) + tok.encode("<think>\n\n</think>\n\nAnimal:", add_special_tokens=False)
    with torch.no_grad():
        att = model(torch.tensor([full], device=a.device), output_attentions=True).attentions
    res = {"model": a.model, "wording": a.wording, "recall_wording": a.recall_wording or a.wording, "reply_tokens": [tok.decode([t]) for t in reply], "heads": {}, "kv_heads": {}}
    for l, A in enumerate(att):
        w = A[0, :, -1, P:P + R].float().cpu()                       # [heads, reply tokens]
        for q in range(cfg.num_attention_heads):
            res["heads"][f"{l}:{q}"] = [float(x) for x in w[q]]
        for h in range(cfg.num_key_value_heads):
            res["kv_heads"][f"{l}:{h}"] = [float(x) for x in w[h * group:(h + 1) * group].mean(0)]
    json.dump(res, open(a.out, "w"), indent=1)
    top = sorted(res["kv_heads"].items(), key=lambda kv: -sum(kv[1]))[:8]
    print("most attention to the reply (kv-heads, summed over reply tokens):", [(k, round(sum(v), 3)) for k, v in top])
    for k in ("21:0", "21:5", "21:6"):
        if k in res["kv_heads"]:
            print(k, [round(x, 3) for x in res["kv_heads"][k]])


if __name__ == "__main__":
    main()
