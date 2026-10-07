"""Weights-only copying score of every attention head for the 50 animal names.

For query head q of layer l (reading key/value head h), C[a, b] = (g * U_b) . W_O^q W_V^h n_l(x_a): the logit of animal b
that the head writes, per unit of attention, when the position it attends to holds animal a's token x_a (n_l: the
layer's input norm, U: unembedding, g: final norm weight; the first token of " Animal").  The copying gain is the mean
of the diagonal of C minus the mean of its off-diagonal entries: positive for a head that raises the animal it reads,
negative for one that suppresses it.  Inputs x_a are the token embeddings and, separately, the unembeddings.
Output: one row per query head, and the mean over the query heads of each key/value head.
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hidden_choice import ANIMALS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu").eval()  # stored in bf16; cast per layer
    cfg = model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    ids = torch.tensor([tok.encode(" " + c.title(), add_special_tokens=False)[0] for c in ANIMALS])
    U = model.lm_head.weight.detach()[ids].float()
    Ug = U * model.model.norm.weight.detach().float()
    inputs = {"embedding": model.get_input_embeddings().weight.detach()[ids].float(), "unembedding": U}
    off = ~np.eye(len(ANIMALS), dtype=bool)
    res = {"model": a.model, "heads": {}}
    with torch.no_grad():
        for l, layer in enumerate(model.model.layers):
            gl = layer.input_layernorm.weight.detach().float()
            Wv = layer.self_attn.v_proj.weight.detach().float()
            Wo = layer.self_attn.o_proj.weight.detach().float()
            for q in range(cfg.num_attention_heads):
                h = q // group
                out = Ug @ Wo[:, q * hd:(q + 1) * hd]
                row = {}
                for name, x in inputs.items():
                    v = (x / x.pow(2).mean(1, keepdim=True).add(cfg.rms_norm_eps).sqrt() * gl) @ Wv[h * hd:(h + 1) * hd].T
                    C = (v @ out.T).numpy()
                    row[name] = float(np.diag(C).mean() - C[off].mean())
                res["heads"][f"{l}:{q}"] = {"kv": h, **row}
    kv = {}
    for key, r in res["heads"].items():
        l = key.split(":")[0]
        kv.setdefault(f"{l}:{r['kv']}", []).append(r)
    res["kv_heads"] = {k: {n: float(np.mean([r[n] for r in rs])) for n in inputs} for k, rs in kv.items()}
    json.dump(res, open(a.out, "w"), indent=1)
    top = sorted(res["kv_heads"].items(), key=lambda kv_: kv_[1]["embedding"])
    print(f"{a.model}: most negative kv-heads", [(k, round(v["embedding"], 2)) for k, v in top[:6]])
    print(f"{a.model}: most positive kv-heads", [(k, round(v["embedding"], 2)) for k, v in top[-6:]])


if __name__ == "__main__":
    main()
