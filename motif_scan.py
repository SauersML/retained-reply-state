"""Scan every layer of a model, from the weights alone, for key/value heads that read the same residual directions but
write with opposite signs: pairs whose value maps' input subspaces overlap (as overlap.py) and whose copying gains
(copying.py) have opposite signs.  Ranked by overlap times the smaller of the two |copying gain| (in units of the
model's spread of gains).  usage: motif_scan.py MODEL COPYING.json OUT.json"""
import itertools
import json
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from overlap import subspace_overlap, value_basis


def main(model_name, copying, out):
    m = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cpu")
    cfg = m.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kvh = cfg.num_key_value_heads
    cp = json.load(open(copying))["kv_heads"]
    gains = np.array([v["embedding"] for v in cp.values()])
    sd = gains.std()
    rows = []
    with torch.no_grad():
        for l, layer in enumerate(m.model.layers):
            Wv = (layer.self_attn.v_proj.weight.detach().float() * layer.input_layernorm.weight.detach().float()).numpy()
            basis = {h: value_basis(Wv[h * hd:(h + 1) * hd]) for h in range(kvh)}
            for a, b in itertools.combinations(range(kvh), 2):
                ga, gb = cp[f"{l}:{a}"]["embedding"], cp[f"{l}:{b}"]["embedding"]
                if ga * gb >= 0:
                    continue
                ov = subspace_overlap(basis[a], basis[b])
                rows.append({"layer": l, "heads": [a, b], "overlap": ov, "gains": [ga, gb],
                             "score": ov * min(abs(ga), abs(gb)) / sd})
    rows.sort(key=lambda r: -r["score"])
    json.dump({"model": model_name, "pairs": rows}, open(out, "w"), indent=1)
    for r in rows[:8]:
        print(f"L{r['layer']} kv{r['heads'][0]}-kv{r['heads'][1]}: overlap {r['overlap']:.2f}, copying gains "
              f"{r['gains'][0]:+.1f} / {r['gains'][1]:+.1f}, score {r['score']:.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
