"""Test in the full model the subcomponents that chain_composition.py finds carrying the word's identity.

For each link of the path (head 2.3's value map at the hidden word, its output map at the later tokens, heads 3.4/3.5's
value maps at the later tokens, their output maps at the cue), the K subcomponents that carry the most of the
weights-only copy are removed where that link acts (output minus (x.v) u, everything later recomputed, retained
condition, all 48 templates), and recall is compared with removing K random subcomponents of the same matrix.
usage: chain_test.py --chain chain_composition.json --out chain_test.json
"""
import argparse
import itertools
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hidden_span import ANIMALS, FRAMES, MIDDLES, RESULTS, vm
from discrimination import accuracy
from tokenizers import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default=os.path.join(RESULTS, "chain_composition.json"))
    ap.add_argument("--ks", default="5,10,20")
    ap.add_argument("--out", default=os.path.join(RESULTS, "chain_test.json"))
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(0)
    tok = Tokenizer.from_file(str(vm.TARGET_DIR / "tokenizer.json"))
    ids_of = lambda s: tok.encode(s).ids
    animal_ids = torch.tensor([ids_of(" " + w)[0] for w in ANIMALS])
    model = vm.load_target("cpu")
    raw = torch.load(str(vm.VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    L, H, D = model.n_layer, model.n_head, model.hd
    UV = {}
    for l, k in itertools.product(range(L), ("v_proj", "o_proj")):
        s = f"h.{l}.attn.{k}"
        UV[s] = (raw[f"_components.{s.replace('.', '-')}.U"].float(), raw[f"_components.{s.replace('.', '-')}.V"].float())

    templates = []
    for frame, middle in itertools.product(FRAMES, MIDDLES):
        pre, mid, cue = ids_of(frame), ids_of(middle), ids_of(" " + frame)
        seq = torch.tensor([pre + [int(i)] + mid + cue for i in animal_ids])
        x, c0, T = len(pre), len(pre) + 1 + len(mid), seq.shape[1]
        pos = {"word": [x], "later": list(range(x + 1, c0)), "cue": list(range(c0, T))}
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        mask[c0:, x] = False
        templates.append((seq, pos, mask))
    chosen = np.tile(np.arange(len(ANIMALS)), len(templates))

    def recall(removed):
        out = []
        for seq, pos, mask in templates:
            rem = {}
            for site, idx, cls in removed:
                rem.setdefault(site, []).append((idx, pos[cls]))
            B, T = seq.shape
            z = model.wte[seq]
            for i in range(L):
                n = lambda k: f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}"

                def lin(k, h):
                    y = h @ model.site(n(k)).W.T
                    for idx, p in rem.get(n(k), []):
                        U, V = UV[n(k)]
                        y[:, p] -= (h[:, p] @ V[:, idx])[..., None] * U[idx]
                    return y
                h = vm.rms(z, model.norms[2 * i], model.eps)
                q = model._rope(lin("q_proj", h).view(B, T, H, D).transpose(1, 2), T)
                k_ = model._rope(lin("k_proj", h).view(B, T, H, D).transpose(1, 2), T)
                v = lin("v_proj", h).view(B, T, H, D).transpose(1, 2)
                att = ((q @ k_.transpose(-1, -2)) / math.sqrt(D)).masked_fill(~mask, float("-inf")).softmax(-1)
                z = z + lin("o_proj", (att @ v).transpose(1, 2).reshape(B, T, -1))
                hh = vm.rms(z, model.norms[2 * i + 1], model.eps)
                z = z + lin("down_proj", vm.gelu_tanh(lin("c_fc", hh)))
            logits = vm.rms(z, model.ln_f, model.eps)[:, -1] @ model.wte.T
            out.append(torch.log_softmax(logits, -1)[:, animal_ids].numpy())
        return float(accuracy(np.concatenate(out), chosen))

    chain = json.load(open(a.chain))
    links = [("copy value at the word", "h.2.attn.v_proj", "word", [r["idx"] for r in chain["3.5"]["copy"]["v_proj"]]),
             ("copy output at the later tokens", "h.2.attn.o_proj", "later", [r["idx"] for r in chain["3.5"]["copy"]["o_proj"]]),
             ("read value at the later tokens", "h.3.attn.v_proj", "later",
              [r["idx"] for r in chain["3.5"]["read"]["v_proj"]] + [r["idx"] for r in chain["3.4"]["read"]["v_proj"]]),
             ("read output at the cue", "h.3.attn.o_proj", "cue",
              [r["idx"] for r in chain["3.5"]["read"]["o_proj"]] + [r["idx"] for r in chain["3.4"]["read"]["o_proj"]])]
    base = recall([])
    out = {"recall": base, "links": {}}
    print(f"unedited recall {100 * base:.1f}%", flush=True)
    for name, site, cls, ranked in links:
        ranked = list(dict.fromkeys(ranked))
        r = {}
        for k in [int(x) for x in a.ks.split(",")]:
            top = recall([(site, i, cls) for i in ranked[:k]])
            rand = [recall([(site, int(i), cls) for i in rng.choice(len(UV[site][0]), k, replace=False)]) for _ in range(3)]
            r[k] = {"top": top, "random": rand}
            print(f"  {name}: top {k} removed {100 * top:.1f}%, random {k}: " + ", ".join(f"{100 * x:.1f}%" for x in rand), flush=True)
        out["links"][name] = {"site": site, "position": cls, "ranked": ranked, "by_k": r}
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
