"""Single removals of named VPD subcomponents in Goodfire's 4-layer model, retained condition, all 48 templates: the
change in the hidden word's raise (hidden_span.template_measures), and the KL(unedited || removed) per token when the
subcomponent is removed at every position of held-out Pile text (rows 2048-2063), to tell recall parts from general
disruptors.  A removal subtracts (x.v) u from its matrix's output at the named position class only.
usage: suspects.py SITE#IDX@CLASS ... --out suspects.json
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
from hidden_span import ANIMALS, FRAMES, MIDDLES, RESULTS, template_measures, vm
from tokenizers import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nodes", nargs="+")
    ap.add_argument("--out", default=os.path.join(RESULTS, "suspects.json"))
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tok = Tokenizer.from_file(str(vm.TARGET_DIR / "tokenizer.json"))
    ids_of = lambda s: tok.encode(s).ids
    animal_ids = torch.tensor([ids_of(" " + w)[0] for w in ANIMALS])
    model = vm.load_target("cpu")
    raw = torch.load(str(vm.VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    L, H, D = model.n_layer, model.n_head, model.hd
    nodes = [(k.split("#")[0], int(k.split("#")[1].split("@")[0]), k.split("@")[1]) for k in a.nodes]
    UV = {s: (raw[f"_components.{s.replace('.', '-')}.U"].float(), raw[f"_components.{s.replace('.', '-')}.V"].float())
          for s in {n[0] for n in nodes}}

    templates = []
    for frame, middle in itertools.product(FRAMES, MIDDLES):
        pre, mid, cue = ids_of(frame), ids_of(middle), ids_of(" " + frame)
        seq = torch.tensor([pre + [int(i)] + mid + cue for i in animal_ids])
        x, c0, T = len(pre), len(pre) + 1 + len(mid), seq.shape[1]
        pos = {"frame": list(range(0, x)), "word": [x], "later": list(range(x + 1, c0)), "cue": list(range(c0, T))}
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        mask[c0:, x] = False
        templates.append((seq, pos, mask))
    chosen = np.tile(np.arange(len(ANIMALS)), len(templates))

    def run(removed=()):
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
        return template_measures(np.concatenate(out), chosen)

    pile = vm.val_tokens(16, 256, 2048)
    ref = [torch.log_softmax(model(pile[j:j + 4]).float(), -1) for j in range(0, len(pile), 4)]

    def kl_everywhere(site, idx):
        U, V = UV[site]
        hook = model.site(site).register_forward_hook(lambda m, inp, out, u=U[idx], v=V[:, idx]: out - (inp[0] @ v)[..., None] * u)
        tot = sum(float((r.exp() * (r - torch.log_softmax(model(pile[j:j + 4]).float(), -1))).sum(-1).sum())
                  for r, j in zip(ref, range(0, len(pile), 4)))
        hook.remove()
        return tot / pile.numel()

    base = run()
    res = {"unedited": base, "nodes": {}}
    print(f"unedited raise {base['raise']:.4f}", flush=True)
    for node, key in zip(nodes, a.nodes):
        m = run([node])
        res["nodes"][key] = dict(m, raise_change=m["raise"] - base["raise"], raise_change_share=m["raise"] / base["raise"] - 1,
                                 kl_everywhere=kl_everywhere(node[0], node[1]))
        r = res["nodes"][key]
        print(f"  {key:30s} raise {r['raise']:.4f} ({r['raise_change_share']:+.1%})  KL everywhere {r['kl_everywhere']:.4f}", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
