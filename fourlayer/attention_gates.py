"""Which query/key VPD subcomponents point the circuit's heads where they look, in Goodfire's 4-layer model.

Copy step: head 2.3's attention from the later tokens to the hidden word, set by layer 2's queries at the later tokens
and keys at the word.  Read step: heads 3.4 and 3.5's attention from the last cue token to the later tokens, set by
layer 3's queries at the cue and keys at the later tokens.  Candidates: the subcomponents with the largest mean
|x.v| times |u in the head's slice| where they act; each is removed there alone (output minus (x.v) u, everything
later recomputed, retained condition, all 48 templates), and the change in the head's attention and in recall is
measured.
usage: attention_gates.py --out attention_gates.json
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
    ap.add_argument("--cands", type=int, default=40, help="candidates per matrix and position")
    ap.add_argument("--out", default=os.path.join(RESULTS, "attention_gates.json"))
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tok = Tokenizer.from_file(str(vm.TARGET_DIR / "tokenizer.json"))
    ids_of = lambda s: tok.encode(s).ids
    animal_ids = torch.tensor([ids_of(" " + w)[0] for w in ANIMALS])
    model = vm.load_target("cpu")
    raw = torch.load(str(vm.VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    L, H, D = model.n_layer, model.n_head, model.hd
    UV = {}
    for l in range(L):
        for k in ("q_proj", "k_proj"):
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
        templates.append((seq, pos, mask, x, c0))
    chosen = np.tile(np.arange(len(ANIMALS)), len(templates))

    def run(removed=()):
        """Recall, head 2.3's attention later -> word, heads 3.4/3.5's attention last cue -> later tokens; and the mean
        |x.v| of every q/k subcomponent per position class (unedited runs only need it)."""
        logp, att23, att3, acts = [], [], [], {}
        for seq, pos, mask, x, c0 in templates:
            rem = {}
            for site, idx, cls in removed:
                rem.setdefault(site, []).append((idx, pos[cls]))
            B, T = seq.shape
            z = model.wte[seq]
            for i in range(L):
                n = lambda k: f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}"
                h = vm.rms(z, model.norms[2 * i], model.eps)
                proj = {}
                for k in ("q_proj", "k_proj", "v_proj"):
                    y = h @ model.site(n(k)).W.T
                    for idx, p in rem.get(n(k), []):
                        U, V = UV[n(k)]
                        y[:, p] -= (h[:, p] @ V[:, idx])[..., None] * U[idx]
                    if k != "v_proj" and not removed:
                        for cls, p in pos.items():
                            acts[(n(k), cls)] = acts.get((n(k), cls), 0) + (h[:, p] @ UV[n(k)][1]).abs().mean((0, 1)) / len(templates)
                    proj[k] = y
                q = model._rope(proj["q_proj"].view(B, T, H, D).transpose(1, 2), T)
                k_ = model._rope(proj["k_proj"].view(B, T, H, D).transpose(1, 2), T)
                v = proj["v_proj"].view(B, T, H, D).transpose(1, 2)
                att = ((q @ k_.transpose(-1, -2)) / math.sqrt(D)).masked_fill(~mask, float("-inf")).softmax(-1)
                if i == 2:
                    att23.append(att[:, 3, x + 1:c0, x].mean().item())
                if i == 3:
                    att3.append([att[:, h_, -1, x + 1:c0].sum(-1).mean().item() for h_ in (4, 5)])
                z = z + model.site(n("o_proj"))((att @ v).transpose(1, 2).reshape(B, T, -1))
                hh = vm.rms(z, model.norms[2 * i + 1], model.eps)
                z = z + model.site(n("down_proj"))(vm.gelu_tanh(model.site(n("c_fc"))(hh)))
            out = vm.rms(z, model.ln_f, model.eps)[:, -1] @ model.wte.T
            logp.append(torch.log_softmax(out, -1)[:, animal_ids].numpy())
        return float(accuracy(np.concatenate(logp), chosen)), float(np.mean(att23)), np.mean(att3, 0).tolist(), acts

    base, b23, b3, acts = run()
    print(f"unedited: recall {100 * base:.1f}%, head 2.3 later->word {b23:.3f}, heads 3.4/3.5 cue->later {b3[0]:.3f}/{b3[1]:.3f}", flush=True)
    plan = [("h.2.attn.q_proj", "later", 3), ("h.2.attn.k_proj", "word", 3),
            ("h.3.attn.q_proj", "cue", (4, 5)), ("h.3.attn.k_proj", "later", (4, 5))]
    out = {"recall": base, "att_2_3": b23, "att_3_45": b3, "gates": {}}
    for site, cls, heads in plan:
        U, V = UV[site]
        hs = heads if isinstance(heads, tuple) else (heads,)
        inhead = sum((U.view(len(U), H, D)[:, h_] ** 2).sum(-1) for h_ in hs).sqrt()
        strength = acts[(site, cls)] * inhead
        for idx in strength.argsort(descending=True)[:a.cands].tolist():
            r, a23, a3, _ = run([(site, idx, cls)])
            key = f"{site}#{idx}@{cls}"
            out["gates"][key] = {"recall": r, "att_2_3": a23, "att_3_45": a3, "strength": float(strength[idx])}
            print(f"  without {key:28s}: recall {100 * r:5.1f}%  head 2.3 {a23:.3f} ({a23 / b23 - 1:+.0%})  "
                  f"heads 3.4/3.5 {a3[0]:.3f}/{a3[1]:.3f} ({a3[0] / b3[0] - 1:+.0%}/{a3[1] / b3[1] - 1:+.0%})", flush=True)
            json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
