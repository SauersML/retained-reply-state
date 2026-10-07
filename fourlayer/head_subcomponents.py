"""Which VPD subcomponents make up each attention step of the recall circuit in Goodfire's 4-layer model.

For a head (layer, h) and the positions where the circuit uses it, the subcomponents of that layer's q/k/v/o matrices
that belong to the head (more than half of |u|^2, or of |v|^2 for o_proj, in the head's slice) are ranked by their mean
|activity x.v| at the positions where each matrix acts in that step:
  copy step (later tokens read the hidden word):  query at the later tokens; key and value at the word; output at the
                                                   later tokens
  read step (the cue reads the later tokens):     query at the cue; key and value at the later tokens; output at the cue
usage: head_subcomponents.py --copy 2.3 --read 3.4,3.5 --out head_subcomponents.json
"""
import argparse
import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hidden_span import ANIMALS, FRAMES, MIDDLES, MASK, RESULTS, masks, vm
from tokenizers import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", default="2.3")
    ap.add_argument("--read", default="3.4,3.5")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(RESULTS, "head_subcomponents.json"))
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tok = Tokenizer.from_file(str(vm.TARGET_DIR / "tokenizer.json"))
    ids_of = lambda s: tok.encode(s).ids
    animal_ids = torch.tensor([ids_of(" " + w)[0] for w in ANIMALS])
    model = vm.load_target("cpu")
    raw = torch.load(str(vm.VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    H, D = model.n_head, model.hd
    UV = lambda site: (raw[f"_components.{site.replace('.', '-')}.U"].float(), raw[f"_components.{site.replace('.', '-')}.V"].float())

    # mean |x.v| per subcomponent and position class, in the retained condition (the cue cannot see the word)
    sites = [f"h.{l}.attn.{k}" for l in range(4) for k in ("q_proj", "k_proj", "v_proj", "o_proj")]
    act = {(s, c): 0.0 for s in sites for c in ("word", "later", "cue")}
    n = 0
    for frame, middle in itertools.product(FRAMES, MIDDLES):
        pre, mid, cue = ids_of(frame), ids_of(middle), ids_of(" " + frame)
        seq = torch.tensor([pre + [int(i)] + mid + cue for i in animal_ids])
        x, c0, T = len(pre), len(pre) + 1 + len(mid), seq.shape[1]
        pos = {"word": [x], "later": list(range(x + 1, c0)), "cue": list(range(c0, T))}
        got = {}
        hooks = [model.site(s).register_forward_hook(lambda m, inp, out, s=s: got.__setitem__(s, inp[0])) for s in sites]
        MASK[0] = masks(T, x, c0, "cpu")["retained"][None, None].expand(len(seq), 1, T, T)
        model(seq)
        MASK[0] = None
        for h in hooks:
            h.remove()
        for s in sites:
            proj = (got[s] @ UV(s)[1]).abs()                         # [B, T, C]
            for c, p in pos.items():
                act[(s, c)] = act[(s, c)] + proj[:, p].mean((0, 1))
        n += 1
    act = {k: v / n for k, v in act.items()}

    steps = {"copy": {"q_proj": "later", "k_proj": "word", "v_proj": "word", "o_proj": "later"},
             "read": {"q_proj": "cue", "k_proj": "later", "v_proj": "later", "o_proj": "cue"}}
    out = {}
    for step, heads in (("copy", a.copy), ("read", a.read)):
        for head in heads.split(","):
            l, h = (int(x) for x in head.split("."))
            out[head] = {"step": step}
            for kind, cls in steps[step].items():
                site = f"h.{l}.attn.{kind}"
                U, V = UV(site)
                side = V.T if kind == "o_proj" else U                    # [C, d] on the head-sliced side
                share = (side.view(len(side), H, D) ** 2).sum(-1)[:, h] / (side ** 2).sum(-1)
                mine = torch.nonzero(share > 0.5).flatten()
                strength = act[(site, cls)][mine] * (side[mine].norm(dim=-1))
                order = mine[strength.argsort(descending=True)][:a.top]
                out[head][kind] = [{"idx": int(i), "share": float(share[i]), "activity": float(act[(site, cls)][i]),
                                    "strength": float(act[(site, cls)][i] * side[i].norm())} for i in order]
                print(f"head {head} ({step}) {kind} at {cls}: {len(mine)} subcomponents in the head; strongest "
                      + ", ".join(f"#{r['idx']} ({r['strength']:.2f})" for r in out[head][kind][:5]), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
