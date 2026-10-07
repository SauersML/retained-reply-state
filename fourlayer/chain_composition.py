"""Does the two-step attention path copy the hidden word's identity, and which VPD subcomponents carry it?

Hypothesis: head 2.3 at the later tokens reads the word's state through its value and output maps, and a layer-3 head
at the cue reads that through its own value and output maps into the logits.  For each of the 50 words a, take the
word's actual normed state at layer 2 (mean over templates, retained condition) and push it through
  W_O[2.3] W_V[2.3]  ->  layer-3 input norm  ->  W_O[3.h] W_V[3.h]  ->  final norm, unembedding
as if each head attended fully (attention is tested separately).  M[a, b] = the logit of word b; the path copies when
the diagonal stands above the rest of its row (copy gain = mean diagonal minus mean off-diagonal, in logit units).
Each matrix is written as its VPD subcomponents, so removing one subcomponent from one matrix of the chain gives the
copy gain it carries.  Weights and states only: a hypothesis generator, tested by removals in the full model.
usage: chain_composition.py --copy 2.3 --read 3.4,3.5 --out chain_composition.json
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hidden_span import ANIMALS, FRAMES, MIDDLES, MASK, RESULTS, masks, vm
from tokenizers import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", default="2.3")
    ap.add_argument("--read", default="3.4,3.5")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(RESULTS, "chain_composition.json"))
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tok = Tokenizer.from_file(str(vm.TARGET_DIR / "tokenizer.json"))
    ids_of = lambda s: tok.encode(s).ids
    animal_ids = torch.tensor([ids_of(" " + w)[0] for w in ANIMALS])
    model = vm.load_target("cpu")
    raw = torch.load(str(vm.VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    H, D = model.n_head, model.hd
    comp = lambda site: (raw[f"_components.{site.replace('.', '-')}.U"].float(), raw[f"_components.{site.replace('.', '-')}.V"].float())

    # the words' normed inputs to layer 2's value map at their own position (retained condition, mean over templates)
    l2 = int(a.copy.split(".")[0])
    states = torch.zeros(len(ANIMALS), model.wte.shape[1])
    n = 0
    for frame, middle in itertools.product(FRAMES, MIDDLES):
        pre, mid, cue = ids_of(frame), ids_of(middle), ids_of(" " + frame)
        seq = torch.tensor([pre + [int(i)] + mid + cue for i in animal_ids])
        x, c0, T = len(pre), len(pre) + 1 + len(mid), seq.shape[1]
        got = {}
        hook = model.site(f"h.{l2}.attn.v_proj").register_forward_hook(lambda m, inp, out: got.__setitem__("h", inp[0]))
        MASK[0] = masks(T, x, c0, "cpu")["retained"][None, None].expand(len(seq), 1, T, T)
        model(seq)
        MASK[0] = None
        hook.remove()
        states += got["h"][:, x]
        n += 1
    states /= n
    Ug = model.wte[animal_ids] * model.ln_f                  # unembedding through the final norm's gain

    def head_maps(layer, h, drop=None):
        """The head's value then output map as d x d, each matrix summed from its subcomponents (one left out if asked)."""
        mats = {}
        for kind in ("v_proj", "o_proj"):
            U, V = comp(f"h.{layer}.attn.{kind}")
            keep = torch.ones(len(U))
            if drop is not None and drop[0] == kind:
                keep[drop[1]] = 0
            W = (U * keep[:, None]).T @ V.T                      # [d_out, d_in] = sum_i u_i v_i^T
            mats[kind] = W[h * D:(h + 1) * D] if kind == "v_proj" else W[:, h * D:(h + 1) * D]
        return mats["o_proj"] @ mats["v_proj"]                # d x d

    def chain(copy_drop=None, read_drop=None, read_head=None):
        l, h = (int(x) for x in a.copy.split("."))
        mid = states @ head_maps(l, h, copy_drop).T             # what the copy head writes into the later tokens
        lr, hr = (int(x) for x in read_head.split("."))
        g = model.norms[2 * lr]                                  # layer-3 input norm gain (the rms factor is common)
        mid = mid / mid.pow(2).mean(-1, keepdim=True).sqrt() * g
        out = mid @ head_maps(lr, hr, read_drop).T
        M = out @ Ug.T                                           # [word a, logit of word b]
        off = ~torch.eye(len(ANIMALS), dtype=torch.bool)
        return float(M.diagonal().mean() - M[off].mean()), M

    res = {}
    for rh in a.read.split(","):
        gain, M = chain(read_head=rh)
        top1 = float((M.argmax(1) == torch.arange(len(ANIMALS))).float().mean())
        res[rh] = {"copy_gain": gain, "own_word_first": top1, "copy": {}, "read": {}}
        print(f"chain {a.copy} -> {rh}: copy gain {gain:+.3f}, own word ranked first for {100 * top1:.0f}% of words", flush=True)
        for side, layer_head in (("copy", a.copy), ("read", rh)):
            l, h = (int(x) for x in layer_head.split("."))
            for kind in ("v_proj", "o_proj"):
                U, V = comp(f"h.{l}.attn.{kind}")
                part = V.T if kind == "o_proj" else U
                share = (part.view(len(part), H, D) ** 2).sum(-1)[:, h] / (part ** 2).sum(-1)
                cand = torch.nonzero(share > 0.05).flatten().tolist()
                losses = []
                for i in cand:
                    g2, _ = chain(**({"copy_drop": (kind, i)} if side == "copy" else {"read_drop": (kind, i)}), read_head=rh)
                    losses.append((gain - g2, i))
                losses.sort(reverse=True)
                res[rh][side][kind] = [{"idx": i, "gain_lost": float(v)} for v, i in losses[:a.top]]
                print(f"  {side} {layer_head} {kind}: carries most of the copy: "
                      + ", ".join(f"#{i} ({v:+.3f})" for v, i in losses[:6]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
