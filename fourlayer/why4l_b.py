"""Two hypotheses about the dials in Goodfire's 4-layer model (the later tokens' cache kept, the word not).

H1  The word-position subcomponents set how strongly each word is carried forward: over the 50 words, the activity
    (v . x at the word, mean over templates) of key #224, key #167 and MLP output #1320 is correlated with the word's
    own effect (L at its runs minus its mean over runs) and with the layer-2 attention from the later tokens to it.
H2  Query #334 holds the read back by sending the cue's layer-3 attention elsewhere: the cue's attention to each
    position class (the first token, the rest of the frame, the later tokens, the cue itself), with and without #334.
"""
import itertools
import json
import os
import sys
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "fourlayer")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hidden_span import ANIMALS, FRAMES, MIDDLES, MASK, masks, vm
from stats import raises
from tokenizers import Tokenizer

WORD_SUBS = {"key #224": ("h.2.attn.k_proj", 224), "key #167": ("h.2.attn.k_proj", 167), "MLP out #1320": ("h.1.mlp.down_proj", 1320)}


def main():
    torch.set_grad_enabled(False)
    tok = Tokenizer.from_file(str(vm.TARGET_DIR / "tokenizer.json"))
    ids_of = lambda s: tok.encode(s).ids
    animal_ids = torch.tensor([ids_of(" " + w)[0] for w in ANIMALS])
    model = vm.load_target("cpu")
    raw = torch.load(str(vm.VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    V = {k: raw[f"_components.{s.replace('.', '-')}.V"][:, c].float() for k, (s, c) in WORD_SUBS.items()}
    u334 = raw["_components.h-3-attn-q_proj.U"][334].float()
    v334 = raw["_components.h-3-attn-q_proj.V"][:, 334].float()
    del raw
    H, na = model.n_head, len(ANIMALS)
    act = {k: [] for k in WORD_SUBS}
    attn2, L = [], []
    cue_attn = {"with #334": [], "#334 deleted": []}
    for frame, middle in itertools.product(FRAMES, MIDDLES):
        pre, mid, cue = ids_of(frame), ids_of(middle), ids_of(" " + frame)
        x, c0 = len(pre), len(pre) + 1 + len(mid)
        seq = torch.tensor([pre + [int(i)] + mid + cue for i in animal_ids])
        T = seq.shape[1]
        mk = masks(T, x, c0, "cpu")["retained"]
        for name in cue_attn:
            got = {}
            hooks = []
            if name == "#334 deleted":
                hooks.append(model.site("h.3.attn.q_proj").register_forward_hook(
                    lambda m, inp, o: o - (inp[0] @ v334)[..., None] * u334))
            for k, (s, c) in WORD_SUBS.items():
                hooks.append(model.site(s).register_forward_hook(lambda m, inp, o, k=k: got.__setitem__(k, inp[0].detach())))
            for l in (2, 3):
                for kind in ("q_proj", "k_proj"):
                    hooks.append(model.site(f"h.{l}.attn.{kind}").register_forward_hook(
                        lambda m, inp, o, key=(l, kind): got.__setitem__(key, o.detach())))
            MASK[0] = mk[None, None].expand(na, 1, T, T)
            logits = model(seq)[:, -1]
            MASK[0] = None
            for h in hooks:
                h.remove()

            def pattern(l):
                q, k = got[(l, "q_proj")], got[(l, "k_proj")]
                qq = model._rope(q.view(na, T, H, model.hd).transpose(1, 2), T)
                kk = model._rope(k.view(na, T, H, model.hd).transpose(1, 2), T)
                return ((qq @ kk.transpose(-1, -2)) / np.sqrt(model.hd)).masked_fill(~mk[None, None], float("-inf")).softmax(-1)
            A3 = pattern(3)[:, :, -1]                                   # [animals, heads, positions]
            cue_attn[name].append([A3[..., 0].mean().item(), A3[..., 1:x].sum(-1).mean().item(),
                                   A3[..., x + 1:c0].sum(-1).mean().item(), A3[..., c0:].sum(-1).mean().item()])
            if name == "with #334":
                for k in WORD_SUBS:
                    act[k].append((got[k][:, x] @ V[k]).numpy())
                attn2.append(pattern(2)[:, :, x + 1:c0, x].mean((1, 2)).numpy())
                L.append(torch.log_softmax(logits.float(), -1)[:, animal_ids].numpy())
    L = np.concatenate(L)
    chosen = np.tile(np.arange(na), len(FRAMES) * len(MIDDLES))
    r = raises(L, chosen)
    effect = np.array([r[chosen == w].mean() for w in range(na)])
    a2 = np.concatenate(attn2)
    attn_word = np.array([a2[chosen == w].mean() for w in range(na)])
    res = {"H1": {}, "H2": {}}
    print(f"H1: layer-2 attention to the word vs the word's effect over 50 words: r = {np.corrcoef(attn_word, effect)[0, 1]:+.2f}")
    res["H1"]["attention_vs_effect"] = float(np.corrcoef(attn_word, effect)[0, 1])
    for k in WORD_SUBS:
        a = np.concatenate(act[k])
        per = np.array([a[chosen == w].mean() for w in range(na)])
        ce, ca = np.corrcoef(per, effect)[0, 1], np.corrcoef(per, attn_word)[0, 1]
        res["H1"][k] = {"vs_effect": float(ce), "vs_attention": float(ca), "per_word": per.tolist()}
        print(f"H1: {k:14s} activity at the word vs effect r = {ce:+.2f}, vs layer-2 attention to the word r = {ca:+.2f}")
    res["H1"]["effect_per_word"] = effect.tolist()
    names = ("first token", "rest of the frame", "later tokens", "cue")
    for name, rows in cue_attn.items():
        m = np.mean(rows, 0)
        res["H2"][name] = dict(zip(names, m.tolist()))
        print(f"H2: {name:13s} cue's layer-3 attention: " + ", ".join(f"{n} {v:.3f}" for n, v in zip(names, m)))
    json.dump(res, open(os.path.join(RESULTS, "why4l_b.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
