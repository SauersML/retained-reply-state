"""Where the 4-layer model carries the hidden word: read side and write side (templates from hidden_span.py).

Pass A: the middle tokens may attend to X, the cue may not (retained); its keys and values are recorded.
Pass B: nobody after X may attend to X (stripped).  Arms recompute pass B, but the cue's queries read the middle
tokens' keys/values from pass A in a chosen part (layer, head, middle token), as a cache would hold them:
  read layer l / read L{l}h{h} / read token +j   which part of the retained middle state the cue reads it from
  write L{l}h{h}   pass A with only head h of layer l at the middle tokens attending to X (every other head and
                   layer at the middle tokens blind to it); the cue reads all of pass A's middle state
"""
import itertools
import json
import os
import sys
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "fourlayer")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hidden_span import ANIMALS, FRAMES, MIDDLES, vm
from stats import animal_level, raises
from tokenizers import Tokenizer


def forward(model, ids, masks, override=None, cue_from=None):
    """Explicit forward through the target's sites.  masks[i]: bool [B or 1, H or 1, T, T] for layer i.
    override[i] = (k, v, sel): keys/values of another pass replace this pass's where sel [H, T] is true, and only
    query rows from cue_from on read the replaced ones.  Returns last-position logits and every layer's (k, v)."""
    B, T = ids.shape
    x = model.wte[ids]
    kv = []
    for i in range(model.n_layer):
        s = lambda k: model.site(f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}")
        h = vm.rms(x, model.norms[2 * i], model.eps)
        q, k, v = (s(n)(h).view(B, T, model.n_head, model.hd).transpose(1, 2) for n in ("q_proj", "k_proj", "v_proj"))
        q, k = model._rope(q, T), model._rope(k, T)
        kv.append((k, v))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=masks[i])
        if override and i in override:
            ko, vo, sel = override[i]
            sel = sel[None, :, :, None]
            y2 = F.scaled_dot_product_attention(q, torch.where(sel, ko, k), torch.where(sel, vo, v), attn_mask=masks[i])
            y = torch.cat([y[:, :, :cue_from], y2[:, :, cue_from:]], 2)
        x = x + s("o_proj")(y.transpose(1, 2).reshape(B, T, -1))
        h = vm.rms(x, model.norms[2 * i + 1], model.eps)
        x = x + s("down_proj")(vm.gelu_tanh(s("c_fc")(h)))
    return vm.rms(x[:, -1], model.ln_f, model.eps) @ model.wte.T, kv


def main():
    device = "cpu"
    tok = Tokenizer.from_file(str(vm.TARGET_DIR / "tokenizer.json"))
    ids_of = lambda s: tok.encode(s).ids
    animal_ids = torch.tensor([ids_of(" " + w)[0] for w in ANIMALS])
    model = vm.load_target(device)
    nl, H = model.n_layer, model.n_head
    arms = {}
    chosen = []
    with torch.no_grad():
        for frame, middle in itertools.product(FRAMES, MIDDLES):
            pre, mid, cue = ids_of(frame), ids_of(middle), ids_of(" " + frame)
            x, m0, c0 = len(pre), len(pre) + 1, len(pre) + 1 + len(mid)
            seq = torch.tensor([pre + [int(i)] + mid + cue for i in animal_ids])
            T = seq.shape[1]
            causal = torch.ones(T, T, dtype=torch.bool).tril()
            strip = causal.clone(); strip[x + 1:, x] = False
            ret = causal.clone(); ret[c0:, x] = False
            middle_pos = torch.zeros(H, T, dtype=torch.bool); middle_pos[:, m0:c0] = True
            readout = lambda logits: torch.log_softmax(logits.float(), -1)[:, animal_ids].numpy()
            add = lambda name, logits: arms.setdefault(name, []).append(readout(logits))

            _, kvA = forward(model, seq, [ret[None, None]] * nl)
            strip_masks = [strip[None, None]] * nl
            def read(sel_by_layer, kvsrc=kvA):
                return forward(model, seq, strip_masks, {l: (kvsrc[l][0], kvsrc[l][1], sel) for l, sel in sel_by_layer.items()}, c0)[0]
            add("retained (all middle state)", read({l: middle_pos for l in range(nl)}))
            for l in range(nl):
                add(f"read layer {l}", read({l: middle_pos}))
            for l in range(nl):
                for hh in range(H):
                    sel = torch.zeros(H, T, dtype=torch.bool); sel[hh, m0:c0] = True
                    add(f"read L{l}h{hh}", read({l: sel}))
            for j in range(min(len(ids_of(m)) for m in MIDDLES)):
                sel = torch.zeros(H, T, dtype=torch.bool); sel[:, m0 + j] = True
                add(f"read token +{j + 1}", read({l: sel for l in range(nl)}))
            for l in range(nl):
                for hh in range(H):
                    wm = []
                    for i in range(nl):
                        m = ret.clone()[None].repeat(H, 1, 1)
                        m[:, m0:c0, x] = False
                        if i == l:
                            m[hh, m0:c0, x] = True
                        wm.append(m[None])
                    _, kvW = forward(model, seq, wm)
                    add(f"write L{l}h{hh}", read({i: middle_pos for i in range(nl)}, kvW))
            chosen.extend(range(len(ANIMALS)))
    chosen = np.array(chosen)
    rng = np.random.default_rng(0)
    res = {}
    for name, rows in arms.items():
        L = np.concatenate(rows)
        z, p = animal_level(L, chosen, rng, 5000)
        res[name] = {"raise": float(raises(L, chosen).mean()), "z": float(z), "p": float(p)}
        print(f"{name:28s} raise {res[name]['raise']:+.4f}  z {z:+6.2f}")
    json.dump({"model": "goodfire 4L Pile (VPD target)", "arms": res}, open(os.path.join(RESULTS, "localize4l.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
