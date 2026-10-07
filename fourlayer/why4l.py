"""Why the dials work, measured on attention in Goodfire's 4-layer model (the later tokens' cache kept, the word not).

For each weight edit (none; query #334 deleted; MLP output #1320 deleted; key #224 x8):
  read   layer 3: the attention mass from the cue's last token to the later tokens, per head (where the word's
         information is read)
  write  layer 2: the attention mass from the later tokens to the word, per head (how much of the word they copy)
Attention is recomputed from the query and key projections with the rotary embedding and the condition's mask.
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
from tokenizers import Tokenizer

EDITS = {"none": [], "query #334 deleted": [("h.3.attn.q_proj", 334, 0.0)],
         "MLP out #1320 deleted": [("h.1.mlp.down_proj", 1320, 0.0)], "key #224 x8": [("h.2.attn.k_proj", 224, 8.0)]}


def main():
    torch.set_grad_enabled(False)
    tok = Tokenizer.from_file(str(vm.TARGET_DIR / "tokenizer.json"))
    ids_of = lambda s: tok.encode(s).ids
    animal_ids = torch.tensor([ids_of(" " + w)[0] for w in ANIMALS])
    model = vm.load_target("cpu")
    raw = torch.load(str(vm.VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    H = model.n_head
    res = {}
    for name, edits in EDITS.items():
        hooks = []
        for site, idx, scale in edits:
            u = raw[f"_components.{site.replace('.', '-')}.U"][idx].float()
            v = raw[f"_components.{site.replace('.', '-')}.V"][:, idx].float()
            hooks.append(model.site(site).register_forward_hook(
                lambda m, inp, o, u=u, v=v, f=scale - 1: o + f * (inp[0] @ v)[..., None] * u))
        read, write = [], []
        for frame, middle in itertools.product(FRAMES, MIDDLES):
            pre, mid, cue = ids_of(frame), ids_of(middle), ids_of(" " + frame)
            x, c0 = len(pre), len(pre) + 1 + len(mid)
            seq = torch.tensor([pre + [int(i)] + mid + cue for i in animal_ids])
            T = seq.shape[1]
            mk = masks(T, x, c0, "cpu")["retained"]
            got = {}
            caps = [model.site(f"h.{l}.attn.{kind}").register_forward_hook(
                        lambda m, inp, o, key=(l, kind): got.__setitem__(key, o.detach()))
                    for l in (2, 3) for kind in ("q_proj", "k_proj")]    # after the edit hooks: sees edited outputs
            MASK[0] = mk[None, None].expand(len(seq), 1, T, T)
            model(seq)
            MASK[0] = None
            for c in caps:
                c.remove()
            for l, store in ((3, read), (2, write)):
                q, k = got[(l, "q_proj")], got[(l, "k_proj")]
                B = q.shape[0]
                qq = model._rope(q.view(B, T, H, model.hd).transpose(1, 2), T)
                kk = model._rope(k.view(B, T, H, model.hd).transpose(1, 2), T)
                z = (qq @ kk.transpose(-1, -2)) / np.sqrt(model.hd)
                A = z.masked_fill(~mk[None, None], float("-inf")).softmax(-1)
                if l == 3:
                    store.append(A[:, :, -1, x + 1:c0].sum(-1).mean(0).numpy())        # cue -> later tokens
                else:
                    store.append(A[:, :, x + 1:c0, x].mean((0, 2)).numpy())            # later tokens -> word
        for h in hooks:
            h.remove()
        rd, wr = np.mean(read, 0), np.mean(write, 0)
        res[name] = {"read_layer3": rd.tolist(), "write_layer2": wr.tolist()}
        print(f"{name:24s} layer-3 cue -> later tokens per head {np.round(rd, 3)} | layer-2 later tokens -> word per head {np.round(wr, 3)}", flush=True)
    json.dump(res, open(os.path.join(RESULTS, "why4l.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
