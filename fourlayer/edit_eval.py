"""Weight edits of VPD subcomponents in Goodfire's 4-layer model, each spec SITE#INDEX:SCALE[,...] (W -> W + (SCALE - 1)
u v^T at every position), evaluated on: the effect with the later tokens' cache kept, top-1 there, the effect with the
word visible to the cue, and the KL from the unedited model on held-out Pile text (nats per token)."""
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hidden_span import ANIMALS, FRAMES, MIDDLES, MASK, masks, vm
from stats import raises
from discrimination import accuracy
from tokenizers import Tokenizer


def main(specs, out):
    torch.set_grad_enabled(False)
    tok = Tokenizer.from_file(str(vm.TARGET_DIR / "tokenizer.json"))
    ids_of = lambda s: tok.encode(s).ids
    animal_ids = torch.tensor([ids_of(" " + w)[0] for w in ANIMALS])
    model = vm.load_target("cpu")
    raw = torch.load(str(vm.VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    templates = []
    for frame, middle in itertools.product(FRAMES, MIDDLES):
        pre, mid, cue = ids_of(frame), ids_of(middle), ids_of(" " + frame)
        seq = torch.tensor([pre + [int(i)] + mid + cue for i in animal_ids])
        templates.append((seq, masks(seq.shape[1], len(pre), len(pre) + 1 + len(mid), "cpu")))
    chosen = np.tile(np.arange(len(ANIMALS)), len(templates))
    pile = vm.val_tokens(16, 256, 2048)

    def run(arm):
        L = []
        for seq, mk in templates:
            MASK[0] = mk[arm][None, None].expand(len(seq), 1, *mk[arm].shape)
            L.append(torch.log_softmax(model(seq)[:, -1].float(), -1)[:, animal_ids].numpy())
            MASK[0] = None
        return np.concatenate(L)

    base_lp = torch.log_softmax(model(pile).float(), -1)
    res = {}
    for spec in [""] + specs:
        hooks = []
        for part in [p for p in spec.split(",") if p]:
            sub, scale = part.split(":")
            site, idx = sub.split("#")
            u = raw[f"_components.{site.replace('.', '-')}.U"][int(idx)].float()
            v = raw[f"_components.{site.replace('.', '-')}.V"][:, int(idx)].float()
            hooks.append(model.site(site).register_forward_hook(
                lambda m, inp, o, u=u, v=v, f=float(scale) - 1: o + f * (inp[0] @ v)[..., None] * u))
        Lr, Lv = run("retained"), run("visible")
        rank = (Lr > Lr[np.arange(len(chosen)), chosen][:, None]).sum(1) + 1
        lp = torch.log_softmax(model(pile).float(), -1)
        kl = float((base_lp.exp() * (base_lp - lp)).sum(-1).mean())
        for h in hooks:
            h.remove()
        r = {"cache_kept": float(raises(Lr, chosen).mean()), "discrimination": float(accuracy(Lr, chosen)), "top1": float(np.mean(rank == 1)),
             "mean_rank": float(rank.mean()), "visible": float(raises(Lv, chosen).mean()), "pile_kl": kl}
        res[spec or "unedited"] = r
        print(f"{spec or 'unedited':70s} cache kept {r['cache_kept']:+.3f}  discr {100 * r['discrimination']:.1f}%  top-1 {100 * r['top1']:4.1f}%  rank {r['mean_rank']:4.1f}  "
              f"visible {r['visible']:+.2f}  Pile KL {kl:.4f}", flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[2:], sys.argv[1])
