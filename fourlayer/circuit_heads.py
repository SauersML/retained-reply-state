"""The circuit behind hidden-word recall in Goodfire's 4-layer model, head by head, from measured removals.

The text is  frame + " X" + later tokens + cue ; in the condition used throughout ("retained") the cue cannot attend to
X, so X can reach the cue only through two attention routes:
  copy   a later token attends to X (some layer, some head) and carries it in its residual stream;
  read   a cue token attends to a later token (some layer, some head) and brings it to the answer.
Each route is blocked for one (layer, head) at a time by masking only those attention entries, everything else
unchanged; the recall left (discrimination over all 48 templates x 50 words) measures that head's share.  Then the
circuit alone: every copy and read entry blocked except at the chosen heads.  The same is repeated with a VPD edit
applied to the weights (W + (f - 1) u v^T per subcomponent), whose query/key/value/output subcomponents are assigned to
the head their u (or v, for o_proj) falls in.
usage: circuit_heads.py --edit sparse_refit.json:50 --out circuit_heads.json
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
    ap.add_argument("--edit", default=os.path.join(RESULTS, "sparse_refit.json") + ":50", help="PATH:K, a saved refit")
    ap.add_argument("--keep", type=int, default=2, help="heads kept per route in the circuit-alone test")
    ap.add_argument("--out", default=os.path.join(RESULTS, "circuit_heads.json"))
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tok = Tokenizer.from_file(str(vm.TARGET_DIR / "tokenizer.json"))
    ids_of = lambda s: tok.encode(s).ids
    animal_ids = torch.tensor([ids_of(" " + w)[0] for w in ANIMALS])
    model = vm.load_target("cpu")
    L, H, D = model.n_layer, model.n_head, model.hd

    templates = []
    for frame, middle in itertools.product(FRAMES, MIDDLES):
        pre, mid, cue = ids_of(frame), ids_of(middle), ids_of(" " + frame)
        seq = torch.tensor([pre + [int(i)] + mid + cue for i in animal_ids])
        x, c0 = len(pre), len(pre) + 1 + len(mid)
        templates.append((seq, x, c0))
    chosen = np.tile(np.arange(len(ANIMALS)), len(templates))

    def masks(T, x, c0, blocked):
        """Per layer [H, T, T]: causal, the cue never sees X, and the (route, layer, head) entries in `blocked` removed."""
        m = torch.ones(L, H, T, T, dtype=torch.bool).tril()
        m[:, :, c0:, x] = False
        for route, l, h in blocked:
            if route == "copy":
                m[l, h, x + 1:c0, x] = False
            else:
                m[l, h, c0:, x + 1:c0] = False
        return m

    def forward(seq, m, record=None):
        B, T = seq.shape
        z = model.wte[seq]
        for i in range(L):
            s = lambda k: model.site(f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}")
            h = vm.rms(z, model.norms[2 * i], model.eps)
            q = model._rope(s("q_proj")(h).view(B, T, H, D).transpose(1, 2), T)
            k = model._rope(s("k_proj")(h).view(B, T, H, D).transpose(1, 2), T)
            v = s("v_proj")(h).view(B, T, H, D).transpose(1, 2)
            att = ((q @ k.transpose(-1, -2)) / math.sqrt(D)).masked_fill(~m[i][None], float("-inf")).softmax(-1)
            if record is not None:
                record.append(att)
            z = z + s("o_proj")((att @ v).transpose(1, 2).reshape(B, T, -1))
            h = vm.rms(z, model.norms[2 * i + 1], model.eps)
            z = z + s("down_proj")(vm.gelu_tanh(s("c_fc")(h)))
        return vm.rms(z, model.ln_f, model.eps)[:, -1] @ model.wte.T

    def recall(blocked=()):
        out = []
        for seq, x, c0 in templates:
            out.append(torch.log_softmax(forward(seq, masks(seq.shape[1], x, c0, blocked)), -1)[:, animal_ids].numpy())
        return float(accuracy(np.concatenate(out), chosen))

    def attention_mass():
        """Per (layer, head): mean attention from later tokens to X (copy) and from the last cue token to later tokens (read)."""
        copy, read = np.zeros((L, H)), np.zeros((L, H))
        for seq, x, c0 in templates:
            rec = []
            forward(seq, masks(seq.shape[1], x, c0, ()), rec)
            for i, att in enumerate(rec):
                copy[i] += att[:, :, x + 1:c0, x].mean((0, 2)).numpy() / len(templates)
                read[i] += att[:, :, -1, x + 1:c0].sum(-1).mean(0).numpy() / len(templates)
        return copy.tolist(), read.tolist()

    def survey(name):
        base = recall()
        r = {"recall": base, "copy": {}, "read": {}, "copy_layer": {}, "read_layer": {}}
        for route in ("copy", "read"):
            for l in range(L):
                r[f"{route}_layer"][str(l)] = recall([(route, l, h) for h in range(H)])
                for h in range(H):
                    r[route][f"{l}.{h}"] = recall([(route, l, h)])
            print(f"{name}: recall {100 * base:.1f}%;  {route} blocked per layer " +
                  ", ".join(f"L{l} {100 * v:.1f}%" for l, v in r[f'{route}_layer'].items()), flush=True)
        r["copy_attention"], r["read_attention"] = attention_mass()
        top = {route: sorted(r[route], key=lambda k: r[route][k])[:a.keep] for route in ("copy", "read")}
        keep = {(route, int(k.split(".")[0]), int(k.split(".")[1])) for route in top for k in top[route]}
        everything = {(route, l, h) for route in ("copy", "read") for l in range(L) for h in range(H)}
        r["circuit"] = {"heads": top, "recall_alone": recall(sorted(everything - keep)),
                        "recall_all_routes_blocked": recall(sorted(everything))}
        print(f"{name}: circuit {top}: alone {100 * r['circuit']['recall_alone']:.1f}%, every route blocked "
              f"{100 * r['circuit']['recall_all_routes_blocked']:.1f}%", flush=True)
        return r

    out = {"heads": H, "layers": L, "unedited": survey("unedited")}
    json.dump(out, open(a.out, "w"), indent=1)

    path, k = a.edit.rsplit(":", 1)
    scales = next(o for o in json.load(open(path))["optimized"] if o.get("support_k") == int(k))["scales"]
    raw = torch.load(str(vm.VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    owner = {}
    for key, f in scales.items():
        site, i = key.split("#")
        u = raw[f"_components.{site.replace('.', '-')}.U"][int(i)].float()
        v = raw[f"_components.{site.replace('.', '-')}.V"][:, int(i)].float()
        model.site(site).W += (f - 1) * torch.outer(u, v)
        side = v if site.endswith("o_proj") else u          # the head slice the subcomponent reads from or writes to
        if site.split(".")[1] == "attn":
            share = (side.view(H, D) ** 2).sum(1) / (side ** 2).sum()
            owner[key] = {"factor": f, "head": int(share.argmax()), "share": float(share.max())}
        else:
            owner[key] = {"factor": f, "head": None, "share": None}
    out["edit"] = {"source": a.edit, "subcomponents": owner, **survey("edited")}
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
