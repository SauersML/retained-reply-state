"""Where Goodfire's 4-layer model uses its MLPs for hidden-word recall, and which attention step each MLP part feeds.

Retained condition, all 48 templates (each with every word as X once).  Recall: hidden_span.template_measures (the raise
of the hidden word's log P within its template, nats).  Attention: head 2.3 from the later tokens to the word (the copy
step); heads 3.4 and 3.5 from the last cue token to the later tokens (the read step).
  blocks      each layer's whole MLP output, then its attention output, replaced at one class of positions (word, later
              tokens, cue) by its mean over the template's 50 runs: what is left is what every word shares, so the change
              in raise is what that block's word-specific output contributes
  parts       in each block whose word-specific output carries at least --share of the raise, the --cands MLP-in (c_fc)
              and MLP-out (down_proj) subcomponents with the largest mean |x.v| there, each set to its template mean
              alone (its word-specific part removed)
  top         the --top parts by effect, also removed outright (output minus (x.v) u); their KL on held-out Pile text
              with the part removed at every position comes from suspects.py
Each stage can be split over processes: --stage blocks|parts|top --shard K --nshard N writes OUT with .STAGE.K.json;
--stage merge combines the shards; --stage combo --combo B,... adds several blocks averaged out together to OUT.
usage: mlp_routes.py --stage blocks --shard 0 --nshard 4 ... then --stage merge --out mlp_routes.json
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

CLASSES = ("word", "later", "cue")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", type=float, default=0.15, help="blocks whose mean-ablation removes this share of the raise")
    ap.add_argument("--cands", type=int, default=40, help="candidates per matrix and block")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(RESULTS, "mlp_routes.json"))
    ap.add_argument("--stage", default="blocks", choices=["blocks", "parts", "top", "merge", "combo"])
    ap.add_argument("--combo", default="", help="blocks averaged out together, e.g. 0.attn@later,0.mlp@later (stage combo)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tok = Tokenizer.from_file(str(vm.TARGET_DIR / "tokenizer.json"))
    ids_of = lambda s: tok.encode(s).ids
    animal_ids = torch.tensor([ids_of(" " + w)[0] for w in ANIMALS])
    model = vm.load_target("cpu")
    raw = torch.load(str(vm.VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    L, H, D = model.n_layer, model.n_head, model.hd
    UV = lambda s: (raw[f"_components.{s.replace('.', '-')}.U"].float(), raw[f"_components.{s.replace('.', '-')}.V"].float())
    uv = {f"h.{l}.mlp.{k}": UV(f"h.{l}.mlp.{k}") for l in range(L) for k in ("c_fc", "down_proj")}

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

    def run(block_mean=(), part_mean=(), removed=(), acts=None):
        """block_mean: (layer, "mlp"|"attn", class) outputs replaced by their template mean; part_mean: (site, idx, class)
        subcomponent activities x.v replaced by their template mean; removed: (site, idx, class) taken out; acts: dict
        collecting every MLP subcomponent's mean |x.v| per (site, class)."""
        logp, att23, att3 = [], [], []
        for seq, pos, mask, x, c0 in templates:
            B, T = seq.shape

            def lin(name, h):
                y = h @ model.site(name).W.T
                for site, idx, cls in part_mean:
                    if site == name:
                        a_ = h[:, pos[cls]] @ uv[name][1][:, idx]
                        y[:, pos[cls]] += (a_.mean(0, keepdim=True) - a_)[..., None] * uv[name][0][idx]
                for site, idx, cls in removed:
                    if site == name:
                        y[:, pos[cls]] -= (h[:, pos[cls]] @ uv[name][1][:, idx])[..., None] * uv[name][0][idx]
                if acts is not None and name in uv:
                    for cls, p in pos.items():
                        acts[(name, cls)] = acts.get((name, cls), 0) + (h[:, p] @ uv[name][1]).abs().mean((0, 1)) / len(templates)
                return y

            def mean_out(out, layer, kind):
                for l_, k_, cls in block_mean:
                    if (l_, k_) == (layer, kind):
                        p = pos[cls]
                        out[:, p] = out[:, p].mean(0, keepdim=True)
                return out

            z = model.wte[seq]
            for i in range(L):
                n = lambda k: f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}"
                h = vm.rms(z, model.norms[2 * i], model.eps)
                q = model._rope(lin(n("q_proj"), h).view(B, T, H, D).transpose(1, 2), T)
                k_ = model._rope(lin(n("k_proj"), h).view(B, T, H, D).transpose(1, 2), T)
                v = lin(n("v_proj"), h).view(B, T, H, D).transpose(1, 2)
                att = ((q @ k_.transpose(-1, -2)) / math.sqrt(D)).masked_fill(~mask, float("-inf")).softmax(-1)
                if i == 2:
                    att23.append(att[:, 3, x + 1:c0, x].mean().item())
                if i == 3:
                    att3.append(np.mean([att[:, h_, -1, x + 1:c0].sum(-1).mean().item() for h_ in (4, 5)]))
                z = z + mean_out(lin(n("o_proj"), (att @ v).transpose(1, 2).reshape(B, T, -1)), i, "attn")
                hh = vm.rms(z, model.norms[2 * i + 1], model.eps)
                z = z + mean_out(lin(n("down_proj"), vm.gelu_tanh(lin(n("c_fc"), hh))), i, "mlp")
            out = vm.rms(z, model.ln_f, model.eps)[:, -1] @ model.wte.T
            logp.append(torch.log_softmax(out, -1)[:, animal_ids].numpy())
        m = template_measures(np.concatenate(logp), chosen)
        return {"raise": m["raise"], "top1": m["top1"], "att_2_3": float(np.mean(att23)), "att_3_45": float(np.mean(att3))}

    def rel(r, base):
        return {"raise": r["raise"] / base["raise"] - 1, "att_2_3": r["att_2_3"] / base["att_2_3"] - 1,
                "att_3_45": r["att_3_45"] / base["att_3_45"] - 1}

    part_of = lambda st, k: a.out.replace(".json", f".{st}.{k}.json")
    shards = lambda st: [json.load(open(part_of(st, k))) for k in range(a.nshard) if os.path.exists(part_of(st, k))]
    if a.stage == "merge":
        out = {"unedited": None, "blocks": {}, "parts": {}, "top": {}}
        for st in ("blocks", "parts", "top"):
            for d in shards(st):
                out["unedited"] = d["unedited"]
                out[st].update(d[st])
        json.dump(out, open(a.out, "w"), indent=1)
        return
    if a.stage == "combo":
        base = run()
        spec = [(int(b.split(".")[0]), b.split(".")[1].split("@")[0], b.split("@")[1]) for b in a.combo.split(",")]
        r = rel(run(block_mean=spec), base)
        d = json.load(open(a.out))
        d.setdefault("combos", {})[a.combo] = r
        json.dump(d, open(a.out, "w"), indent=1)
        print(f"  {a.combo} together: raise {100 * r['raise']:+.1f}%", flush=True)
        return
    acts = {}
    base = run(acts=acts)
    mine = lambda items: items[a.shard::a.nshard]
    res = {"unedited": base, a.stage: {}}
    if a.stage == "blocks":
        for l, kind, cls in mine(list(itertools.product(range(L), ("mlp", "attn"), CLASSES))):
            r = rel(run(block_mean=[(l, kind, cls)]), base)
            res["blocks"][f"{l}.{kind}@{cls}"] = r
            print(f"  layer {l} {kind:4s} at {cls:5s}: raise {100 * r['raise']:+6.1f}%  copy-step attention {100 * r['att_2_3']:+6.1f}%  "
                  f"read-step attention {100 * r['att_3_45']:+6.1f}%", flush=True)
    elif a.stage == "parts":
        blocks = {k: v for d in shards("blocks") for k, v in d["blocks"].items()}
        todo = [(f"h.{int(k.split('.')[0])}.mlp.{kind}", k.split("@")[1]) for k, r in blocks.items()
                if ".mlp@" in k and r["raise"] <= -a.share for kind in ("c_fc", "down_proj")]
        items = [(site, idx, cls) for site, cls in todo
                 for idx in torch.argsort(acts[(site, cls)], descending=True)[: a.cands].tolist()]
        for site, idx, cls in mine(items):
            r = rel(run(part_mean=[(site, idx, cls)]), base)
            res["parts"][f"{site}#{idx}@{cls}"] = r
            print(f"  {site}#{idx}@{cls}: word-specific part {100 * r['raise']:+.1f}% raise", flush=True)
    else:
        parts = {k: v for d in shards("parts") for k, v in d["parts"].items()}
        for key, r in mine(sorted(parts.items(), key=lambda kv: kv[1]["raise"])[: a.top]):
            site, rest = key.split("#")
            idx, cls = int(rest.split("@")[0]), rest.split("@")[1]
            rr = rel(run(removed=[(site, idx, cls)]), base)
            res["top"][key] = {"word_specific": r, "removed": rr}
            print(f"  {key}: word-specific part {100 * r['raise']:+.1f}% raise (copy {100 * r['att_2_3']:+.1f}%, read "
                  f"{100 * r['att_3_45']:+.1f}%); removed {100 * rr['raise']:+.1f}%", flush=True)
    json.dump(res, open(part_of(a.stage, a.shard), "w"), indent=1)


if __name__ == "__main__":
    main()
