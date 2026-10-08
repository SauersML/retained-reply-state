"""Learned weight edit in VPD subcomponents for hidden-word recall in Goodfire's 4-layer model.

One log-scale s_i per VPD subcomponent u_i v_i^T of the attention sites of layers 2 and 3 (with --mlp1 also layer 1's
down_proj): W -> W + sum_i (exp(s_i) - 1) u_i v_i^T at every position, one hook per site. Adam minimizes
  task + lambda * KL(unedited || edited) on training Pile windows + mu * sum_i |s_i|
where task is, in "retained", minus the hidden word's log P relative to its mean over the runs of the same template
(each template has every word once; the softmax normalizer cancels in it, so only the 50 animal logits are needed) or,
with --loss pairwise, the logistic surrogate of discrimination (softplus of minus the hidden word's margin over each
other animal, both taken relative to their means within the template).  Held-out recall: hidden_span.template_measures
(the raise of the hidden word's log P within its template, nats; within-template discrimination; top-1: the hidden word
has the highest log P of the 50 words).
lambda runs from large to small, each solution warm-starting the next; each solution is evaluated as learned, with
every |s_i| <= 0.05 set to 0 (the edit that is saved and listed), and with --top-k cut to its K largest |s_i|.
Layers 0-1 run once: layer 1's down_proj edit is added from its cached input, and only layers 2-3 run per step (layer 3
at the last cue token only).
Held out: the templates whose frame and middle were both unused in training, and edit_eval.py's Pile rows 2048-2063
(training uses rows 0-63).  The hand-picked edits of edit_sweep.json are evaluated on the same split.
With --support PATH:INDEX --k K,...: for each K, a fresh fit in which only the K largest subcomponents of that saved
edit may move (the fewest subcomponents that reach a given recall at a given cost).  With --ablate PATH:K, the saved
refit of support K is evaluated whole and with each of its subcomponents left out.
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hidden_span import ANIMALS, FRAMES, MIDDLES, MASK, RESULTS, masks, template_measures, vm
from tokenizers import Tokenizer


def frontier(points):
    """The points ({"kl", "top1", ...}) not beaten in held-out top-1 by any point of lower or equal KL."""
    best, out = -1.0, []
    for p in sorted(points, key=lambda p: (p["kl"], -p["top1"])):
        if p["top1"] > best:
            best = p["top1"]
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "optimize_edit.json"))
    ap.add_argument("--loss", default="centered", choices=["centered", "pairwise"])
    ap.add_argument("--lambdas", default="300,100,30,10,3,1")
    ap.add_argument("--mu", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--templates", type=int, default=5, help="training templates per step (x 50 words)")
    ap.add_argument("--windows", type=int, default=8, help="training Pile windows (128 tokens) per step")
    ap.add_argument("--mlp1", action="store_true")
    ap.add_argument("--hand", action="store_true", help="also evaluate the hand-picked edits of edit_sweep.json")
    ap.add_argument("--top-k", default="", help="also evaluate each solution cut to its K largest |s_i| (K,...)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--support", default="", help="PATH:INDEX: refit only the K largest subcomponents of that saved edit")
    ap.add_argument("--k", default="", help="with --support: the support sizes K to refit, comma separated")
    ap.add_argument("--ablate", default="", help="PATH:K: the saved refit with support K, evaluated whole and with each of its "
                                                 "subcomponents left out (factor set back to 1)")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(0)

    tok = Tokenizer.from_file(str(vm.TARGET_DIR / "tokenizer.json"))
    ids_of = lambda s: tok.encode(s).ids
    animal_ids = torch.tensor([ids_of(" " + w)[0] for w in ANIMALS])
    model = vm.load_target("cpu")
    H, D, eps = model.n_head, model.hd, model.eps
    wte, wa = model.wte, model.wte[animal_ids]

    raw = torch.load(str(vm.VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    names = [k[len("_components."):-2].replace("-", ".") for k in raw if k.startswith("_components.") and k.endswith(".U")]
    sites = [n for n in names if n.split(".")[1] in ("2", "3") and n.split(".")[2] == "attn"]
    if a.mlp1:
        sites.append("h.1.mlp.down_proj")
    hand = [k for k in json.load(open(os.path.join(RESULTS, "edit_sweep.json"))) if k != "unedited"] if a.hand else []
    hooked = sorted(set(sites) | {p.split("#")[0] for spec in hand for p in spec.split(",")})
    U = {s: raw[f"_components.{s.replace('.', '-')}.U"].float().clone() for s in hooked}
    V = {s: raw[f"_components.{s.replace('.', '-')}.V"].float().clone() for s in hooked}
    del raw

    fr, mi = rng.permutation(len(FRAMES)), rng.permutation(len(MIDDLES))
    split = {"train_frames": [FRAMES[i] for i in sorted(fr[:4])], "test_frames": [FRAMES[i] for i in sorted(fr[4:])],
             "train_middles": [MIDDLES[i] for i in sorted(mi[:5])], "test_middles": [MIDDLES[i] for i in sorted(mi[5:])],
             "train_pile_rows": [0, 64], "test_pile_rows": [2048, 2064]}

    def template(frame, middle):
        pre, mid, cue = ids_of(frame), ids_of(middle), ids_of(" " + frame)
        seq = torch.tensor([pre + [int(i)] + mid + cue for i in animal_ids])
        return seq, masks(seq.shape[1], len(pre), len(pre) + 1 + len(mid), "cpu")

    train = [template(f, m) for f, m in itertools.product(split["train_frames"], split["train_middles"])]
    test = [template(f, m) for f, m in itertools.product(split["test_frames"], split["test_middles"])]
    every = [template(f, m) for f, m in itertools.product(FRAMES, MIDDLES)]

    def rope(x, lo, hi):
        rot = torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1)
        return x * model.cos[lo:hi] + rot * model.sin[lo:hi]

    def attn(i, x, mask, last):
        B, T, _ = x.shape
        s = lambda k: model.site(f"h.{i}.attn.{k}")
        h = vm.rms(x, model.norms[2 * i], eps)
        n = 1 if last else T
        q = rope(s("q_proj")(h[:, T - n:]).view(B, n, H, D).transpose(1, 2), T - n, T)
        k = rope(s("k_proj")(h).view(B, T, H, D).transpose(1, 2), 0, T)
        v = s("v_proj")(h).view(B, T, H, D).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[T - n:])
        return x[:, T - n:] + s("o_proj")(y.transpose(1, 2).reshape(B, n, H * D))

    mlp_in = lambda i, x: vm.gelu_tanh(model.site(f"h.{i}.mlp.c_fc")(vm.rms(x, model.norms[2 * i + 1], eps)))
    down = lambda i: model.site(f"h.{i}.mlp.down_proj")

    def prefix(ids, mask):
        x = attn(0, model.wte[ids], mask, False)
        x = attn(1, x + down(0)(mlp_in(0, x)), mask, False)
        g = mlp_in(1, x)
        return x + down(1)(g), (g @ V["h.1.mlp.down_proj"] if a.mlp1 else None)

    logs = {s: torch.zeros(U[s].shape[0], requires_grad=True) for s in sites}
    keep = {s: torch.ones(U[s].shape[0]) for s in sites}          # with --support, the subcomponents allowed to move
    DELTA = {}

    def tail(x, gv, mask, last):
        if gv is not None:
            x = x + (gv * torch.expm1(logs["h.1.mlp.down_proj"] * keep["h.1.mlp.down_proj"])) @ U["h.1.mlp.down_proj"]
        for i in (2, 3):
            x = attn(i, x, mask, last and i == 3)
            x = x + down(i)(mlp_in(i, x))
        return vm.rms(x, model.ln_f, eps)

    with torch.no_grad():
        train_pre = [prefix(seq, mk["retained"]) for seq, mk in train]
        pile = vm.val_tokens(64, 128, 0)
        causal = torch.ones(128, 128, dtype=torch.bool).tril()
        pile_pre = prefix(pile, causal)
        pile_base = tail(*pile_pre, causal, False)
        test_pile = vm.val_tokens(16, 256, 2048)
        test_base = model.hidden(test_pile)
        seq, mk = train[0]
        MASK[0] = mk["retained"][None, None].expand(len(seq), 1, *mk["retained"].shape)
        assert torch.allclose(tail(*train_pre[0], mk["retained"], True)[:, 0], model.hidden(seq)[:, -1], atol=1e-4)
        MASK[0] = None

    for s in hooked:
        model.site(s).register_forward_hook(
            lambda m, inp, out, s=s: out if s not in DELTA else out + ((inp[0] @ V[s]) * DELTA[s]) @ U[s])

    def task_logp(tmpls, arm):
        L = []
        for seq, mk in tmpls:
            MASK[0] = mk[arm][None, None].expand(len(seq), 1, *mk[arm].shape)
            L.append(torch.log_softmax(model.hidden(seq)[:, -1] @ wte.T, -1)[:, animal_ids].numpy())
            MASK[0] = None
        return np.concatenate(L)

    def pile_kl():
        tot = 0.0
        for j in range(0, len(test_pile), 4):
            lp = torch.log_softmax(test_base[j:j + 4] @ wte.T, -1)
            lq = torch.log_softmax(model.hidden(test_pile[j:j + 4]) @ wte.T, -1)
            tot += float((lp.exp() * (lp - lq)).sum(-1).sum())
        return tot / test_pile.numel()

    def evaluate(full=False):
        with torch.no_grad():
            c = np.tile(np.arange(len(ANIMALS)), len(test))
            L = task_logp(test, "retained")
            rank = (L > L[np.arange(len(c)), c][:, None]).sum(1) + 1
            r = dict(template_measures(L, c), mean_rank=float(rank.mean()), hidden_logp=float(L[np.arange(len(c)), c].mean()),
                     mean_animal_logp=float(L.mean()), pile_kl=pile_kl())
            if full:
                Lv, Ls = task_logp(test, "visible"), task_logp(test, "stripped")
                mv, ms = template_measures(Lv, c), template_measures(Ls, c)
                r.update(visible_discrimination=mv["discrimination"], visible_raise=mv["raise"], visible_top1=mv["top1"],
                         stripped_discrimination=ms["discrimination"], stripped_max_spread=float(np.abs(Ls.reshape(len(test), 50, 50) - Ls.reshape(len(test), 50, 50)[:, :1]).max()))
                ce = np.tile(np.arange(len(ANIMALS)), len(every))
                me = template_measures(task_logp(every, "retained"), ce)
                r.update(all48_discrimination=me["discrimination"], all48_raise=me["raise"], all48_top1=me["top1"])
        return r

    def show(name, r):
        print(f"{name:72s} held-out top-1 {100 * r['top1']:.1f}%  raise {r['raise']:+.3f}  discr {100 * r['discrimination']:.1f}%  "
              f"rank {r['mean_rank']:4.1f}  Pile KL {r['pile_kl']:.4f}", flush=True)

    out = {"split": split, "args": vars(a), "sites": sites}
    out["unedited"] = evaluate(full=True)
    show("unedited", out["unedited"])
    out["hand_picked"] = {}
    for spec in hand:
        DELTA.clear()
        for part in spec.split(","):
            sub, f = part.split(":")
            site, idx = sub.split("#")
            DELTA.setdefault(site, torch.zeros(U[site].shape[0]))[int(idx)] = float(f) - 1
        out["hand_picked"][spec] = evaluate()
        show(spec, out["hand_picked"][spec])
    DELTA.clear()
    json.dump(out, open(a.out, "w"), indent=1)

    if a.ablate:                                  # which subcomponents of a saved edit carry its effect
        path, k = a.ablate.rsplit(":", 1)
        edit = next(o for o in json.load(open(path))["optimized"] if o.get("support_k") == int(k))["scales"]

        def apply(scales):
            DELTA.clear()
            for key, f in scales.items():
                site, i = key.split("#")
                DELTA.setdefault(site, torch.zeros(U[site].shape[0]))[int(i)] = f - 1

        apply(edit)
        out["ablate"] = {"source": a.ablate, "scales": edit, "whole": evaluate(), "left_out": {}}
        show(f"edit of {len(edit)} subcomponents", out["ablate"]["whole"])
        out["ablate"]["site_left_out"] = {}
        for site in sorted({key.split("#")[0] for key in edit}):     # each weight matrix's share of the edit left out
            apply({kk: f for kk, f in edit.items() if kk.split("#")[0] != site})
            out["ablate"]["site_left_out"][site] = evaluate()
            show(f"    without its {site} subcomponents", out["ablate"]["site_left_out"][site])
            json.dump(out, open(a.out, "w"), indent=1)
        for key in edit:
            apply({kk: f for kk, f in edit.items() if kk != key})
            out["ablate"]["left_out"][key] = evaluate()
            show(f"    without {key} (x{edit[key]:.2f})", out["ablate"]["left_out"][key])
            json.dump(out, open(a.out, "w"), indent=1)
        DELTA.clear()
        return

    c_batch = torch.arange(len(ANIMALS)).repeat(a.templates)
    out["optimized"] = []
    rounds = [(float(x), None) for x in a.lambdas.split(",")]
    if a.support:
        path, index = a.support.rsplit(":", 1)
        ranked = list(json.load(open(path))["optimized"][int(index)]["scales"])
        rounds = [(float(a.lambdas.split(",")[0]), int(k)) for k in a.k.split(",") if int(k) <= len(ranked)]
        out["support"] = a.support
    opt = torch.optim.Adam(list(logs.values()), lr=a.lr)
    for lam, k_support in rounds:
        if k_support is not None:                 # a fresh fit on the K largest subcomponents of the saved edit
            for s in sites:
                keep[s].zero_()
                logs[s].data.zero_()
            for key in ranked[:k_support]:
                site, i = key.split("#")
                keep[site][int(i)] = 1.0
            opt = torch.optim.Adam(list(logs.values()), lr=a.lr)
        t0 = time.time()
        for step in range(a.steps):
            for s in sites:
                if s != "h.1.mlp.down_proj":
                    DELTA[s] = torch.expm1(logs[s] * keep[s])
            z = torch.cat([tail(*train_pre[t], train[t][1]["retained"], True)[:, 0] @ wa.T
                           for t in rng.choice(len(train), a.templates, replace=False)])
            zt = z.view(a.templates, len(ANIMALS), len(ANIMALS))          # [template, run, word]
            Dz = (zt - zt.mean(1, keepdim=True)).reshape(len(z), len(ANIMALS))
            own = Dz[torch.arange(len(z)), c_batch]
            if a.loss == "centered":
                task = -own.mean()
            else:
                margin = own[:, None] - Dz
                task = (F.softplus(-margin).sum(1) - F.softplus(torch.zeros(()))).mean() / (len(ANIMALS) - 1)
            w = torch.as_tensor(rng.choice(len(pile), a.windows, replace=False))
            lq = torch.log_softmax(tail(pile_pre[0][w], None if pile_pre[1] is None else pile_pre[1][w], causal, False) @ wte.T, -1)
            with torch.no_grad():
                lp = torch.log_softmax(pile_base[w] @ wte.T, -1)
            kl = (lp.exp() * (lp - lq)).sum(-1).mean()
            l1 = sum((logs[s] * keep[s]).abs().sum() for s in sites)
            loss = task + lam * kl + a.mu * l1
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 25 == 0 or step == a.steps - 1:
                print(f"lambda {lam:g} step {step:4d}  task {task.item():+.4f}  KL {kl.item():.4f}  L1 {l1.item():.2f}  "
                      f"{time.time() - t0:.0f}s", flush=True)
        DELTA.clear()
        for s in sites:
            logs[s].data *= keep[s]
        for s in sites:
            DELTA[s] = torch.expm1(logs[s].detach())
        r_dense = evaluate()
        for s in sites:
            DELTA[s] = torch.expm1(logs[s].detach() * (logs[s].detach().abs() > 0.05))
        r = evaluate(full=True)
        allv = torch.cat([logs[s].detach() for s in sites])
        scales = {f"{s}#{i}": float(np.exp(float(logs[s].detach()[i]))) for s in sites for i in torch.nonzero(logs[s].detach().abs() > 0.05).flatten().tolist()}
        scales = dict(sorted(scales.items(), key=lambda kv: -abs(np.log(kv[1]))))
        top_k = {}
        for k in [int(x) for x in a.top_k.split(",") if x and int(x) < len(scales)]:
            DELTA.clear()
            for key, f in list(scales.items())[:k]:
                s, i = key.split("#")
                DELTA.setdefault(s, torch.zeros(U[s].shape[0]))[int(i)] = f - 1
            top_k[k] = evaluate()
            show(f"    its top {k}", top_k[k])
        out["optimized"].append({"lambda": lam, "mu": a.mu, "steps": a.steps, "support_k": k_support, "metrics": r, "metrics_unpruned": r_dense, "n_moved": len(scales),
                                 "l1": float(allv.abs().sum()), "scales": scales, "top_k": top_k})
        DELTA.clear()
        show(f"optimized {a.loss} lambda {lam:g} unpruned", r_dense)
        show(f"optimized {a.loss} lambda {lam:g} ({len(scales)} with |s| > 0.05 kept)", r)
        print(f"    visible discr {100 * r['visible_discrimination']:.1f}%  stripped {100 * r['stripped_discrimination']:.1f}%  "
              f"all-48 top-1 {100 * r['all48_top1']:.1f}%  mean animal log P {r['mean_animal_logp']:.3f}", flush=True)
        out["frontier"] = frontier([{"kl": o["metrics"]["pile_kl"], "top1": o["metrics"]["top1"], "lambda": o["lambda"],
                                     "mu": o["mu"], "loss": a.loss, "n_subcomponents": o["n_moved"]} for o in out["optimized"]])
        out["hand_picked_frontier"] = frontier([{"kl": r["pile_kl"], "top1": r["top1"], "edit": k}
                                                for k, r in out["hand_picked"].items()])
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
