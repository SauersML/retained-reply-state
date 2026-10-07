"""The circuit behind the hidden-word effect in Goodfire's 4-layer model, through VPD subcomponents.

1. Specificity: each subcomponent deleted from the weights (W - u v^T at every position).  Reported: the effect with the
   later tokens' cache kept (the circuit), the effect with the word visible to the cue (direct copying), and the KL from
   the unedited model's next-token distributions on held-out Pile text (general damage), in nats per token.
2. What #2103 responds to: its activity v.x on held-out Pile text (tokens with the highest mean), at the hidden word
   against other positions, and the correlation over the 50 words between its activity at the word and the word's
   effect with the later tokens' cache kept.
3. Edges: deleting an upstream subcomponent, the change of a downstream subcomponent's activity where it acts (mean
   over templates and words, relative to its unedited mean |activity| there).
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

SUBS = {"L0 MLP #53": ("h.0.mlp.c_fc", 53, "word"), "L0 MLP out #1098": ("h.0.mlp.down_proj", 1098, "word"),
        "L1 MLP #2103": ("h.1.mlp.c_fc", 2103, "word"), "L2 key #167": ("h.2.attn.k_proj", 167, "word"),
        "L2 key #224": ("h.2.attn.k_proj", 224, "word"), "L2 query #436": ("h.2.attn.q_proj", 436, "middle"),
        "L2 out #735": ("h.2.attn.o_proj", 735, "middle"), "L3 key #145": ("h.3.attn.k_proj", 145, "middle"),
        "L3 query #334": ("h.3.attn.q_proj", 334, "cue"), "L3 out #806": ("h.3.attn.o_proj", 806, "cue")}
EDGES = [("L0 MLP #53", "L1 MLP #2103"), ("L0 MLP out #1098", "L1 MLP #2103"), ("L1 MLP #2103", "L2 key #167"),
         ("L1 MLP #2103", "L2 key #224"), ("L2 key #224", "L2 out #735"), ("L2 key #167", "L2 out #735"),
         ("L1 MLP #2103", "L2 out #735"), ("L2 out #735", "L3 key #145"), ("L1 MLP #2103", "L3 key #145")]


def main():
    torch.set_grad_enabled(False)
    tok = Tokenizer.from_file(str(vm.TARGET_DIR / "tokenizer.json"))
    ids_of = lambda s: tok.encode(s).ids
    animal_ids = torch.tensor([ids_of(" " + w)[0] for w in ANIMALS])
    model = vm.load_target("cpu")
    raw = torch.load(str(vm.VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    UV = {k: (raw[f"_components.{s.replace('.', '-')}.U"][c].float(), raw[f"_components.{s.replace('.', '-')}.V"][:, c].float())
          for k, (s, c, _) in SUBS.items()}
    del raw
    na = len(ANIMALS)
    templates = []
    for frame, middle in itertools.product(FRAMES, MIDDLES):
        pre, mid, cue = ids_of(frame), ids_of(middle), ids_of(" " + frame)
        x, m0, c0 = len(pre), len(pre) + 1 + 0, len(pre) + 1 + len(mid)
        seq = torch.tensor([pre + [int(i)] + mid + cue for i in animal_ids])
        pos = {"word": [x], "middle": list(range(m0, c0)), "cue": list(range(c0, seq.shape[1]))}
        templates.append((seq, masks(seq.shape[1], x, c0, "cpu"), pos))
    chosen = np.tile(np.arange(na), len(templates))
    pile = vm.val_tokens(16, 256, 2048)

    hooks = []
    def delete(keys):
        for k in keys:
            site = model.site(SUBS[k][0])
            u, v = UV[k]
            hooks.append(site.register_forward_hook(lambda m, inp, out, u=u, v=v: out - (inp[0] @ v)[..., None] * u))
    def clear():
        while hooks:
            hooks.pop().remove()

    acts = {}
    def record(keys):
        for k in keys:
            site = model.site(SUBS[k][0])
            v = UV[k][1]
            hooks.append(site.register_forward_hook(lambda m, inp, out, k=k, v=v: acts.setdefault(k, []).append((inp[0] @ v).detach())))

    def run(arm):
        """Readout [templates * animals, animals] in one condition, and the recorded activities."""
        out = []
        for seq, mk, pos in templates:
            MASK[0] = mk[arm][None, None].expand(len(seq), 1, *mk[arm].shape)
            out.append(torch.log_softmax(model(seq)[:, -1].float(), -1)[:, animal_ids].numpy())
            MASK[0] = None
        return np.concatenate(out)

    def pile_logp():
        return torch.log_softmax(model(pile).float(), -1)

    res = {"delete": {}}
    for k in (SUBS if "--delete" in sys.argv else []):
        delete([k])
        r = {arm: float(raises(run(arm), chosen).mean()) for arm in ("retained", "visible")}
        lp = pile_logp()
        clear()
        kl = float((base_lp.exp() * (base_lp - lp)).sum(-1).mean())
        r["pile_kl"] = kl
        res["delete"][k] = r
        print(f"delete {k:18s}: cache kept {r['retained']:+.3f} ({100 * (1 - r['retained'] / base['retained']):+.0f}% removed), "
              f"word visible {r['visible']:+.3f} ({100 * (1 - r['visible'] / base['visible']):+.0f}%), Pile KL {kl:.4f} nats/token", flush=True)

    # 2. what #2103 responds to
    k = "L1 MLP #2103"
    acts.clear(); record([k])
    pile64 = vm.val_tokens(64, 512, 1024)
    for b in range(0, 64, 4):
        model.hidden(pile64[b:b + 4])
    clear()
    a = torch.cat(acts[k]).reshape(-1).numpy()
    toks = pile64.reshape(-1).numpy()
    sign = np.sign(a.mean()) or 1
    a = a * sign
    uniq, inv, cnt = np.unique(toks, return_inverse=True, return_counts=True)
    mean = np.bincount(inv, weights=a) / cnt
    top = [j for j in np.argsort(np.where(cnt >= 15, mean, -np.inf))[::-1][:25]]
    print(f"{k}: highest mean activity on Pile:", [(tok.decode([int(uniq[j])]), round(float(mean[j] / a.std()), 1)) for j in top])
    acts.clear(); record([k])
    run("retained"); clear()
    word_act = np.concatenate([acts[k][i][:, pos["word"][0]].numpy() for i, (_, _, pos) in enumerate(templates)]) * sign
    rest = np.concatenate([np.delete(acts[k][i].numpy(), pos["word"][0], 1).ravel() for i, (_, _, pos) in enumerate(templates)]) * sign
    print(f"{k}: at the hidden word {word_act.mean() / a.std():+.2f} sd, elsewhere {rest.mean() / a.std():+.2f} sd")
    L = run("retained")
    r = raises(L, chosen)
    per_word_effect = np.array([r[chosen == w].mean() for w in range(na)])
    per_word_act = np.array([word_act[chosen == w].mean() for w in range(na)])
    corr = float(np.corrcoef(per_word_act, per_word_effect)[0, 1])
    print(f"{k}: correlation over the 50 words of its activity at the word with the word's effect: {corr:+.2f}")
    res["2103"] = {"top_tokens": [tok.decode([int(uniq[j])]) for j in top], "word_sd": float(word_act.mean() / a.std()),
                   "elsewhere_sd": float(rest.mean() / a.std()), "corr_act_effect": corr,
                   "per_word": {ANIMALS[w]: [float(per_word_act[w]), float(per_word_effect[w])] for w in range(na)}}

    # 3. edges: deleting upstream changes downstream activity where it acts
    res["edges"] = {}
    def downstream_activity(dk, arm="retained"):
        acts.clear(); record([dk]); run(arm); clear()
        where = SUBS[dk][2]
        return np.concatenate([acts[dk][i][:, pos[where]].mean(1).numpy() for i, (_, _, pos) in enumerate(templates)])
    base_act = {dk: downstream_activity(dk) for dk in {d for _, d in EDGES}}
    for up, dk in EDGES:
        delete([up])
        after = downstream_activity(dk)
        clear()
        b = base_act[dk]
        change = float((after - b).mean() / np.abs(b).mean())
        res["edges"][f"{up} -> {dk}"] = change
        print(f"edge {up:18s} -> {dk:14s}: downstream activity changes by {100 * change:+.0f}% of its mean |activity|", flush=True)
    json.dump(res, open(os.path.join(RESULTS, "circuit4l.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
