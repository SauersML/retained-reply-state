"""Weight-level account of the retained-state readout through chosen key/value heads (found by localize.py).

Heads are given as layer:kv_head pairs.  In every arm below, only the listed heads keep (part of) the retained reply
keys and values; everything else is stripped.
  arms            heads retained; values only (keys stripped); keys only (values stripped)
  head output     delta o_i^q = attention output of query head q (a head reading a listed kv-head) at the last recall
                  position, retained minus stripped.  Layers before the head see identical inputs in both, so delta
                  o_i^q is exactly what the head adds there
  direct logit    (U g / rms) W_O^q delta o_i^q for every animal's first token: the part of the effect the head writes
                  straight to the output (U unembedding, g and rms the final norm's weight and the stripped run's scale)
  copying score   weights only: C[a, b] = (U_b g) W_O^q W_V n(x_a), x_a the token embedding (or unembedding) of animal
                  a, n the layer's input norm; animal-level z of the diagonal, for every head of the model
  animal subspace delta v_i = retained minus stripped values of the head, averaged over the reply tokens; the SVD of
                  its animal means gives directions; the top k are removed from the retained values, or kept alone,
                  against k random directions.  Cross-fitted: directions from one half of the runs, readout on the other
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hidden_choice import ANIMALS, prompts
from kvtools import layers_of, recall_logp, shift_keys
from stats import animal_level, raises


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--heads", required=True, help="layer:kv_head pairs, comma separated")
    ap.add_argument("--k", default="1,2,4,8,16")
    ap.add_argument("--max-runs", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    heads = [tuple(int(x) for x in p.split(":")) for p in a.heads.split(",")]
    ks = [int(x) for x in a.k.split(",")]
    d = json.load(open(a.result))
    TURN1, RECALL = prompts(d)
    tok = AutoTokenizer.from_pretrained(d["model"])
    model = AutoModelForCausalLM.from_pretrained(d["model"], dtype=torch.bfloat16, device_map=a.device,
                                                 attn_implementation="sdpa").eval()
    cfg = model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    inv_freq = model.model.rotary_emb.inv_freq.detach().cpu()
    chat = lambda msgs: tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                           enable_thinking=True), add_special_tokens=False)
    prompt = chat([{"role": "user", "content": TURN1}])
    reply = tok.encode("I understand.<|im_end|>", add_special_tokens=False)
    P, R = len(prompt), len(reply)
    full = chat([{"role": "user", "content": TURN1}, {"role": "assistant", "content": "I understand."},
                 {"role": "user", "content": RECALL}])
    suffix = full[P + R:] + tok.encode("<think>\n\n</think>\n\nAnimal:", add_special_tokens=False)
    forms = [(c, tok.encode(f, add_special_tokens=False)) for c in ANIMALS for f in (" " + c.title(), " " + c)]
    first_tok = torch.tensor([tok.encode(" " + c.title(), add_special_tokens=False)[0] for c in ANIMALS])
    chosen = np.array([ANIMALS.index(x) for x in d["chosen"]])[: a.max_runs]
    n = len(chosen)
    rng = np.random.default_rng(0)
    res = {"model": d["model"], "runs": n, "heads": a.heads}
    save = lambda: json.dump(res, open(a.out, "w"), indent=1)

    # weights only: copying score of every query head, with embeddings and unembeddings as inputs
    g = model.model.norm.weight.detach().float().cpu()
    U = model.lm_head.weight.detach().float().cpu()[first_tok]
    Ug = U * g
    inputs = {"embedding": model.get_input_embeddings().weight.detach().float().cpu()[first_tok], "unembedding": U}
    copy = {}
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            gl = layer.input_layernorm.weight.detach().float().cpu()
            Wv = layer.self_attn.v_proj.weight.detach().float().cpu()
            Wo = layer.self_attn.o_proj.weight.detach().float().cpu()
            for name, x in inputs.items():
                X = (x / x.pow(2).mean(1, keepdim=True).sqrt() * gl) @ Wv.T
                for q in range(cfg.num_attention_heads):
                    h = q // group
                    C = (X[:, h * hd:(h + 1) * hd] @ (Ug @ Wo[:, q * hd:(q + 1) * hd]).T).numpy()
                    copy[(name, li, q)] = animal_level(C, np.arange(len(ANIMALS)), rng, 2000)[0]
    res["copying"] = {}
    for name in inputs:
        zs = {(li, q): z for (nm, li, q), z in copy.items() if nm == name}
        ranked = sorted(zs, key=zs.get, reverse=True)
        res["copying"][name] = {f"{li}:{q}": float(zs[(li, q)]) for li, q in ranked}
        print(f"copying score ({name} inputs), top query heads: "
              + ", ".join(f"L{li} q{q} (kv {q // group}) z {zs[(li, q)]:+.1f}" for li, q in ranked[:8]), flush=True)
        for l, h in heads:
            for q in range(h * group, (h + 1) * group):
                print(f"  L{l} q{q} (kv {h}): z {zs[(l, q)]:+.1f}, rank {ranked.index((l, q)) + 1} of {len(ranked)}")
    save()

    cap, armed = {}, set()
    def grab(key, x):
        if key in armed:
            armed.discard(key)
            cap[key] = x[:, -1].float().cpu()
    for l in sorted({l for l, _ in heads}):
        model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda m, args, l=l: grab(l, args[0]))
    model.model.norm.register_forward_pre_hook(lambda m, args: grab("final", args[0]))

    with torch.no_grad():
        clean = layers_of(model(torch.tensor([prompt + reply], device=a.device), use_cache=True).past_key_values)
        prompt_kv = [(k[:, :, :P].cpu(), v[:, :, :P].cpu()) for k, v in clean]
        strip_kv = [(k[:, :, P:].cpu(), v[:, :, P:].cpu()) for k, v in clean]
        retained = []
        for th in d["thinking"][:n]:
            think = tok.encode("<think>\n" + th + "\n</think>\n\n", add_special_tokens=False)
            lay = layers_of(model(torch.tensor([prompt + think + reply], device=a.device), use_cache=True).past_key_values)
            retained.append({(l, h): (shift_keys(lay[l][0][0, h, -R:].cpu(), -len(think), inv_freq), lay[l][1][0, h, -R:].cpu())
                             for l, h in heads})

        def recall(edits):
            """Readouts [len(edits), animals] with the stripped cache, where edits[b][(l, h)] = (keys, values)
            replaces head h of layer l for batch element b; also the head outputs and final stream at the last
            recall position."""
            kvs = []
            for edit in edits:
                kv = [(k.clone(), v.clone()) for k, v in strip_kv]
                for (l, h), (k, v) in edit.items():
                    kv[l][0][0, h], kv[l][1][0, h] = k, v
                kvs.append(kv)
            armed.update([l for l, _ in heads] + ["final"])
            L = recall_logp(model, prompt_kv, kvs, suffix, forms, ANIMALS, a.device)
            return L, dict(cap)

        def batched(make_edit):
            """Readouts of every run, edit of run i = make_edit(i), in batches."""
            return np.concatenate([recall([make_edit(i) for i in range(s, min(n, s + a.batch))])[0]
                                   for s in range(0, n, a.batch)])

        _, base = recall([{}])
        base = {k: v[0] for k, v in base.items()}
        rms = float(base["final"].pow(2).mean().add(cfg.rms_norm_eps).sqrt())
        res["arms"] = {}

        def report(name, L):
            z, p = animal_level(L, chosen, rng, 5000)
            res["arms"][name] = {"raise": float(raises(L, chosen).mean()), "z": float(z), "p": float(p)}
            print(f"{name:34s} raise {res['arms'][name]['raise']:+.4f}  animal-level z {z:+.2f}  p {p:.3g}", flush=True)
            save()

        # retained heads, values only, keys only; direct logit effect from the exact head outputs
        Ls = {m: np.zeros((n, len(ANIMALS))) for m in ("both", "values", "keys")}
        direct = {(l, h): np.zeros((n, len(ANIMALS))) for l, h in heads}
        strip_head = {(l, h): (strip_kv[l][0][0, h], strip_kv[l][1][0, h]) for l, h in heads}
        to_logit = {(l, q): Ug @ model.model.layers[l].self_attn.o_proj.weight.detach().float().cpu()[:, q * hd:(q + 1) * hd]
                    for l, h in heads for q in range(h * group, (h + 1) * group)}
        for s0 in range(0, n, a.batch):
            idx = range(s0, min(n, s0 + a.batch))
            Ls["both"][s0:s0 + len(idx)], out = recall([retained[i] for i in idx])
            for l, h in heads:
                for q in range(h * group, (h + 1) * group):
                    do = out[l][:, q * hd:(q + 1) * hd] - base[l][q * hd:(q + 1) * hd]
                    direct[(l, h)][s0:s0 + len(idx)] += (do @ to_logit[(l, q)].T).numpy() / rms
        Ls["values"] = batched(lambda i: {p: (strip_head[p][0], retained[i][p][1]) for p in heads})
        Ls["keys"] = batched(lambda i: {p: (retained[i][p][0], strip_head[p][1]) for p in heads})
        report("heads retained", Ls["both"])
        report("values only", Ls["values"])
        report("keys only", Ls["keys"])
        for l, h in heads:
            report(f"direct logit effect L{l} kv{h}", direct[(l, h)])

        # animal subspace of the value differences, cross-fitted over two halves of each animal's runs
        dv = {p: torch.stack([(retained[i][p][1] - strip_head[p][1]).float().mean(0) for i in range(n)]) for p in heads}
        fold = np.zeros(n, int)
        for c in np.unique(chosen):
            idx = np.flatnonzero(chosen == c)
            fold[idx[rng.permutation(len(idx))]] = (np.arange(len(idx)) + rng.integers(2)) % 2
        basis = {}
        for f in (0, 1):
            fit = fold != f
            for p in heads:
                M = torch.stack([dv[p][fit & (chosen == c)].mean(0) for c in np.unique(chosen[fit])])
                basis[(f, p)] = torch.linalg.svd((M - M.mean(0)).double(), full_matrices=False)[2].float()
        res["subspace_spectrum"] = {f"L{l} kv{h}": [float(s) for s in torch.linalg.svdvals(
            torch.stack([dv[(l, h)][chosen == c].mean(0) for c in np.unique(chosen)]).double())] for l, h in heads}
        for k in ks:
            arms = {"remove top": np.zeros((n, len(ANIMALS))), "remove random": np.zeros((n, len(ANIMALS))),
                    "keep top only": np.zeros((n, len(ANIMALS)))}
            rand = {p: torch.linalg.qr(torch.randn(hd, k, generator=torch.Generator().manual_seed(k)))[0].T for p in heads}
            def edit(i, arm):
                out = {}
                for p in heads:
                    V = rand[p] if arm == "remove random" else basis[(fold[i], p)][:k]
                    diff = (retained[i][p][1] - strip_head[p][1]).float()
                    part = diff @ V.T @ V
                    v = strip_head[p][1].float() + (part if arm == "keep top only" else diff - part)
                    out[p] = (retained[i][p][0], v.to(strip_head[p][1].dtype))
                return out
            for arm in arms:
                arms[arm] = batched(lambda i: edit(i, arm))
            for arm, L in arms.items():
                report(f"{arm} {k} direction{'s' * (k > 1)}", L)


if __name__ == "__main__":
    main()
