"""Dose-response of the retained reply state through chosen key/value heads (run on a hidden_choice.py result).

For a group of heads (layer:kv_head pairs, or "all"), recall reads the reply keys/values  S + a (R - S)  in those heads,
S stripped and R retained; every other head holds S (--base stripped) or R (--base retained).  a = 0 removes the
group's retained part, a = 1 is the retained state, a < 0 reverses it, a > 1 amplifies it.  Reported per dose: the
own-animal raise, animal-level z, top-1 accuracy (chance 1/50) and the mean rank of the chosen animal.
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
    ap.add_argument("--groups", required=True, help="groups separated by ';', each 'all' or layer:kv_head pairs")
    ap.add_argument("--base", default="stripped", choices=["stripped", "retained"])
    ap.add_argument("--doses", default="-2,-1,0,0.5,1,2,4,8")
    ap.add_argument("--max-runs", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    doses = [float(x) for x in a.doses.split(",")]
    d = json.load(open(a.result))
    TURN1, RECALL = prompts(d)
    tok = AutoTokenizer.from_pretrained(d["model"])
    model = AutoModelForCausalLM.from_pretrained(d["model"], dtype=torch.bfloat16, device_map=a.device,
                                                 attn_implementation="sdpa").eval()
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
    chosen = np.array([ANIMALS.index(x) for x in d["chosen"]])[: a.max_runs]
    n = len(chosen)
    nl, kvh = model.config.num_hidden_layers, model.config.num_key_value_heads
    groups = {}
    for g in a.groups.split(";"):
        groups[g] = ([(l, h) for l in range(nl) for h in range(kvh)] if g == "all"
                     else [tuple(int(x) for x in p.split(":")) for p in g.split(",")])
    rng = np.random.default_rng(0)
    res = {"model": d["model"], "runs": n, "base": a.base, "groups": {}}

    with torch.no_grad():
        clean = layers_of(model(torch.tensor([prompt + reply], device=a.device), use_cache=True).past_key_values)
        prompt_kv = [(k[:, :, :P].cpu(), v[:, :, :P].cpu()) for k, v in clean]
        strip_kv = [(k[:, :, P:].cpu(), v[:, :, P:].cpu()) for k, v in clean]
        retained = []
        for th in d["thinking"][:n]:
            think = tok.encode("<think>\n" + th + "\n</think>\n\n", add_special_tokens=False)
            lay = layers_of(model(torch.tensor([prompt + think + reply], device=a.device), use_cache=True).past_key_values)
            retained.append([(shift_keys(k[:, :, -R:].cpu(), -len(think), inv_freq), v[:, :, -R:].cpu()) for k, v in lay])

        def dosed(i, heads, dose):
            out = []
            for l in range(nl):
                (ks, vs), (kr, vr) = strip_kv[l], retained[i][l]
                k, v = (ks.clone(), vs.clone()) if a.base == "stripped" else (kr.clone(), vr.clone())
                for hl, h in heads:
                    if hl == l:
                        k[:, h] = (ks[:, h].float() + dose * (kr[:, h].float() - ks[:, h].float())).to(ks.dtype)
                        v[:, h] = (vs[:, h].float() + dose * (vr[:, h].float() - vs[:, h].float())).to(vs.dtype)
                out.append((k, v))
            return out

        for g, heads in groups.items():
            res["groups"][g] = []
            for dose in doses:
                L = np.concatenate([recall_logp(model, prompt_kv, [dosed(i, heads, dose) for i in range(s, min(n, s + a.batch))],
                                                suffix, forms, ANIMALS, a.device) for s in range(0, n, a.batch)])
                rank = (L > L[np.arange(n), chosen][:, None]).sum(1) + 1
                z, p = animal_level(L, chosen, rng, 5000) if not np.allclose(L, L[:1]) else (0.0, 1.0)
                row = {"dose": dose, "raise": float(raises(L, chosen).mean()), "z": float(z), "p": float(p),
                       "top1": float(np.mean(rank == 1)), "mean_rank": float(rank.mean())}
                res["groups"][g].append(row)
                print(f"{g:16s} dose {dose:+5.1f}: raise {row['raise']:+.4f}  z {z:+6.2f}  top-1 {row['top1']:.3f}  "
                      f"mean rank {row['mean_rank']:.1f}", flush=True)
                json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
