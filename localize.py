"""Where does recall read the retained reply state?  (Run on a hidden_choice.py result.)

For each run, the reply's turn-1 keys and values are kept only in part of the cache and replaced elsewhere by the
stripped ones (computed without the thinking):
  layer bands   one band of consecutive layers retained
  tokens        one reply token retained, in every layer
  kv heads      one key/value head retained, in one layer (layers given by --head-layers)
Keys and values of layer l are read only by the attention of layer l, so a part whose retention alone reproduces the
own-animal raise is where recall reads the hidden choice.  Readout and statistics as in hidden_choice.py / stats.py.
"""
import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hidden_choice import ANIMALS, RECALL, TURN1
from kvtools import layers_of, make_cache, score_candidates, shift_keys
from stats import animal_level, raises


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--band", type=int, default=4)
    ap.add_argument("--head-layers", default="")
    ap.add_argument("--max-runs", type=int, default=600)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.result))
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
    thinking = d["thinking"][: a.max_runs]
    n = len(chosen)

    with torch.no_grad():
        clean = layers_of(model(torch.tensor([prompt + reply], device=a.device), use_cache=True).past_key_values)
        prompt_kv = [(k[:, :, :P].cpu(), v[:, :, :P].cpu()) for k, v in clean]
        strip_kv = [(k[:, :, P:].cpu(), v[:, :, P:].cpu()) for k, v in clean]
        L_layers = len(strip_kv)
        kv_heads = strip_kv[0][0].shape[1]
        reply_kv = []
        for th in thinking:
            think = tok.encode("<think>\n" + th + "\n</think>\n\n", add_special_tokens=False)
            lay = layers_of(model(torch.tensor([prompt + think + reply], device=a.device), use_cache=True).past_key_values)
            reply_kv.append([(shift_keys(k[:, :, -R:].cpu(), -len(think), inv_freq), v[:, :, -R:].cpu()) for k, v in lay])

        arms = {}
        for lo in range(0, L_layers, a.band):
            arms[f"layers {lo}-{min(L_layers, lo + a.band) - 1}"] = ("layers", set(range(lo, min(L_layers, lo + a.band))), None)
        arms["all layers"] = ("layers", set(range(L_layers)), None)
        for t in range(R):
            arms[f"token {t} {tok.decode([reply[t]])!r}"] = ("token", t, None)
        for l in [int(x) for x in a.head_layers.split(",") if x]:
            for h in range(kv_heads):
                arms[f"layer {l} kv-head {h}"] = ("head", l, h)

        def mixed(i, kind, sel, h):
            out = []
            for li, ((ks, vs), (kr, vr)) in enumerate(zip(strip_kv, reply_kv[i])):
                k, v = ks.clone(), vs.clone()
                if kind == "layers" and li in sel:
                    k, v = kr, vr
                elif kind == "token":
                    k[:, :, sel], v[:, :, sel] = kr[:, :, sel], vr[:, :, sel]
                elif kind == "head" and li == sel:
                    k[:, h], v[:, h] = kr[:, h], vr[:, h]
                out.append((k, v))
            return out

        results = {}
        t0 = time.time()
        for name, (kind, sel, h) in arms.items():
            L = np.zeros((n, len(ANIMALS)))
            for i in range(n):
                o = model(torch.tensor([suffix], device=a.device),
                          past_key_values=make_cache([prompt_kv, mixed(i, kind, sel, h)], a.device), use_cache=True)
                last = torch.log_softmax(o.logits[0, -1].float(), -1)
                scores = score_candidates(model, layers_of(o.past_key_values), last, [t for _, t in forms], a.device)
                best = {}
                for (c, _), s in zip(forms, scores):
                    best[c] = np.logaddexp(best.get(c, -np.inf), s)
                L[i] = [best[c] for c in ANIMALS]
            z, p = animal_level(L, chosen, np.random.default_rng(0), 5000)
            results[name] = {"raise": float(raises(L, chosen).mean()), "z": float(z), "p": float(p)}
            print(f"{name:28s} raise {results[name]['raise']:+.4f}  animal-level z {z:+.2f}  p {p:.3g}  ({time.time() - t0:.0f}s)", flush=True)
            json.dump({"model": d["model"], "runs": n, "arms": results}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
