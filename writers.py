"""Turn 1: which attention heads write the hidden choice into the reply state that recall reads?

A writer is an attention head at the reply tokens in turn 1.  Its output there (the input of the layer's output
projection, one slice per query head) is replaced by its mean over runs, which removes what is specific to the run
and keeps the average; the reply keys and values are recomputed with that change, and recall reads them, retained in
--read-layers only (stripped elsewhere).  The drop of the own-animal raise is how much of the readout passes through
the writer.  Stage 1 ablates whole layers; stage 2 single query heads within --head-layers.
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
    ap.add_argument("--read-layers", required=True, help="first-last, layers whose retained reply state recall reads")
    ap.add_argument("--head-layers", default="")
    ap.add_argument("--no-layer-stage", action="store_true")
    ap.add_argument("--max-runs", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.read_layers.split("-"))
    read = set(range(lo, hi + 1))
    d = json.load(open(a.result))
    TURN1, RECALL = prompts(d)
    tok = AutoTokenizer.from_pretrained(d["model"])
    model = AutoModelForCausalLM.from_pretrained(d["model"], dtype=torch.bfloat16, device_map=a.device,
                                                 attn_implementation="sdpa").eval()
    cfg = model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
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
    thinks = [tok.encode("<think>\n" + th + "\n</think>\n\n", add_special_tokens=False) for th in d["thinking"][:n]]
    rng = np.random.default_rng(0)
    nl = cfg.num_hidden_layers

    # hooks on every output projection: record the reply-token slice, or overwrite parts of it with the mean
    rec, edit = {}, {}
    def hook(m, args, l):
        x = args[0]
        if "record" in edit:
            rec[l] = x[0, -R:].detach().to("cpu", torch.float16)
        if l in edit:
            x = x.clone()
            for q, mean in edit[l]:
                sl = slice(None) if q is None else slice(q * hd, (q + 1) * hd)
                x[0, -R:, sl] = mean[:, sl].to(x.device, x.dtype)
            return (x,)
    for l, layer in enumerate(model.model.layers):
        layer.self_attn.o_proj.register_forward_pre_hook(lambda m, args, l=l: hook(m, args, l))

    with torch.no_grad():
        clean = layers_of(model(torch.tensor([prompt + reply], device=a.device), use_cache=True).past_key_values)
        prompt_kv = [(k[:, :, :P].cpu(), v[:, :, :P].cpu()) for k, v in clean]
        strip_kv = [(k[:, :, P:].cpu(), v[:, :, P:].cpu()) for k, v in clean]

        def turn1(i):
            lay = layers_of(model(torch.tensor([prompt + thinks[i] + reply], device=a.device), use_cache=True).past_key_values)
            return [(shift_keys(k[:, :, -R:].cpu(), -len(thinks[i]), inv_freq), v[:, :, -R:].cpu()) if l in read
                    else strip_kv[l] for l, (k, v) in enumerate(lay)]

        recall = lambda kvs: recall_logp(model, prompt_kv, kvs, suffix, forms, ANIMALS, a.device)

        # unablated pass: record every layer's reply-token head outputs, and the baseline readout
        mean = {l: torch.zeros(R, cfg.num_attention_heads * hd) for l in range(nl)}
        kvs = []
        for i in range(n):
            edit.clear(); edit["record"] = True
            kvs.append(turn1(i))
            edit.clear()
            for l in range(nl):
                mean[l] += rec[l].float() / n
        L0 = np.concatenate([recall(kvs[s:s + a.batch]) for s in range(0, n, a.batch)])
        del kvs
        res = {"model": d["model"], "runs": n, "read_layers": a.read_layers, "arms": {}}

        def report(name, L):
            z, p = animal_level(L, chosen, rng, 5000)
            res["arms"][name] = {"raise": float(raises(L, chosen).mean()), "z": float(z), "p": float(p)}
            print(f"{name:30s} raise {res['arms'][name]['raise']:+.4f}  animal-level z {z:+.2f}  p {p:.3g}", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)

        def ablated(spec):
            out = []
            for s in range(0, n, a.batch):
                kvs = []
                for i in range(s, min(n, s + a.batch)):
                    edit.clear(); edit.update(spec)
                    kvs.append(turn1(i))
                    edit.clear()
                out.append(recall(kvs))
            return np.concatenate(out)

        report(f"no ablation, read layers {a.read_layers}", L0)
        for l in ([] if a.no_layer_stage else range(hi + 1)):
            report(f"writer layer {l}", ablated({l: [(None, mean[l])]}))
        for l in [int(x) for x in a.head_layers.split(",") if x]:
            for q in range(cfg.num_attention_heads):
                report(f"writer layer {l} q{q}", ablated({l: [(q, mean[l])]}))


if __name__ == "__main__":
    main()
