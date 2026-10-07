"""Out-of-distribution check of the weight-level account: on ordinary web text, does each head raise or lower the
probability of tokens that already occurred earlier in the context, as its weights predict?

For every key/value head (all of its query heads together) of the given layers, the head's attention output is
replaced by its mean over all positions of the text (mean ablation), and the change of log p(next token) is measured
at every position.  Repeat effect = the mean change where the next token already occurred earlier in the window,
minus the mean change where it did not (which removes the head's general contribution to prediction).  A head that
copies tokens it attends to lowers repeats when ablated (negative repeat effect); a copy-suppression head raises them.
Weights-only prediction: the copying gain of copying.py, computed on the 2,000 most frequent tokens of the text
instead of the animal names.
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("windows", help="uint32 token windows of length 512 (Qwen3 tokenizer)")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--layers", default="")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    X = torch.from_numpy(np.fromfile(a.windows, dtype=np.uint32).reshape(-1, 512)[: a.n].astype(np.int64))
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=a.device,
                                                 attn_implementation="sdpa").eval()
    cfg = model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    layers = [int(x) for x in a.layers.split(",")] if a.layers else list(range(cfg.num_hidden_layers))

    # repeat mask: the next token already occurred earlier in the window
    nxt = X[:, 1:]
    rep = torch.zeros_like(nxt, dtype=torch.bool)
    for b in range(X.shape[0]):
        seen = set()
        for t in range(nxt.shape[1]):
            seen.add(int(X[b, t]))
            rep[b, t] = int(nxt[b, t]) in seen

    edit, rec = {}, {}
    def hook(m, args, l):
        x = args[0]
        if l in rec:
            rec[l] += x.float().sum((0, 1)).cpu()
        if l in edit:
            x = x.clone()
            for q in edit[l]:
                x[..., q * hd:(q + 1) * hd] = mean[l][q * hd:(q + 1) * hd].to(x.device, x.dtype)
            return (x,)
    for l in layers:
        model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda m, args, l=l: hook(m, args, l))

    def logp():
        out = []
        for s in range(0, X.shape[0], a.batch):
            ids = X[s:s + a.batch].to(a.device)
            lg = model(ids).logits[:, :-1].float()
            out.append((lg.gather(-1, ids[:, 1:, None])[..., 0] - lg.logsumexp(-1)).cpu())
        return torch.cat(out)

    with torch.no_grad():
        rec.update({l: torch.zeros(cfg.num_attention_heads * hd) for l in layers})
        base = logp()
        mean = {l: rec[l] / X.numel() for l in layers}
        rec.clear()

        # weights-only copying gain on the text's most frequent tokens
        uniq, cnt = torch.unique(X, return_counts=True)
        top = uniq[torch.argsort(cnt, descending=True)[:2000]]
        E = model.get_input_embeddings().weight.detach().float()[top]
        Ug = model.lm_head.weight.detach().float()[top] * model.model.norm.weight.detach().float()
        res = {"model": a.model, "windows": a.n, "repeat_fraction": float(rep.float().mean()), "kv_heads": {}}
        for l in layers:
            layer = model.model.layers[l]
            gl = layer.input_layernorm.weight.detach().float()
            Wv = layer.self_attn.v_proj.weight.detach().float()
            Wo = layer.self_attn.o_proj.weight.detach().float()
            n = E / E.pow(2).mean(1, keepdim=True).add(cfg.rms_norm_eps).sqrt() * gl
            for h in range(cfg.num_key_value_heads):
                qs = list(range(h * group, (h + 1) * group))
                gains = []
                for q in qs:
                    C = (n @ Wv[h * hd:(h + 1) * hd].T) @ (Ug @ Wo[:, q * hd:(q + 1) * hd]).T
                    off = C.sum() - C.diagonal().sum()
                    gains.append(float(C.diagonal().mean() - off / (C.numel() - C.shape[0])))
                edit.clear(); edit[l] = qs
                d = logp() - base
                edit.clear()
                eff = float(d[rep].mean() - d[~rep].mean())
                res["kv_heads"][f"{l}:{h}"] = {"copy_gain": float(np.mean(gains)), "repeat_effect": eff,
                                               "repeat_change": float(d[rep].mean()), "other_change": float(d[~rep].mean())}
                print(f"L{l} kv{h}: copying gain {np.mean(gains):+8.2f}   repeat effect of ablation {eff:+.4f}", flush=True)
                json.dump(res, open(a.out, "w"), indent=1)
    g = np.array([v["copy_gain"] for v in res["kv_heads"].values()])
    e = np.array([v["repeat_effect"] for v in res["kv_heads"].values()])
    print(f"correlation of copying gain with the repeat effect of ablation: {np.corrcoef(g, e)[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
