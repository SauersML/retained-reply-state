"""Edit the weights of chosen attention heads and measure what the edited model does with the reply cache kept.

An edit scales the output projection of a key/value head's query heads, W_O[:, q*hd:(q+1)*hd] -> s W_O[:, ...], in the
model itself, so turn 1 (the reply's keys and values) and turn 2 are both computed by the edited model.  Reported per
edit, on a hidden_choice.py result's runs and thinking texts:
  discrimination  share of other animals the hidden animal is ranked above, each animal relative to its mean over runs
                  (50% = chance; below 50% the answer points away from the hidden animal); two-sided permutation p
  top-1, mean rank of the hidden animal; the own-animal raise (nats)
  damage          KL from the unedited model's next-token distributions on held-out web text (nats per token)
With --turn1-drop, turn 1 is recomputed without one sentence of its instruction, over the same thinking texts.
usage: edit_heads.py RESULT.json WINDOWS.u32 --edits "21:0*4;21:5*0,21:6*0;..." --out OUT.json
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from discrimination import accuracy
from hidden_choice import task_of
from kvtools import layers_of, recall_logp, shift_keys
from stats import raises


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("windows")
    ap.add_argument("--edits", required=True, help="edits separated by ';', each layer:kv_head*scale pairs, comma separated")
    ap.add_argument("--max-runs", type=int, default=600)
    ap.add_argument("--recall", default=None, choices=["A", "B"], help="ask the recall question of this wording instead")
    ap.add_argument("--turn1-drop", default=None, help="remove this sentence from turn 1 (the thinking texts are kept)")
    ap.add_argument("--arms", default="retained", help="retained, neutral (any-animal question), visible (thinking kept)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.result))
    T = task_of(d)
    TURN1, RECALL, NEUTRAL, ANIMALS = T["turn1"], T["recall"], T["neutral"], T["items"]
    if a.recall:
        from hidden_choice import task
        RECALL = task(a.recall, d.get("items", "animals"))["recall"]
    if a.turn1_drop:
        assert a.turn1_drop in TURN1, "the sentence to drop is not in turn 1"
        TURN1 = TURN1.replace(a.turn1_drop, "")
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
    answer = tok.encode(f"<think>\n\n</think>\n\n{T['label']}:", add_special_tokens=False)
    suffix = full[P + R:] + answer
    full_n = chat([{"role": "user", "content": TURN1}, {"role": "assistant", "content": "I understand."},
                   {"role": "user", "content": NEUTRAL}])
    suffixes = {"retained": suffix, "neutral": full_n[P + R:] + answer, "visible": suffix}
    forms = [(c, tok.encode(f, add_special_tokens=False)) for c in ANIMALS for f in (" " + c.title(), " " + c)]
    chosen = np.array([ANIMALS.index(x) for x in d["chosen"]])[: a.max_runs]
    thinks = [tok.encode("<think>\n" + th + "\n</think>\n\n", add_special_tokens=False) for th in d["thinking"][: a.max_runs]]
    n = len(chosen)
    web = torch.from_numpy(np.fromfile(a.windows, dtype=np.uint32).reshape(-1, 512)[:4].astype(np.int64))
    original = {l: model.model.layers[l].self_attn.o_proj.weight.detach().clone() for l in range(cfg.num_hidden_layers)}

    def web_logp():
        """Next-token log-probabilities on the web windows, one window at a time, kept in bfloat16."""
        return [torch.log_softmax(model(web[s:s + 1].to(a.device)).logits.float(), -1).to("cpu", torch.bfloat16)
                for s in range(len(web))]

    def web_kl(lp):
        return float(np.mean([(b.float().exp() * (b.float() - x.float())).sum(-1).mean().item() for b, x in zip(base_web, lp)]))

    rng = np.random.default_rng(0)
    res = {"model": d["model"], "runs": n, "turn1_drop": a.turn1_drop, "edits": {}}
    with torch.no_grad():
        base_web = web_logp()
        for spec in ["none"] + a.edits.split(";"):
            for l, W in original.items():
                model.model.layers[l].self_attn.o_proj.weight.copy_(W)
            if spec != "none":
                for part in spec.split(","):
                    lh, scale = part.split("*")
                    l, h = (int(x) for x in lh.split(":"))
                    W = model.model.layers[l].self_attn.o_proj.weight
                    W[:, h * group * hd:(h + 1) * group * hd] *= float(scale)
            clean = layers_of(model(torch.tensor([prompt], device=a.device), use_cache=True).past_key_values)
            prompt_kv = [(k.cpu(), v.cpu()) for k, v in clean]
            arms = a.arms.split(",")
            Ls = {arm: [] for arm in arms}
            for s in range(0, n, a.batch):
                kvs, fulls = [], []
                for i in range(s, min(n, s + a.batch)):
                    lay = layers_of(model(torch.tensor([prompt + thinks[i] + reply], device=a.device), use_cache=True).past_key_values)
                    kvs.append([(shift_keys(k[:, :, -R:].cpu(), -len(thinks[i]), inv_freq), v[:, :, -R:].cpu()) for k, v in lay])
                    if "visible" in arms:
                        fulls.append([(k.cpu(), v.cpu()) for k, v in lay])
                for arm in arms:
                    if arm == "visible":
                        empty = [(k[:, :, :0], v[:, :, :0]) for k, v in fulls[0]]
                        Ls[arm].append(np.concatenate([recall_logp(model, empty, [f], suffix, forms, ANIMALS, a.device) for f in fulls]))
                    else:
                        Ls[arm].append(recall_logp(model, prompt_kv, kvs, suffixes[arm], forms, ANIMALS, a.device))
                if a.device == "mps":
                    torch.mps.empty_cache()         # the allocator's cache of variable-length blocks otherwise grows past 16 GB
            r = {}
            for arm in arms:
                L = np.concatenate(Ls[arm])
                present = np.unique(chosen)
                acc = accuracy(L, chosen)
                null = np.array([accuracy(L, chosen, rng.permutation(present)) for _ in range(1000)])
                pv = float((1 + np.sum(np.abs(null - 0.5) >= abs(acc - 0.5))) / 1001)
                rank = (L > L[np.arange(n), chosen][:, None]).sum(1) + 1
                r[arm] = {"discrimination": float(acc), "p": pv, "top1": float(np.mean(rank == 1)),
                          "mean_rank": float(rank.mean()), "raise": float(raises(L, chosen).mean())}
            kl = web_kl(web_logp())
            r["web_kl"] = kl
            res["edits"][spec] = r
            print(f"{spec:28s} " + "  ".join(f"{arm}: {100 * r[arm]['discrimination']:5.1f}% (p {r[arm]['p']:.2g}, top-1 "
                                             f"{100 * r[arm]['top1']:.1f}%)" for arm in arms) + f"  web KL {kl:.4f}", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
