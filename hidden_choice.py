"""Hidden choice, fixed visible reply, later recall.

Turn 1: the user asks the model to pick one animal from a list of 50 in its thinking and to reply "I understand.".
The animal is drawn uniformly at random and written into the start of the thinking (with varied wording, so every run
has a distinct context); the model writes the rest of the thinking and the reply.  Runs whose reply is exactly
"I understand." are kept.

Turn 2 asks for the animal.  The thinking is not in the turn-2 context in any arm except 'visible':
  stripped  the reply's keys and values are computed without the thinking (what a server does when it re-encodes the
            transcript); every run's turn-2 context is then identical
  retained  the reply's keys and values from turn 1, computed while the thinking was present, are kept; the thinking's
            own entries are removed and the reply's keys moved to the positions they hold without the thinking
  neutral   as 'retained', but turn 2 asks for any animal from the list instead of the chosen one (priming control)
  visible   the thinking stays in context (positive control)
Readout: log P of each of the 50 animals as the answer, read after an empty thinking block and "Animal:".
Also saved: the residual stream at the reply's last text token for every layer (for linear probes).
"""
import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvtools import layers_of, make_cache, score_candidates, shift_keys

ANIMALS = ["aardvark", "albatross", "alpaca", "armadillo", "axolotl", "badger", "beaver", "bison", "camel", "capybara",
           "chameleon", "cheetah", "chinchilla", "cobra", "condor", "coyote", "crocodile", "dolphin", "dugong", "falcon",
           "ferret", "flamingo", "gazelle", "gecko", "giraffe", "gorilla", "hedgehog", "heron", "hyena", "iguana",
           "jackal", "jaguar", "kangaroo", "koala", "lemur", "lobster", "lynx", "manatee", "meerkat", "mongoose",
           "narwhal", "ocelot", "octopus", "okapi", "orca", "otter", "pangolin", "pelican", "penguin", "porcupine"]
TURN1 = ("Here is a list of animals: " + ", ".join(ANIMALS) + ". In your thinking only, choose one animal from this "
         "list at random and hold it in mind; later you will be asked which one you chose, without your thinking being "
         "available. Do not reveal it. Reply with exactly the phrase: \"I understand.\" (Without the quotation marks.)")
RECALL = ("In the previous turn you chose one animal from the list in your thinking. That thinking is not shown to you "
          "now. Which animal did you choose? Answer in exactly this format: Animal: <animal>")
NEUTRAL = "Name one animal from the list above, whichever you like. Answer in exactly this format: Animal: <animal>"
OPEN = ["Okay, I need to pick one at random.", "Let me choose randomly from the list.", "Alright, a random pick.",
        "I'll just let chance decide.", "Hmm, picking one without overthinking.", "Fine, one animal from the list."]
CLOSE = ["It came up first in my draw.", "No particular reason.", "That one stands out.", "Good enough.",
         "I'll stick with it.", "Decided.", "That's my pick, final."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=600)
    ap.add_argument("--arms", default="stripped,visible,retained,neutral")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=a.device,
                                                 attn_implementation="sdpa").eval()
    inv_freq = model.model.rotary_emb.inv_freq.detach().cpu()
    chat = lambda msgs: tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                           enable_thinking=True), add_special_tokens=False)
    prompt = chat([{"role": "user", "content": TURN1}])
    reply = tok.encode("I understand.<|im_end|>", add_special_tokens=False)
    P, R = len(prompt), len(reply)
    answer = tok.encode("<think>\n\n</think>\n\nAnimal:", add_special_tokens=False)
    suffix = {}
    for name, question in (("recall", RECALL), ("neutral", NEUTRAL)):
        full = chat([{"role": "user", "content": TURN1}, {"role": "assistant", "content": "I understand."},
                     {"role": "user", "content": question}])
        assert full[:P] == prompt and full[P:P + R] == reply, "chat template does not keep the turn-1 prefix"
        suffix[name] = full[P + R:] + answer
    forms = [(c, tok.encode(f, add_special_tokens=False)) for c in ANIMALS for f in (" " + c.title(), " " + c)]

    runs = []
    with torch.no_grad():
        t0 = time.time()
        for s0 in range(0, a.n, a.batch):
            B = min(a.batch, a.n - s0)
            chosen = [ANIMALS[k] for k in rng.integers(0, len(ANIMALS), B)]
            starts = [f"<think>\n{OPEN[rng.integers(len(OPEN))]} Draw #{rng.integers(100, 1000)}: {c}. "
                      f"{CLOSE[rng.integers(len(CLOSE))]} I'll keep {c} in mind." for c in chosen]
            seqs = [prompt + tok.encode(s, add_special_tokens=False) for s in starts]
            W = max(map(len, seqs))
            pad = tok.pad_token_id
            inp = torch.tensor([[pad] * (W - len(s)) + s for s in seqs], device=a.device)
            att = torch.tensor([[0] * (W - len(s)) + [1] * len(s) for s in seqs], device=a.device)
            out = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=a.max_new, do_sample=True,
                                 temperature=0.6, top_p=0.95, top_k=20, pad_token_id=pad)
            for j in range(B):
                text = starts[j] + tok.decode(out[j, W:], skip_special_tokens=False).split("<|im_end|>")[0]
                if "</think>" not in text:
                    continue
                thinking, visible = text.split("</think>", 1)
                if visible.strip() == "I understand.":
                    runs.append({"animal": chosen[j], "thinking": thinking.replace("<think>", "", 1).strip()})
            del out
            print(f"turn 1: {s0 + B} generated, {len(runs)} kept, {time.time() - t0:.0f}s", flush=True)

        clean = layers_of(model(torch.tensor([prompt + reply], device=a.device), use_cache=True).past_key_values)
        prompt_kv = [(k[:, :, :P].cpu(), v[:, :, :P].cpu()) for k, v in clean]
        stripped_reply_kv = [(k[:, :, P:].cpu(), v[:, :, P:].cpu()) for k, v in clean]
        states = []
        for r in runs:
            think = tok.encode("<think>\n" + r["thinking"] + "\n</think>\n\n", add_special_tokens=False)
            o = model(torch.tensor([prompt + think + reply], device=a.device), use_cache=True, output_hidden_states=True)
            r["reply_kv"] = [(shift_keys(k[:, :, -R:].cpu(), -len(think), inv_freq), v[:, :, -R:].cpu())
                             for k, v in layers_of(o.past_key_values)]
            r["think_ids"] = think
            states.append(torch.stack([h[0, -2].float().cpu() for h in o.hidden_states]).numpy())
        np.save(a.out.replace(".json", "_states.npy"), np.stack(states).astype(np.float16))

        result = {"model": a.model, "animals": ANIMALS, "chosen": [r["animal"] for r in runs],
                  "thinking": [r["thinking"] for r in runs], "arms": {}}
        for arm in a.arms.split(","):
            L = np.zeros((len(runs), len(ANIMALS)))
            for i, r in enumerate(runs if arm != "stripped" else runs[:1]):
                if arm == "visible":
                    o = model(torch.tensor([prompt + r["think_ids"] + reply + suffix["recall"]], device=a.device), use_cache=True)
                else:
                    kv = stripped_reply_kv if arm == "stripped" else r["reply_kv"]
                    o = model(torch.tensor([suffix["neutral" if arm == "neutral" else "recall"]], device=a.device),
                              past_key_values=make_cache([prompt_kv, kv], a.device), use_cache=True)
                last = torch.log_softmax(o.logits[0, -1].float(), -1)
                scores = score_candidates(model, layers_of(o.past_key_values), last, [t for _, t in forms], a.device)
                best = {}
                for (c, _), s in zip(forms, scores):
                    best[c] = np.logaddexp(best.get(c, -np.inf), s)
                L[i] = [best[c] for c in ANIMALS]
            if arm == "stripped":
                L[:] = L[0]
            result["arms"][arm] = L.tolist()
            print(f"{arm}: done", flush=True)
        json.dump(result, open(a.out, "w"))


if __name__ == "__main__":
    main()
