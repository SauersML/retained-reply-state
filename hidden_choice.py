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
import os
import re
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
COUNTRIES = ["argentina", "australia", "austria", "belgium", "brazil", "canada", "chile", "china", "colombia", "cuba",
             "denmark", "egypt", "ethiopia", "finland", "france", "germany", "ghana", "greece", "iceland", "india",
             "indonesia", "iran", "iraq", "ireland", "israel", "italy", "japan", "jordan", "kenya", "malaysia", "mexico",
             "mongolia", "morocco", "nepal", "netherlands", "nigeria", "norway", "pakistan", "peru", "philippines",
             "poland", "portugal", "russia", "spain", "sweden", "switzerland", "thailand", "turkey", "ukraine", "vietnam"]
# the hidden item's category: its list, plural, singular and answer label
ITEMS = {"animals": (ANIMALS, "animals", "animal", "Animal"), "countries": (COUNTRIES, "countries", "country", "Country")}
TURN1_T = ("Here is a list of {pl}: {items}. In your thinking only, choose one {sg} from this list at random and hold it "
           "in mind; later you will be asked which one you chose, without your thinking being available. Do not reveal "
           "it. Reply with exactly the phrase: \"I understand.\" (Without the quotation marks.)")
RECALL_T = ("In the previous turn you chose one {sg} from the list in your thinking. That thinking is not shown to you "
            "now. Which {sg} did you choose? Answer in exactly this format: {lab}: <{sg}>")
# a second wording of both turns, to test whether results depend on the phrasing
TURN1_B_T = ("You are participating in an experiment about how your internal reasoning relates to your later answers. "
             "Here is a list of {pl}: {items}. In your internal thinking only, choose one {sg} from this list at random "
             "and hold it firmly in mind; later you will be asked to recall it without your thinking being available. Do "
             "not reveal it in your visible answer. In your final answer for this turn, respond with exactly the phrase: "
             "\"I understand.\" (Without the quotation marks.)")
RECALL_B_T = ("In the preceding turn of this conversation, during your internal thinking, you chose one {sg} from the "
              "list. That thinking is not shown to you now. Recall, introspect, or reconstruct which {sg} you chose, and "
              "answer in exactly this format: {lab}: <{sg}>")
NEUTRAL_T = "Name one {sg} from the list above, whichever you like. Answer in exactly this format: {lab}: <{sg}>"


def task(wording="A", items="animals"):
    """The prompts of a wording and item category: turn 1, recall, the neutral question, the answer label, the items."""
    names, pl, sg, lab = ITEMS[items]
    f = dict(pl=pl, sg=sg, lab=lab, items=", ".join(names))
    t1, rc = (TURN1_T, RECALL_T) if wording == "A" else (TURN1_B_T, RECALL_B_T)
    return {"turn1": t1.format(**f), "recall": rc.format(**f), "neutral": NEUTRAL_T.format(**f), "label": lab, "items": names}


TURN1, RECALL = task("A")["turn1"], task("A")["recall"]
TURN1_B, RECALL_B = task("B")["turn1"], task("B")["recall"]
WORDINGS = {"A": (TURN1, RECALL), "B": (TURN1_B, RECALL_B)}


def last_named(text, names=ANIMALS):
    """The item of the list whose name ends last in the text (whole words, any case), or None."""
    low, best, end = text.lower(), None, -1
    for c in names:
        for m in re.finditer(r"\b" + c + r"s?\b", low):
            if m.end() > end:
                best, end = c, m.end()
    return best


def prompts(result):
    """The first-turn and recall prompts a hidden_choice.py result was made with."""
    t = task_of(result)
    return t["turn1"], t["recall"]


def task_of(result):
    """The full task (prompts, answer label, items) a hidden_choice.py result was made with."""
    return task(result.get("wording", "A"), result.get("items", "animals"))


NEUTRAL = task("A")["neutral"]
OPEN = ["Okay, I need to pick one at random.", "Let me choose randomly from the list.", "Alright, a random pick.",
        "I'll just let chance decide.", "Hmm, picking one without overthinking.", "Fine, one {sg} from the list."]
CLOSE = ["It came up first in my draw.", "No particular reason.", "That one stands out.", "Good enough.",
         "I'll stick with it.", "Decided.", "That's my pick, final."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=600)
    ap.add_argument("--arms", default="stripped,visible,retained,neutral")
    ap.add_argument("--choice", default="forced", choices=["forced", "free"],
                    help="forced: the animal is drawn uniformly and written into the thinking; free: the model chooses, "
                         "and the chosen animal is the last of the 50 names its thinking mentions")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wording", default="A", choices=sorted(WORDINGS))
    ap.add_argument("--items", default="animals", choices=sorted(ITEMS))
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
    T = task(a.wording, a.items)
    TURN1, RECALL, NEUTRAL, NAMES = T["turn1"], T["recall"], T["neutral"], T["items"]
    prompt = chat([{"role": "user", "content": TURN1}])
    reply = tok.encode("I understand.<|im_end|>", add_special_tokens=False)
    P, R = len(prompt), len(reply)
    answer = tok.encode(f"<think>\n\n</think>\n\n{T['label']}:", add_special_tokens=False)
    suffix = {}
    for name, question in (("recall", RECALL), ("neutral", NEUTRAL)):
        full = chat([{"role": "user", "content": TURN1}, {"role": "assistant", "content": "I understand."},
                     {"role": "user", "content": question}])
        assert full[:P] == prompt and full[P:P + R] == reply, "chat template does not keep the turn-1 prefix"
        suffix[name] = full[P + R:] + answer
    forms = [(c, tok.encode(f, add_special_tokens=False)) for c in NAMES for f in (" " + c.title(), " " + c)]

    # turn 1 is saved batch by batch, so a killed run resumes where it stopped (the draws of finished batches are replayed)
    ckpt = a.out.replace(".json", "_turn1.jsonl")
    done = [json.loads(line) for line in open(ckpt)] if os.path.exists(ckpt) else []
    runs = [r for b in done for r in b["runs"]]
    finished = {b["s0"] for b in done}
    with torch.no_grad():
        t0 = time.time()
        for s0 in range(0, a.n, a.batch):
            B = min(a.batch, a.n - s0)
            chosen = [NAMES[k] for k in rng.integers(0, len(NAMES), B)]
            starts = ["<think>\n"] * B if a.choice == "free" else [f"<think>\n{OPEN[rng.integers(len(OPEN))].format(sg=ITEMS[a.items][2])} Draw #{rng.integers(100, 1000)}: {c}. "
                      f"{CLOSE[rng.integers(len(CLOSE))]} I'll keep {c} in mind." for c in chosen]
            if s0 in finished:
                continue
            new = []
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
                    thinking = thinking.replace("<think>", "", 1).strip()
                    if a.choice == "free":
                        named = last_named(thinking, NAMES)
                        if named is None:
                            continue
                        chosen[j] = named
                    new.append({"animal": chosen[j], "thinking": thinking})
            del out
            if a.device == "mps":
                torch.mps.empty_cache()
            runs += new
            with open(ckpt, "a") as f:
                f.write(json.dumps({"s0": s0, "runs": new}) + "\n")
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
            del o
            if a.device == "mps" and len(states) % 64 == 0:
                torch.mps.empty_cache()
        np.save(a.out.replace(".json", "_states.npy"), np.stack(states).astype(np.float16))

        result = {"model": a.model, "wording": a.wording, "items": a.items, "choice": a.choice, "animals": NAMES, "chosen": [r["animal"] for r in runs],
                  "thinking": [r["thinking"] for r in runs], "arms": {}}
        for arm in a.arms.split(","):
            L = np.zeros((len(runs), len(NAMES)))
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
                L[i] = [best[c] for c in NAMES]
            if arm == "stripped":
                L[:] = L[0]
            result["arms"][arm] = L.tolist()
            print(f"{arm}: done", flush=True)
        json.dump(result, open(a.out, "w"))


if __name__ == "__main__":
    main()
