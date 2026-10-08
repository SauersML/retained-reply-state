"""Hidden-span test on the VPD paper's 4-layer Pile model (base text, no chat or thinking).

Each text is  frame + " X" + middle + cue , X one of 50 single-token animals, cue = the frame repeated (or a neutral
animal cue).  Attention masks, applied in every layer, decide who may read the X position:
  visible   everyone after X                                    (positive control)
  stripped  nobody after X: X has no influence on anything      (own advantage exactly 0)
  retained  the middle tokens may, the cue tokens may not: the cue sees X only through the middle tokens' keys and
            values, which were computed while X was visible -- the base-model analogue of a retained reply state
Readout: log P(animal) at the last cue token over the 50 animals.  Full factorial: every (frame, middle) template with
every animal; statistics as in retained-reply-state/stats.py (animal level).
"""
import argparse
import itertools
import json
import os
import sys
import types
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "fourlayer")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.environ.get("VPD_MODEL_DIR", os.path.expanduser("~/gam/bench/vpd_2951")))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vpd_model as vm
from stats import animal_level, raises
from discrimination import accuracy
from tokenizers import Tokenizer

ANIMALS = ["dog", "cat", "horse", "cow", "pig", "sheep", "goat", "lion", "tiger", "bear", "wolf", "fox", "deer",
           "rabbit", "mouse", "rat", "bird", "eagle", "duck", "chicken", "goose", "fish", "shark", "whale", "snake",
           "frog", "turtle", "monkey", "elephant", "zebra", "camel", "donkey", "bull", "ox", "bat", "bee", "ant",
           "spider", "crab", "seal", "crow", "lamb", "puppy", "pony", "worm", "butterfly", "moth", "fly", "mosquito",
           "hen"]
FRAMES = ["My pet is a", "The animal in the box is a", "In the picture there is a", "Her favourite animal is the",
          "The word on the card was", "Behind the barn we found a"]
MIDDLES = [". I like it a lot.", ". Nobody else knows.", ". That is all I will say about it.",
           ". We talked about it for a while.", ". It was a long day.", ". I will not tell anyone else.",
           ". The weather was cold and wet.", ". Then we went home."]
NEUTRAL = " There are many animals, such as the"

# every causal attention in the target model goes through this mask when one is set
MASK = [None]
_shim = types.SimpleNamespace(**{k: getattr(F, k) for k in dir(F) if not k.startswith("__")})
def _sdpa(q, k, v, is_causal=False, **kw):
    if is_causal and MASK[0] is not None:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=MASK[0])
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, **kw)
_shim.scaled_dot_product_attention = _sdpa
vm.F = _shim


def template_measures(L, chosen, per=50):
    """Recall measures for runs that come as templates of `per` runs (every word once as X), in order.  Each word's log P
    is taken relative to its mean over the runs of the same template, which removes how much each template favors each
    word (the between-template spread of a word's log P, about 1.9 nats, is seven times the hidden word's raise).
      raise           the hidden word's log P minus its mean in the template, averaged over runs (nats; 0 = no recall)
      discrimination  share of other words the hidden word is ranked above, within the template (50% = chance)
      top1            share of runs whose hidden word has the highest log P of the 50 (2% = chance)
      pooled          discrimination with each word centered over all templates pooled (the earlier, noisier measure)"""
    groups = np.arange(len(chosen)) // per
    assert (chosen == np.tile(np.arange(per), len(chosen) // per)).all(), "runs are not templates x words in order"
    return {"raise": float(raises(L, chosen, groups).mean()), "discrimination": float(accuracy(L, chosen, groups=groups)),
            "top1": float(np.mean(L.argmax(1) == chosen)), "pooled": float(accuracy(L, chosen))}


def masks(T, x, mid_end, device):
    causal = torch.ones(T, T, dtype=torch.bool, device=device).tril()
    stripped = causal.clone()
    stripped[x + 1:, x] = False
    retained = causal.clone()
    retained[mid_end:, x] = False
    return {"visible": causal, "stripped": stripped, "retained": retained}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(RESULTS, "hidden_span.json"))
    ap.add_argument("--edit", default="", help="SITE#INDEX:SCALE[,...]: each subcomponent u v^T of VPD's decomposition at "
                                               "SITE is scaled, W -> W + (SCALE - 1) u v^T at every position")
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(vm.TARGET_DIR / "tokenizer.json"))
    ids_of = lambda s: tok.encode(s).ids
    animal_ids = [ids_of(" " + w) for w in ANIMALS]
    assert all(len(t) == 1 for t in animal_ids)
    animal_ids = torch.tensor([t[0] for t in animal_ids])
    model = vm.load_target(a.device)
    if a.edit:
        raw = torch.load(str(vm.VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
        for spec in a.edit.split(","):
            sub, scale = spec.split(":")
            site, idx = sub.split("#")
            u = raw[f"_components.{site.replace('.', '-')}.U"][int(idx)].float().to(a.device)
            v = raw[f"_components.{site.replace('.', '-')}.V"][:, int(idx)].float().to(a.device)
            model.site(site).register_forward_hook(
                lambda m, inp, out, u=u, v=v, f=float(scale) - 1: out + f * (inp[0] @ v)[..., None] * u)
        del raw
    arms = {k: [] for k in ("visible", "stripped", "retained", "neutral visible", "neutral retained")}
    chosen = []
    with torch.no_grad():
        for frame, middle in itertools.product(FRAMES, MIDDLES):
            for cue_name, cue in (("", " " + frame), ("neutral ", NEUTRAL)):
                pre, mid, cu = ids_of(frame), ids_of(middle), ids_of(cue)
                x = len(pre)
                seq = torch.tensor([pre + [int(i)] + mid + cu for i in animal_ids], device=a.device)
                T = seq.shape[1]
                for name, m in masks(T, x, x + 1 + len(mid), a.device).items():
                    arm = cue_name + name
                    if arm not in arms:
                        continue
                    MASK[0] = m[None, None].expand(len(ANIMALS), 1, T, T)
                    logp = torch.log_softmax(model(seq)[:, -1].float(), -1)[:, animal_ids]
                    MASK[0] = None
                    arms[arm].append(logp.cpu().numpy())
            chosen.extend(range(len(ANIMALS)))
    chosen = np.array(chosen)
    rng = np.random.default_rng(0)
    out = {"model": "Goodfire 4-layer" + (f" ({a.edit})" if a.edit else ""), "animals": ANIMALS,
           "chosen": [ANIMALS[c] for c in chosen], "arms": {}}
    for arm, rows in arms.items():
        L = np.concatenate(rows)
        out["arms"][arm] = L.tolist()
        if np.allclose(L[chosen == 0], L[chosen == 1]) and arm.endswith("stripped"):
            print(f"{arm:18s} identical across animals (own advantage exactly 0)")
            continue
        z, p = animal_level(L, chosen, rng, 20000)
        top1 = np.mean(L.argmax(1) == chosen)
        print(f"{arm:18s} raise {raises(L, chosen).mean():+.4f}  animal-level z {z:+.2f}  p {p:.3g}  top-1 {top1:.3f}")
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
