"""Checks against direct recomputation: moved keys equal keys computed at the new positions, and packed candidate scores
equal scores from full forward passes.  Run with: python test_kvtools.py [model]"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvtools import layers_of, score_candidates, shift_keys

name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="sdpa").eval()
ids = torch.tensor([tok.encode("The animal I chose was:", add_special_tokens=False)])
with torch.no_grad():
    base = layers_of(model(ids, use_cache=True).past_key_values)
    moved = layers_of(model(ids, position_ids=torch.arange(7, 7 + ids.shape[1])[None], use_cache=True).past_key_values)
    err = max(float((shift_keys(k0, 7, model.model.rotary_emb.inv_freq) - k1).abs().max()) for (k0, _), (k1, _) in zip(base, moved))
    print(f"largest key difference after moving 7 positions: {err:.2e}")
    assert err < 1e-3

    out = model(ids, use_cache=True)
    last = torch.log_softmax(out.logits[0, -1].float(), -1)
    cands = [tok.encode(" " + w, add_special_tokens=False) for w in ["axolotl", "Pangolin", "dog", "star-nosed mole"]]
    packed = score_candidates(model, layers_of(out.past_key_values), last, cands, "cpu")
    for c, p in zip(cands, packed):
        full = model(torch.cat([ids, torch.tensor([c])], 1)).logits[0].float().log_softmax(-1)
        ref = sum(float(full[ids.shape[1] - 1 + t, c[t]]) for t in range(len(c)))
        assert abs(ref - p) < 1e-3, (ref, p)
    print("packed candidate scores match full forward passes")
