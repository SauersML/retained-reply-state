"""Dose-response of the hidden-word circuit in the 4-layer model (templates from hidden_span.py).

The cue reads keys/values  B + a (A - B)  of the middle tokens: A from the pass where the middle tokens see the word
(retained), B from the pass where nothing after the word does (stripped).  a = 0 is stripped, a = 1 retained.
  state    every layer and head
  readers  layer 3 heads 4 and 5 only (others stripped)
  writer   pass A with layer 2 head 3's output at the middle tokens set to  o_B + a (o_A - o_B); the cue reads
           all of that pass's middle state
Reported per a: own-word raise, animal-level z, top-1 accuracy (chance 1/50) and the mean rank of the hidden word.
"""
import itertools
import json
import os
import sys
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "fourlayer")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hidden_span import ANIMALS, FRAMES, MIDDLES, vm
from localize4l import forward
from stats import animal_level, raises
from tokenizers import Tokenizer

DOSES = [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6, 8]


def main():
    torch.set_grad_enabled(False)
    tok = Tokenizer.from_file(str(vm.TARGET_DIR / "tokenizer.json"))
    ids_of = lambda s: tok.encode(s).ids
    animal_ids = torch.tensor([ids_of(" " + w)[0] for w in ANIMALS])
    model = vm.load_target("cpu")
    nl, H, hd = model.n_layer, model.n_head, model.hd
    o3 = model.site("h.2.attn.o_proj")
    L = {}
    for frame, middle in itertools.product(FRAMES, MIDDLES):
        pre, mid, cue = ids_of(frame), ids_of(middle), ids_of(" " + frame)
        x, m0, c0 = len(pre), len(pre) + 1, len(pre) + 1 + len(mid)
        seq = torch.tensor([pre + [int(i)] + mid + cue for i in animal_ids])
        T = seq.shape[1]
        causal = torch.ones(T, T, dtype=torch.bool).tril()
        strip = causal.clone(); strip[x + 1:, x] = False
        ret = causal.clone(); ret[c0:, x] = False
        readout = lambda lg: torch.log_softmax(lg.float(), -1)[:, animal_ids].numpy()
        # o_proj inputs of layer 2 in both passes, to dose the writer head
        o3.cache_input = True
        _, kvA = forward(model, seq, [ret[None, None]] * nl); oA = o3.last_input
        _, kvB = forward(model, seq, [strip[None, None]] * nl); oB = o3.last_input
        o3.cache_input = False; o3.last_input = None
        def mix(kv1, kv0, a, sel):
            return {l: (kv0[l][0] + a * (kv1[l][0] - kv0[l][0]), kv0[l][1] + a * (kv1[l][1] - kv0[l][1]), s) for l, s in sel.items()}
        allpos = torch.zeros(H, T, dtype=torch.bool); allpos[:, m0:c0] = True
        rd = torch.zeros(H, T, dtype=torch.bool); rd[[4, 5], m0:c0] = True
        for a in DOSES:
            L.setdefault(("state", a), []).append(readout(forward(model, seq, [strip[None, None]] * nl, mix(kvA, kvB, a, {l: allpos for l in range(nl)}), c0)[0]))
            L.setdefault(("readers", a), []).append(readout(forward(model, seq, [strip[None, None]] * nl, mix(kvA, kvB, a, {3: rd}), c0)[0]))
            # writer: pass A with L2h3's output at the middle tokens dosed
            def dose_in(inp, a=a):
                inp = inp.clone()
                sl = slice(3 * hd, 4 * hd)
                inp[:, m0:c0, sl] = oB[:, m0:c0, sl] + a * (oA[:, m0:c0, sl] - oB[:, m0:c0, sl])
                return inp
            o3.in_fn = dose_in
            _, kvW = forward(model, seq, [ret[None, None]] * nl)
            o3.in_fn = None
            L.setdefault(("writer", a), []).append(readout(forward(model, seq, [strip[None, None]] * nl, mix(kvW, kvB, 1.0, {l: allpos for l in range(nl)}), c0)[0]))
    chosen = np.tile(np.arange(len(ANIMALS)), len(FRAMES) * len(MIDDLES))
    rng = np.random.default_rng(0)
    res = {}
    for (arm, a), rows in L.items():
        M = np.concatenate(rows)
        r = raises(M, chosen).mean()
        z = animal_level(M, chosen, rng, 2000)[0] if a != 0 else 0.0
        rank = (M > M[np.arange(len(chosen)), chosen][:, None]).sum(1) + 1
        res.setdefault(arm, []).append({"dose": a, "raise": float(r), "z": float(z), "top1": float(np.mean(rank == 1)), "mean_rank": float(rank.mean())})
        print(f"{arm:8s} a={a:+5.1f}: raise {r:+.3f}  z {z:+6.1f}  top-1 {np.mean(rank == 1):.3f}  mean rank {rank.mean():5.1f}", flush=True)
    json.dump(res, open(os.path.join(RESULTS, "dose4l.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
