"""Stress tests of the 4-layer recall circuit (Goodfire's model, VPD subcomponents), retained condition, 48 templates.

  routes      necessity: block only the circuit's attention routes (head 2.3 from the later tokens to the word; heads 3.4
              and 3.5 from the cue to the later tokens); sufficiency: block every other head's routes, keep the circuit's
  gates       remove together the subcomponents that switch the routes on (layer-2 keys at the word, layer-2 query at the
              later tokens, layer-3 key at the later tokens, layer-3 query at the cue) and, separately, those that hold
              them back (layer-3 queries at the cue); against random sets of the same size from the same matrices and
              positions
  frames      each gate set's effect on each of the 6 sentence frames separately (same sign everywhere?)
  mediation   over all single gates measured by attention_gates.py: does the change in the head's attention predict the
              change in recall?
usage: stress_circuit.py --out stress_circuit.json
"""
import argparse
import itertools
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hidden_span import ANIMALS, FRAMES, MIDDLES, RESULTS, vm
from discrimination import accuracy
from tokenizers import Tokenizer

NEED = [("h.2.attn.k_proj", 224, "word"), ("h.2.attn.k_proj", 206, "word"), ("h.2.attn.q_proj", 436, "later"),
        ("h.3.attn.k_proj", 145, "later"), ("h.3.attn.q_proj", 182, "cue")]
HOLD = [("h.3.attn.q_proj", 334, "cue"), ("h.3.attn.q_proj", 60, "cue")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(RESULTS, "stress_circuit.json"))
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(0)
    tok = Tokenizer.from_file(str(vm.TARGET_DIR / "tokenizer.json"))
    ids_of = lambda s: tok.encode(s).ids
    animal_ids = torch.tensor([ids_of(" " + w)[0] for w in ANIMALS])
    model = vm.load_target("cpu")
    raw = torch.load(str(vm.VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    L, H, D = model.n_layer, model.n_head, model.hd
    UV = {}
    for l, k in itertools.product(range(L), ("q_proj", "k_proj")):
        s = f"h.{l}.attn.{k}"
        UV[s] = (raw[f"_components.{s.replace('.', '-')}.U"].float(), raw[f"_components.{s.replace('.', '-')}.V"].float())

    templates = []
    for fi, (frame, middle) in enumerate(itertools.product(FRAMES, MIDDLES)):
        pre, mid, cue = ids_of(frame), ids_of(middle), ids_of(" " + frame)
        seq = torch.tensor([pre + [int(i)] + mid + cue for i in animal_ids])
        x, c0, T = len(pre), len(pre) + 1 + len(mid), seq.shape[1]
        pos = {"word": [x], "later": list(range(x + 1, c0)), "cue": list(range(c0, T))}
        templates.append((seq, pos, x, c0, fi // len(MIDDLES)))

    def run(removed=(), blocked=(), temps=None):
        temps = templates if temps is None else temps
        out = []
        for seq, pos, x, c0, _ in temps:
            B, T = seq.shape
            m = torch.ones(L, H, T, T, dtype=torch.bool).tril()
            m[:, :, c0:, x] = False
            for route, l, h in blocked:
                if route == "copy":
                    m[l, h, x + 1:c0, x] = False
                else:
                    m[l, h, c0:, x + 1:c0] = False
            rem = {}
            for site, idx, cls in removed:
                rem.setdefault(site, []).append((idx, pos[cls]))
            z = model.wte[seq]
            for i in range(L):
                n = lambda k: f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}"

                def lin(k, h):
                    y = h @ model.site(n(k)).W.T
                    for idx, p in rem.get(n(k), []):
                        U, V = UV[n(k)]
                        y[:, p] -= (h[:, p] @ V[:, idx])[..., None] * U[idx]
                    return y
                h = vm.rms(z, model.norms[2 * i], model.eps)
                q = model._rope(lin("q_proj", h).view(B, T, H, D).transpose(1, 2), T)
                k_ = model._rope(lin("k_proj", h).view(B, T, H, D).transpose(1, 2), T)
                v = lin("v_proj", h).view(B, T, H, D).transpose(1, 2)
                att = ((q @ k_.transpose(-1, -2)) / math.sqrt(D)).masked_fill(~m[i][None], float("-inf")).softmax(-1)
                z = z + lin("o_proj", (att @ v).transpose(1, 2).reshape(B, T, -1))
                hh = vm.rms(z, model.norms[2 * i + 1], model.eps)
                z = z + lin("down_proj", vm.gelu_tanh(lin("c_fc", hh)))
            logits = vm.rms(z, model.ln_f, model.eps)[:, -1] @ model.wte.T
            out.append(torch.log_softmax(logits, -1)[:, animal_ids].numpy())
        Lp = np.concatenate(out)
        return float(accuracy(Lp, np.tile(np.arange(len(ANIMALS)), len(temps))))

    res = {}
    base = run()
    res["recall"] = base
    copy, read = [("copy", 2, 3)], [("read", 3, 4), ("read", 3, 5)]
    others = [(r, l, h) for r in ("copy", "read") for l in range(L) for h in range(H) if (r, l, h) not in copy + read]
    res["routes"] = {"circuit blocked": run(blocked=copy + read), "copy step blocked": run(blocked=copy),
                     "read step blocked": run(blocked=read), "only the circuit open": run(blocked=others),
                     "every route blocked": run(blocked=copy + read + others)}
    print(f"recall {100 * base:.1f}%; " + "; ".join(f"{k} {100 * v:.1f}%" for k, v in res["routes"].items()), flush=True)

    def random_like(gates):
        picks = []
        for site, idx, cls in gates:
            j = int(rng.integers(len(UV[site][0])))
            picks.append((site, j, cls))
        return picks

    for name, gates in (("needed gates removed", NEED), ("holding gates removed", HOLD)):
        r = run(removed=gates)
        rand = [run(removed=random_like(gates)) for _ in range(a.random)]
        res[name] = {"recall": r, "random": rand}
        print(f"{name}: {100 * r:.1f}%; random sets of {len(gates)}: " + ", ".join(f"{100 * x:.1f}" for x in rand), flush=True)

    res["frames"] = {}
    for f in range(len(FRAMES)):
        temps = [t for t in templates if t[4] == f]
        b = run(temps=temps)
        res["frames"][FRAMES[f]] = {"recall": b, "needed gates removed": run(removed=NEED, temps=temps),
                                    "holding gates removed": run(removed=HOLD, temps=temps)}
        v = res["frames"][FRAMES[f]]
        print(f"  frame '{FRAMES[f]}': {100 * b:.1f}% -> needed removed {100 * v['needed gates removed']:.1f}%, "
              f"holding removed {100 * v['holding gates removed']:.1f}%", flush=True)

    g = json.load(open(os.path.join(RESULTS, "attention_gates.json")))
    da, dr = [], []
    for k, v in g["gates"].items():
        if k.startswith("h.2"):
            da.append(v["att_2_3"] / g["att_2_3"] - 1)
        else:
            da.append(np.mean(v["att_3_45"]) / np.mean(g["att_3_45"]) - 1)
        dr.append(v["recall"] - g["recall"])
    res["mediation"] = {"corr_attention_recall": float(np.corrcoef(da, dr)[0, 1]), "n": len(da)}
    print(f"mediation: correlation of attention change and recall change over {len(da)} gates: {res['mediation']['corr_attention_recall']:+.2f}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
