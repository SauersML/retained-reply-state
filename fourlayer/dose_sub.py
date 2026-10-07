"""Dose-response of single VPD subcomponents in the 4-layer model: the weight matrix W of the subcomponent's site becomes
W + (a - 1) u v^T everywhere (a = 0 deletes the subcomponent, 1 is the model, 4 makes it four times as strong), and the
retained condition of hidden_span.py is measured: the own-word raise and top-1 (the hidden word ranked first)."""
import os
import itertools, json, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pd4l
from hidden_span import ANIMALS, FRAMES, MIDDLES, vm
from stats import raises
from tokenizers import Tokenizer
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "fourlayer")

SUBS = [("h.2.attn.k_proj", 224), ("h.2.attn.k_proj", 206), ("h.2.attn.o_proj", 735), ("h.3.attn.k_proj", 145),
        ("h.3.attn.o_proj", 806), ("h.3.attn.q_proj", 334)]
DOSES = [0, 0.5, 1, 2, 3, 4, 6, 8]


def main():
    import argparse
    global SUBS
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default="", help="site:index pairs, comma separated (default: the attention subcomponents)")
    ap.add_argument("--out", default=os.path.join(RESULTS, "dose_sub.json"))
    args = ap.parse_args()
    if args.subs:
        SUBS = [(x.split(":")[0], int(x.split(":")[1])) for x in args.subs.split(",")]
    torch.set_grad_enabled(False)
    tok = Tokenizer.from_file(str(vm.TARGET_DIR / "tokenizer.json"))
    ids_of = lambda s: tok.encode(s).ids
    animal_ids = torch.tensor([ids_of(" " + w)[0] for w in ANIMALS])
    model = vm.load_target("cpu")
    raw = torch.load(str(vm.VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    UV = {(s, c): (raw[f"_components.{s.replace('.', '-')}.U"][c].float(), raw[f"_components.{s.replace('.', '-')}.V"][:, c].float())
          for s, c in SUBS}
    del raw
    na, nd = len(ANIMALS), len(DOSES)
    res = {}
    for site, c in SUBS:
        u, v = UV[(site, c)]
        L = [[] for _ in DOSES]
        for frame, middle in itertools.product(FRAMES, MIDDLES):
            pre, mid, cue = ids_of(frame), ids_of(middle), ids_of(" " + frame)
            x, c0 = len(pre), len(pre) + 1 + len(mid)
            seq = torch.tensor([pre + [int(i)] + mid + cue for i in animal_ids])
            T = seq.shape[1]
            ret = torch.ones(T, T, dtype=torch.bool).tril(); ret[c0:, x] = False
            Vb = v[:, None].repeat(1, nd * na)
            Ub = torch.cat([(1 - a) * u[None].repeat(na, 1) for a in DOSES])      # forward subtracts (h.v) Ub
            logits = pd4l.forward(model, seq.repeat(nd, 1), ret[None, None], (site, Vb, Ub, torch.ones(T)))
            lp = torch.log_softmax(logits.float(), -1)[:, animal_ids].numpy()
            for k in range(nd):
                L[k].append(lp[k * na:(k + 1) * na])
        chosen = np.tile(np.arange(na), len(FRAMES) * len(MIDDLES))
        rows = []
        for a, Lk in zip(DOSES, L):
            M = np.concatenate(Lk)
            rank = (M > M[np.arange(len(chosen)), chosen][:, None]).sum(1) + 1
            rows.append({"dose": a, "raise": float(raises(M, chosen).mean()), "top1": float(np.mean(rank == 1))})
        res[f"{site}#{c}"] = rows
        print(f"{site} #{c}: " + "  ".join(f"a={r['dose']:g}: {r['raise']:+.3f} ({100 * r['top1']:.1f}%)" for r in rows), flush=True)
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
