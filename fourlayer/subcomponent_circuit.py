"""The circuit of VPD subcomponents behind hidden-word recall in Goodfire's 4-layer model, from measured removals.

Text: frame + " X" + later tokens + cue; the cue cannot attend to X ("retained"), so X reaches the answer only through
the later tokens.  A node is a subcomponent u v^T of one weight matrix at one class of positions (frame, X, later
tokens, cue); removing it subtracts (x.v) u from that matrix's output at those positions only, everything later
recomputed.
  1. candidates: VPD's causal-importance network, run on the same texts (unmasked, as it was trained), lists the
     subcomponents with mean importance above --ci at each position class (a filter only, not evidence: the network
     sees the whole text in both directions);
  2. each candidate is removed alone; recall lost = the unedited model's raise (the hidden word's log P minus its mean
     within the template, nats; hidden_span.template_measures) minus the raise with the node removed, on screening
     templates, and the largest losses and the largest gains are measured again on all 48 templates; with --kl-top, the
     strongest of each are also removed at every position of held-out Pile text and their KL(unedited || removed)
     measured, to tell circuit parts from general disruptors;
  3. the circuit is the smallest set of nodes, taken in order of their single losses, whose joint removal takes recall
     to within --rest of chance; random sets of candidates of the same size are removed for comparison;
  4. edges: each circuit node is removed and every later circuit node's activity (x.v where it acts) is measured; the
     change, relative to that node's mean |activity|, is the edge (a total effect, through every path).
With --screen-only, steps 3-4 are skipped.
usage: subcomponent_circuit.py --out subcomponent_circuit.json
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
from hidden_span import ANIMALS, FRAMES, MIDDLES, RESULTS, template_measures, vm
from tokenizers import Tokenizer

CLASSES = ("frame", "word", "later", "cue")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", type=float, default=0.1, help="mean causal importance a candidate needs at a position class")
    ap.add_argument("--screen", type=int, default=12, help="templates used to screen candidates")
    ap.add_argument("--confirm", type=int, default=60, help="candidates measured again on all templates")
    ap.add_argument("--rest", type=float, default=0.05, help="joint removal must leave this share of the raise or less")
    ap.add_argument("--gains", type=int, default=20, help="largest gains measured again on all templates")
    ap.add_argument("--kl-top", type=int, default=10, help="Pile KL of this many of the largest losses and gains")
    ap.add_argument("--screen-only", action="store_true")
    ap.add_argument("--nodes", default="", help="JSON list of nodes site#idx@class: skip the screen, measure these")
    ap.add_argument("--out", default=os.path.join(RESULTS, "subcomponent_circuit.json"))
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    tok = Tokenizer.from_file(str(vm.TARGET_DIR / "tokenizer.json"))
    ids_of = lambda s: tok.encode(s).ids
    animal_ids = torch.tensor([ids_of(" " + w)[0] for w in ANIMALS])
    model = vm.load_target("cpu")
    vpd = vm.load_vpd(model, "cpu")
    L, H, D = model.n_layer, model.n_head, model.hd
    UV = {n: (model.site(n).U, model.site(n).V) for n in vpd.names}      # U [C, d_out], V [d_in, C]

    templates = []
    for frame, middle in itertools.product(FRAMES, MIDDLES):
        pre, mid, cue = ids_of(frame), ids_of(middle), ids_of(" " + frame)
        seq = torch.tensor([pre + [int(i)] + mid + cue for i in animal_ids])
        x, c0, T = len(pre), len(pre) + 1 + len(mid), seq.shape[1]
        pos = {"frame": list(range(0, x)), "word": [x], "later": list(range(x + 1, c0)), "cue": list(range(c0, T))}
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        mask[c0:, x] = False
        templates.append((seq, pos, mask))
    order = rng.permutation(len(templates))
    screen = [templates[i] for i in order[:a.screen]]

    def forward(seq, pos, mask, removed=(), record=None):
        """Logits at the last token with the nodes in `removed` taken out; `record` collects (site, idx, class) -> x.v."""
        rem = {}
        for site, idx, cls in removed:
            rem.setdefault(site, []).append((idx, pos[cls]))
        want = {}
        if record is not None:
            for site, idx, cls in record:
                want.setdefault(site, []).append((idx, cls))

        def lin(name, x):
            y = x @ model.site(name).W.T
            for idx, p in rem.get(name, []):
                U, V = UV[name]
                y[:, p] -= (x[:, p] @ V[:, idx])[..., None] * U[idx]
            for idx, cls in want.get(name, []):
                record[(name, idx, cls)] = (x[:, pos[cls]] @ UV[name][1][:, idx]).mean(1)
            return y

        B, T = seq.shape
        z = model.wte[seq]
        for i in range(L):
            n = lambda k: f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}"
            h = vm.rms(z, model.norms[2 * i], model.eps)
            q = model._rope(lin(n("q_proj"), h).view(B, T, H, D).transpose(1, 2), T)
            k = model._rope(lin(n("k_proj"), h).view(B, T, H, D).transpose(1, 2), T)
            v = lin(n("v_proj"), h).view(B, T, H, D).transpose(1, 2)
            att = ((q @ k.transpose(-1, -2)) / math.sqrt(D)).masked_fill(~mask, float("-inf")).softmax(-1)
            z = z + lin(n("o_proj"), (att @ v).transpose(1, 2).reshape(B, T, -1))
            h = vm.rms(z, model.norms[2 * i + 1], model.eps)
            z = z + lin(n("down_proj"), vm.gelu_tanh(lin(n("c_fc"), h)))
        return vm.rms(z, model.ln_f, model.eps)[:, -1] @ model.wte.T

    def recall(temps, removed=()):
        L_ = np.concatenate([torch.log_softmax(forward(s, p, m, removed), -1)[:, animal_ids].numpy() for s, p, m in temps])
        return template_measures(L_, np.tile(np.arange(len(ANIMALS)), len(temps)))["raise"]

    base = recall(templates)
    if a.nodes:                                                  # given nodes: their single losses and the edges between them
        parse = lambda k: (k.split("#")[0], int(k.split("#")[1].split("@")[0]), k.split("@")[1])
        circuit = [parse(k) for k in json.load(open(a.nodes))]
        confirmed = {n: base - recall(templates, [n]) for n in circuit}
        single, cands, top, joint, random_sets = dict(confirmed), circuit, circuit, [recall(templates, circuit)], []
        print(f"{len(circuit)} given nodes removed together: raise {joint[0]:.4f}", flush=True)
    else:
        circuit = None

    if circuit is None:
        # 1. candidates from the causal-importance network
        ci_sum = {}
        for seq, pos, _ in templates:
            _, ci = vpd.target_and_ci(seq)
            for site, c in ci.items():                                   # c: [B, T, C]
                for cls in CLASSES:
                    ci_sum.setdefault((site, cls), torch.zeros(c.shape[-1]))
                    ci_sum[(site, cls)] += c[:, pos[cls]].mean((0, 1)) / len(templates)
        cands = [(site, int(i), cls) for (site, cls), m in ci_sum.items() for i in torch.nonzero(m > a.ci).flatten().tolist()]
        print(f"{len(cands)} candidate nodes (mean importance > {a.ci})", flush=True)

        base_screen, base = recall(screen), recall(templates)
        print(f"raise: screening templates {base_screen:.4f}, all {base:.4f} nats", flush=True)
        single = {}
        for j, node in enumerate(cands):
            single[node] = base_screen - recall(screen, [node])
            if j % 200 == 0:
                print(f"  screened {j}/{len(cands)}", flush=True)
        top = sorted(cands, key=lambda n: -single[n])[:a.confirm]
        gains = sorted(cands, key=lambda n: single[n])[:a.gains]
        confirmed = {n: base - recall(templates, [n]) for n in top + gains}
        ranked = sorted(top, key=lambda n: -confirmed[n])
        for n in ranked[:20]:
            print(f"  {n[0]}#{n[1]} at {n[2]}: raise lost {confirmed[n]:+.4f} nats ({confirmed[n] / base:+.0%})", flush=True)
        for n in sorted(gains, key=lambda n: confirmed[n])[:10]:
            print(f"  {n[0]}#{n[1]} at {n[2]}: raise gained {-confirmed[n]:+.4f} nats ({-confirmed[n] / base:+.0%})", flush=True)
        kl = {}
        if a.kl_top:
            pile = vm.val_tokens(16, 256, 2048)
            with torch.no_grad():
                ref = [torch.log_softmax(model(pile[j:j + 4]).float(), -1) for j in range(0, len(pile), 4)]
            for n in ranked[:a.kl_top] + sorted(gains, key=lambda n: confirmed[n])[:a.kl_top]:
                U, V = UV[n[0]]
                hook = model.site(n[0]).register_forward_hook(
                    lambda m, inp, out, u=U[n[1]], v=V[:, n[1]]: out - (inp[0] @ v)[..., None] * u)
                with torch.no_grad():
                    tot = sum(float((r.exp() * (r - torch.log_softmax(model(pile[j:j + 4]).float(), -1))).sum(-1).sum())
                              for r, j in zip(ref, range(0, len(pile), 4)))
                hook.remove()
                kl[n] = tot / pile.numel()
                print(f"  KL of removing {n[0]}#{n[1]} everywhere: {kl[n]:.4f} nats per token", flush=True)
        if a.screen_only:
            key = lambda n: f"{n[0]}#{n[1]}@{n[2]}"
            json.dump({"raise": base, "n_candidates": len(cands), "ci_threshold": a.ci,
                       "single_screen": {key(n): single[n] for n in cands}, "single": {key(n): confirmed[n] for n in confirmed},
                       "kl_everywhere": {key(n): v for n, v in kl.items()}}, open(a.out, "w"), indent=1)
            return

        # 3. the smallest set, in order of single losses, whose joint removal reaches chance
        circuit, joint = [], []
        for n in ranked:
            circuit.append(n)
            r = recall(templates, circuit)
            joint.append(r)
            print(f"  first {len(circuit)} removed together: raise {r:.4f}", flush=True)
            if r <= a.rest * base:
                break
        random_sets = [recall(templates, [cands[i] for i in rng.choice(len(cands), len(circuit), replace=False)]) for _ in range(5)]
        print(f"circuit of {len(circuit)} nodes: removed together raise {joint[-1]:.4f}; random sets of {len(circuit)}: "
              + ", ".join(f"{r:.4f}" for r in random_sets), flush=True)

    # 4. edges: remove each circuit node, measure every later circuit node's activity
    layer = lambda n: int(n[0].split(".")[1])
    stage = {"q_proj": 0, "k_proj": 0, "v_proj": 0, "o_proj": 1, "c_fc": 2, "down_proj": 3}
    rank = lambda n: (layer(n), stage[n[0].split(".")[-1]])
    base_act = {n: [] for n in circuit}
    for s, p, m in templates:
        rec = {n: None for n in circuit}
        forward(s, p, m, (), rec)
        for n in circuit:
            base_act[n].append(rec[n])
    base_act = {n: torch.cat(v) for n, v in base_act.items()}
    edges = {}
    for src in circuit:
        later_nodes = [n for n in circuit if rank(n) > rank(src) or (rank(n) == rank(src) and CLASSES.index(n[2]) > CLASSES.index(src[2]))]
        if not later_nodes:
            continue
        after = {n: [] for n in later_nodes}
        for s, p, m in templates:
            rec = {n: None for n in later_nodes}
            forward(s, p, m, [src], rec)
            for n in later_nodes:
                after[n].append(rec[n])
        for n in later_nodes:
            b, e = base_act[n], torch.cat(after[n])
            edges[f"{src[0]}#{src[1]}@{src[2]} -> {n[0]}#{n[1]}@{n[2]}"] = float((e - b).mean() / b.abs().mean().clamp_min(1e-9))
    key = lambda n: f"{n[0]}#{n[1]}@{n[2]}"
    out = {"recall": base, "n_candidates": len(cands), "ci_threshold": a.ci,
           "single_screen": {key(n): single[n] for n in cands}, "single": {key(n): confirmed[n] for n in top},
           "circuit": [key(n) for n in circuit], "joint": joint, "random_sets": random_sets,
           "activity": {key(n): float(base_act[n].mean()) for n in circuit}, "edges": edges}
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
