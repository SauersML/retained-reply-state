"""VPD subcomponents behind the hidden-word readout in the 4-layer model (templates from hidden_span.py).

Every subcomponent (a rank-one slice u v^T of VPD's decomposition) of the attention sites of layers 2 and 3 is
removed exactly (output x W^T - (x.v) u, everything later recomputed) in the retained arm, at the positions where the
localized circuit uses the site:
  write (layer 2: heads at the middle tokens read the word's position)   v_proj, k_proj at the word; q_proj, o_proj
                                                                         at the middle tokens
  read  (layer 3: heads at the cue read the middle tokens)               v_proj, k_proj at the middle tokens;
                                                                         q_proj, o_proj at the cue
Drop = retained raise minus the raise with the slice removed.  Candidates: the 128 slices per site with the largest
output where the circuit uses it (mean |x.v| over those positions times the norm of u in the localized heads' block).
Screen: 8 templates, 16 removals per forward.  The 8 largest drops per site are measured again on all 48 templates (animal-level test of what remains).
Lens of those slices (weights only, as gam's bench/oracle/vpd_lens.py): its read activity on each animal token's
normed embedding, and its write's effect on each animal's logit through the final norm and the tied unembedding.
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
from stats import animal_level, raises
from tokenizers import Tokenizer

HEADS = {2: [2, 3], 3: [0, 1, 2, 4, 5]}  # localize4l.py: writers at layer 2, readers at layer 3
NCAND = 128
SITES = {"h.2.attn.v_proj": "word", "h.2.attn.k_proj": "word", "h.2.attn.q_proj": "middle", "h.2.attn.o_proj": "middle",
         "h.3.attn.v_proj": "middle", "h.3.attn.k_proj": "middle", "h.3.attn.q_proj": "cue", "h.3.attn.o_proj": "cue"}
K = 16
EVERYWHERE = [False]


def forward(model, ids, mask, remove=None):
    """Retained-arm forward; remove = (site, V_b [d_in, B], U_b [B, d_out], pos [T] bool): batch element b loses
    the slice (V_b[:, b], U_b[b]) at the given positions."""
    B, T = ids.shape
    x = model.wte[ids]
    for i in range(model.n_layer):
        def s(k, h):
            name = f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}"
            y = model.site(name)(h)
            if remove is not None and remove[0] == name:
                _, Vb, Ub, pos = remove
                coef = torch.einsum("btd,db->bt", h, Vb) * pos[None, :]
                y = y - coef[:, :, None] * Ub[:, None, :]
            return y
        h = vm.rms(x, model.norms[2 * i], model.eps)
        q, k, v = (s(n, h).view(B, T, model.n_head, model.hd).transpose(1, 2) for n in ("q_proj", "k_proj", "v_proj"))
        q, k = model._rope(q, T), model._rope(k, T)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + s("o_proj", y.transpose(1, 2).reshape(B, T, -1))
        h = vm.rms(x, model.norms[2 * i + 1], model.eps)
        x = x + s("down_proj", vm.gelu_tanh(s("c_fc", h)))
    return vm.rms(x[:, -1], model.ln_f, model.eps) @ model.wte.T


def site_input(model, ids, mask, site):
    """The input [B, T, d_in] that `site` receives in the retained-arm forward."""
    st = model.site(site)
    st.cache_input = True
    forward(model, ids, mask)
    st.cache_input = False
    x, st.last_input = st.last_input, None
    return x


def main():
    import argparse
    global SITES, NCAND
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="", help="site:positions pairs, comma separated (default: the attention sites)")
    ap.add_argument("--ncand", type=int, default=NCAND)
    ap.add_argument("--confirm", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(RESULTS, "pd4l.json"))
    ap.add_argument("--everywhere", action="store_true", help="delete the subcomponent at every position (a weight edit)")
    a = ap.parse_args()
    if a.sites:
        SITES = dict(x.split(":") for x in a.sites.split(","))
    NCAND = a.ncand
    EVERYWHERE[0] = a.everywhere
    torch.set_grad_enabled(False)
    tok = Tokenizer.from_file(str(vm.TARGET_DIR / "tokenizer.json"))
    ids_of = lambda s: tok.encode(s).ids
    animal_ids = torch.tensor([ids_of(" " + w)[0] for w in ANIMALS])
    model = vm.load_target("cpu")
    raw = torch.load(str(vm.VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    UV = {n: (raw[f"_components.{n.replace('.', '-')}.U"].float(), raw[f"_components.{n.replace('.', '-')}.V"].float()) for n in SITES}
    del raw
    templates = []
    for frame, middle in itertools.product(FRAMES, MIDDLES):
        pre, mid, cue = ids_of(frame), ids_of(middle), ids_of(" " + frame)
        x, m0, c0 = len(pre), len(pre) + 1, len(pre) + 1 + len(mid)
        seq = torch.tensor([pre + [int(i)] + mid + cue for i in animal_ids])
        T = seq.shape[1]
        ret = torch.ones(T, T, dtype=torch.bool).tril()
        ret[c0:, x] = False
        pos = {"word": torch.zeros(T), "middle": torch.zeros(T), "cue": torch.zeros(T)}
        pos["word"][x] = 1; pos["middle"][m0:c0] = 1; pos["cue"][c0:] = 1
        if EVERYWHERE[0]:
            pos = {k: torch.ones(T) for k in pos}
        templates.append((seq, ret, pos))
    screen = templates[::6]
    na = len(ANIMALS)
    readout = lambda logits: torch.log_softmax(logits.float(), -1)[:, animal_ids].numpy()

    def run(temps, site=None, comps=()):
        """Readout L [templates * animals, animals] per removed slice (or none)."""
        out = [[] for _ in range(max(1, len(comps)))]
        for seq, ret, pos in temps:
            if site is None:
                out[0].append(readout(forward(model, seq, ret[None, None])))
                continue
            U, V = UV[site]
            c = torch.tensor(comps).repeat_interleave(na)
            L = readout(forward(model, seq.repeat(len(comps), 1), ret[None, None], (site, V[:, c], U[c], pos[SITES[site]])))
            for g in range(len(comps)):
                out[g].append(L[g * na:(g + 1) * na])
        return [np.concatenate(o) for o in out]

    chosen_s = np.tile(np.arange(na), len(screen))
    chosen_all = np.tile(np.arange(na), len(templates))
    base_L = run(screen)[0]
    base_s = raises(base_L, chosen_s).mean()
    L_all = run(templates)[0]
    base_all = raises(L_all, chosen_all).mean()
    print(f"retained raise: screen {base_s:+.4f}, all templates {base_all:+.4f}", flush=True)
    rng = np.random.default_rng(0)
    res = {"base_raise": float(base_all), "screen_base": float(base_s), "sites": {}}
    g_f = model.ln_f.float()
    E = model.wte[animal_ids].float()
    E_read = {}
    for site in SITES:
        U, V = UV[site]
        C = U.shape[0]
        layer = int(site.split(".")[1])
        # candidate slices: largest output at the circuit's positions and heads
        acts = []
        for seq, ret, pos in screen:
            hs = site_input(model, seq, ret[None, None], site)
            acts.append((hs[:, pos[SITES[site]] > 0] @ V).abs().mean(1))   # [animals, C]
        act = torch.stack(acts).mean(0).max(0).values
        blk = torch.zeros(model.n_head, dtype=torch.bool); blk[HEADS.get(layer, list(range(model.n_head)))] = True
        if ".mlp." in site:
            size = act * U.norm(dim=1)
        elif site.endswith("o_proj"):
            size = act * U.norm(dim=1) * (V.view(model.n_head, model.hd, C)[blk].reshape(-1, C).norm(dim=0) / V.norm(dim=0))
        else:
            size = act * U.view(C, model.n_head, model.hd)[:, blk].reshape(C, -1).norm(dim=1)
        cand = torch.argsort(size, descending=True)[:NCAND].tolist()
        drops = np.full(C, np.nan)
        word_drop = np.full((C, na), np.nan)
        base_word = np.array([raises(base_L, chosen_s)[chosen_s == w].mean() for w in range(na)])
        for s0 in range(0, len(cand), K):
            comps = cand[s0:s0 + K]
            for c, L in zip(comps, run(screen, site, comps)):
                r = raises(L, chosen_s)
                drops[c] = base_s - r.mean()
                word_drop[c] = base_word - np.array([r[chosen_s == w].mean() for w in range(na)])
        best_word = np.nanmax(word_drop, 1)
        print(f"   per word: largest single-slice drops " + ", ".join(
            f"#{c} {ANIMALS[int(np.nanargmax(word_drop[c]))]} {best_word[c]:+.3f} (of {base_word[int(np.nanargmax(word_drop[c]))]:+.3f})"
            for c in np.argsort(np.nan_to_num(best_word, nan=-np.inf))[::-1][:8]), flush=True)
        # how concentrated each word's raise is: share removed by the word's single best slice
        top_share = np.nanmax(word_drop, 0) / base_word
        print(f"   per word, the best single slice removes a median {np.median(top_share):.2f} of that word's raise "
              f"(max {top_share.max():.2f}, {ANIMALS[int(top_share.argmax())]})", flush=True)
        order = [c for c in np.argsort(np.nan_to_num(drops, nan=-np.inf))[::-1] if np.isfinite(drops[c])]
        print(f"{site} ({SITES[site]}): {C} slices; largest screen drops "
              + ", ".join(f"#{c} {drops[c]:+.4f}" for c in order[:8])
              + f"; sum of {len(cand)} drops {np.nansum(drops):+.3f}; largest rises " + ", ".join(f"#{c} {drops[c]:+.4f}" for c in order[-3:]), flush=True)
        norm_g = model.norms[2 * layer].float()
        kept = []
        for c in order[:a.confirm]:
            L = run(templates, site, [int(c)])[0]
            z, p = animal_level(L, chosen_all, rng, 5000)
            r = float(raises(L, chosen_all).mean())
            # lens: read on normed animal embeddings, write onto animal logits (v_proj through its layer's o_proj)
            read = (vm.rms(E, norm_g, model.eps) @ V[:, c]).numpy() if site.endswith(("v_proj", "k_proj", "q_proj")) and ".attn." in site else None
            u = U[c]
            if site.endswith("v_proj"):
                u = model.site(f"h.{layer}.attn.o_proj").W @ u
            write = ((g_f * E) @ u).numpy() if site.endswith(("v_proj", "o_proj")) else None
            copy = None
            if read is not None and write is not None:
                M = np.outer(read, write)
                copy = float(animal_level(M, np.arange(na), rng, 2000)[0])
            heads = (u.view(model.n_head, model.hd).norm(dim=1) / u.norm()).numpy() if site.endswith(("q_proj", "k_proj", "v_proj")) and False else None
            kept.append({"slice": int(c), "screen_drop": float(drops[c]), "raise_without": r, "z_without": float(z),
                         "share_removed": float((base_all - r) / base_all),
                         "write_heads": (U[c].view(model.n_head, model.hd).norm(dim=1) / U[c].norm()).numpy().round(3).tolist()
                         if site.endswith(("q_proj", "k_proj", "v_proj")) else None,
                         "read_heads": (V[:, c].view(model.n_head, model.hd).norm(dim=1) / V[:, c].norm()).numpy().round(3).tolist()
                         if site.endswith("o_proj") else None,
                         "lens_copy_z": copy,
                         "lens_write_top": [ANIMALS[j] for j in np.argsort(write)[::-1][:5]] if write is not None else None})
            print(f"   #{c}: raise without it {r:+.4f} ({100 * (base_all - r) / base_all:.1f}% of the retained raise removed), "
                  f"z {z:+.1f}; heads {kept[-1]['write_heads'] or kept[-1]['read_heads']}; lens copy z {copy}", flush=True)
        res["sites"][site] = {"positions": SITES[site], "candidates": cand, "word_drop": [[None if not np.isfinite(v) else float(v) for v in row] for row in word_drop[cand]], "screen_drops": [None if not np.isfinite(x) else float(x) for x in drops], "kept": kept}
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
