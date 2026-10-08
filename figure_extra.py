"""Further figures from the saved float32 results (figs/*.png), each self-contained.

documents_head   Qwen3-1.7B, held-out runs: for each question, the hidden animal's rank (share of other animals it is ranked
                 above, 50% = chance) with the question alone, after a CPU explainer and after Janus's LLM explainer, with
                 head 21.6 on and switched off.  Points: mean over hidden animals, bars: 95% interval from resampling hidden
                 animals; p: sign flips of each run's paired change in the hidden animal's centered log P.
wording          the same for minimal edits between the plain and the introspective recall question (wording.py)
attention        head 21.6's attention to the reply under each question, against the recall it gives (attention_runs.py)
flow             Goodfire's 4-layer model: for each layer and position, how much of the hidden word's raise is lost when
                 that block's attention or MLP output is set to its mean over the template's words (mlp_routes.py)
usage: figure_extra.py documents_head|wording|attention|flow
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_main import BLUE, CORAL, GOLD, INK, SLATE, CLOUD, centred_logp, p_paired, per_animal, saved_logp

plt.rcParams.update({"font.family": "Avenir Next", "font.size": 20, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.4, "xtick.major.width": 1.4, "ytick.major.width": 1.4})
HELD = "results/fp32/q17A_heldout.npz"


def mean_ci(L, c, rng, B=2000):
    """Mean over hidden animals of the per-animal rank, and its 95% interval from resampling hidden animals."""
    a = per_animal(L, c)
    boots = a[rng.integers(0, len(a), (B, len(a)))].mean(1)
    return a.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def p_text(p):
    return "p<0.001" if p < 1e-3 else f"p={p:.2g}"


def documents_head(out="figs/documents_head.png"):
    rng = np.random.default_rng(0)
    qs = [("retained", "cpu_A", "doc_A", "“Which animal did you choose?”"),
          ("recall_B", "cpu_B", "doc_B", "“Recall, introspect, or reconstruct…”"),
          ("neutral", "cpu_neutral", "doc_neutral", "control: “Name one animal…”")]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.4), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.84, bottom=0.17, top=0.8, wspace=0.12)
    for ax, (q0, qc, qd, title) in zip(axes, qs):
        for edit, col, name in (("none", BLUE, "head 21.6 working"), ("21:6*0", SLATE, "head 21.6 switched off")):
            Ls = [saved_logp(HELD, f"{edit}|{q}") for q in (q0, qc, qd)]
            c = Ls[0][1]
            m = [mean_ci(L, c, rng) for L, _ in Ls]
            xs = np.arange(3)
            ax.plot(xs, [v[0] for v in m], "-", color=col, lw=3.2, zorder=2)
            for x, (mu, lo, hi) in zip(xs, m):
                ax.plot([x, x], [lo, hi], color=col, lw=2.2, zorder=2)
                ax.scatter([x], [mu], s=170, color=col, edgecolor="white", lw=1.5, zorder=3)
            g = [centred_logp(L, c) for L, _ in Ls]
            for k, x in enumerate((1, 2)):                   # gray labels above the gray points, blue below the blue
                p = p_paired(g[k + 1] - g[0], rng)
                up = edit != "none"
                ax.text(x if up else x + 0.07, (m[k + 1][2] + 0.6) if up else (m[k + 1][1] - 0.6), p_text(p), fontsize=15,
                        ha="center" if up else "left", color=col if p < 0.05 else CLOUD, va="bottom" if up else "top")
        ax.axhline(50, color=SLATE, lw=1.4, ls=(0, (4, 3)), zorder=0)
        ax.set_xticks(range(3), ["question\nalone", "CPU\nexplainer\nfirst", "Janus's\nexplainer\nfirst"], fontsize=17)
        ax.set_xlim(-0.35, 2.6)
        ax.set_title(title, fontsize=19, pad=14)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("hidden animal ranked above\nanother animal (%)", fontsize=20)
    axes[0].set_ylim(33, 82)
    axes[-1].text(2.7, 72, "head 21.6 switched off", color=SLATE, fontsize=19, va="center")
    axes[-1].text(2.7, 57, "head 21.6 working", color=BLUE, fontsize=19, va="center")
    fig.text(0.07, 0.94, "Qwen3-1.7B: a document before the question raises recall mainly by easing head 21.6's suppression",
             fontsize=24, weight="bold")
    fig.savefig(out, dpi=110, facecolor="white")


def flow(out="figs/flow4l.png"):
    from matplotlib.colors import LinearSegmentedColormap
    r = json.load(open("results/fourlayer/mlp_routes.json"))["blocks"]
    cols = [("word", "hidden word\n\u201cfox\u201d"), ("later", "later tokens\n\u201c. Nobody else knows.\u201d"),
            ("cue", "where it is recalled\n\u201cMy pet is a\u201d")]
    cmap = LinearSegmentedColormap.from_list("lost", ["white", "#c9dbf0", BLUE, INK])
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.6))
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.2, top=0.82, wspace=0.08)
    for ax, (kind, name) in zip(axes, (("attn", "attention output"), ("mlp", "MLP output"))):
        M = np.array([[max(0.0, -100 * r[f"{l}.{kind}@{c}"]["raise"]) for c, _ in cols] for l in range(4)])
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto", origin="lower")
        for l in range(4):
            for j in range(3):
                v = M[l, j]
                ax.text(j, l, f"\u2212{v:.0f}%" if v >= 0.5 else "0", ha="center", va="center", fontsize=21,
                        color="white" if v > 55 else INK)
        ax.set_xticks(range(3), [lab for _, lab in cols], fontsize=17)
        ax.set_yticks(range(4), [f"layer {l}" for l in range(4)] if kind == "attn" else [""] * 4, fontsize=19)
        ax.set_title(name, fontsize=22, pad=12)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_axes([0.88, 0.2, 0.015, 0.62]))
    cb.set_label("recall lost without the block's\nword-specific output (%)", fontsize=17)
    cb.outline.set_visible(False)
    fig.text(0.08, 0.93, "Goodfire 4-layer model: where the hidden word's information flows", fontsize=24, weight="bold")
    fig.savefig(out, dpi=110, facecolor="white")


if __name__ == "__main__":
    {"documents_head": documents_head, "flow": flow}[sys.argv[1]]()
