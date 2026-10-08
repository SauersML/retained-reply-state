"""Further figures from the saved float32 results (figs/*.png), each self-contained.

documents_head   Qwen3-1.7B (held-out runs) and Qwen3-0.6B: for each question, the hidden animal's rank (share of other animals it is ranked
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
from figure_goodfire import flow_maps

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
    """Rows: Qwen3-1.7B (held-out runs) and Qwen3-0.6B; columns: the three questions.  Lines: head 21.6 working (blue)
    and switched off (gray)."""
    rng = np.random.default_rng(0)
    rows = [("Qwen3-1.7B", HELD, "recall_B", (33, 82)), ("Qwen3-0.6B", "results/fp32/q06A_allq.npz", "preB_askB", (8, 68))]
    fig, axes = plt.subplots(2, 3, figsize=(20, 13.5))
    fig.subplots_adjust(left=0.1, right=0.84, bottom=0.1, top=0.86, wspace=0.12, hspace=0.42)
    for (model, spec, qb, ylim), row in zip(rows, axes):
        qs = [("retained", "cpu_A", "doc_A", "\u201cWhich animal did you choose?\u201d"),
              (qb, "cpu_B", "doc_B", "\u201cRecall, introspect, or reconstruct\u2026\u201d"),
              ("neutral", "cpu_neutral", "doc_neutral", "control: \u201cName one animal\u2026\u201d")]
        for ax, (q0, qc, qd, title) in zip(row, qs):
            for edit, col in (("none", BLUE), ("21:6*0", SLATE)):
                Ls = [saved_logp(spec, f"{edit}|{q}") for q in (q0, qc, qd)]
                c = Ls[0][1]
                m = [mean_ci(L, c, rng) for L, _ in Ls]
                ax.plot(range(3), [v[0] for v in m], "-", color=col, lw=3.2, zorder=2)
                for x, (mu, lo, hi) in enumerate(m):
                    ax.plot([x, x], [lo, hi], color=col, lw=2.2, zorder=2)
                    ax.scatter([x], [mu], s=170, color=col, edgecolor="white", lw=1.5, zorder=3)
                g = [centred_logp(L, c) for L, _ in Ls]
                up = edit != "none"                 # the gray line lies above the blue one in every panel: labels outside
                for k, x in enumerate((1, 2)):
                    p = p_paired(g[k + 1] - g[0], rng)
                    ax.text(x if up else x + 0.07, (m[k + 1][2] + 0.6) if up else (m[k + 1][1] - 0.6), p_text(p), fontsize=14,
                            ha="center" if up else "left", color=col if p < 0.05 else CLOUD, va="bottom" if up else "top",
                            bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=4)
            ax.axhline(50, color=SLATE, lw=1.4, ls=(0, (4, 3)), zorder=0)
            ax.set_xticks(range(3), ["question\nalone", "CPU\nexplainer\nfirst", "Janus's\nexplainer\nfirst"], fontsize=16)
            ax.set_xlim(-0.35, 2.6)
            ax.set_ylim(*ylim)
            ax.set_title(title, fontsize=18, pad=12)
            ax.tick_params(axis="x", length=0)
            if ax is not row[0]:
                ax.set_yticklabels([])
        row[0].set_ylabel(f"{model}\n\nhidden animal ranked above\nanother animal (%)", fontsize=19)
    axes[0, -1].text(2.75, 72, "head 21.6 switched off", color=SLATE, fontsize=18, va="center")
    axes[0, -1].text(2.75, 57, "head 21.6 working", color=BLUE, fontsize=18, va="center")
    fig.text(0.1, 0.94, "a document before the question acts partly through head 21.6: in Qwen3-1.7B it eases the head's\n"
             "suppression of the hidden animal; in Qwen3-0.6B it strengthens it (switching the head off halves the drop)",
             fontsize=21, weight="bold", va="center")
    fig.savefig(out, dpi=110, facecolor="white")


def flow(out="figs/flow4l.png"):
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.6))
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.2, top=0.82, wspace=0.08)
    flow_maps(fig, axes, fig.add_axes([0.88, 0.2, 0.015, 0.62]),
              [("word", "hidden word\n\u201cfox\u201d"), ("later", "later tokens\n\u201c. Nobody else knows.\u201d"),
               ("cue", "where it is recalled\n\u201cMy pet is a\u201d")])
    fig.text(0.08, 0.93, "Goodfire 4-layer model: where the hidden word's information flows", fontsize=24, weight="bold")
    fig.savefig(out, dpi=110, facecolor="white")


WORDING = [("preA_askA", "\u201cWhich animal did you choose?\u201d"),
           ("preA_introspect_which", "\u201cIntrospect: which animal did you choose?\u201d"),
           ("preA_recall", "\u201cRecall which animal you chose\u201d"),
           ("preA_introspect", "\u201cIntrospect which animal you chose\u201d"),
           ("preA_reconstruct", "\u201cReconstruct which animal you chose\u201d"),
           ("preA_askB", "\u201cRecall, introspect, or reconstruct\u2026\u201d"),
           ("preB_askA", "introspective preamble + \u201cWhich animal\u2026?\u201d"),
           ("preB_askB", "introspective preamble + \u201cRecall, introspect,\nor reconstruct\u2026\u201d")]
ATT_SOURCES = {"A": (HELD, "retained"), "B": (HELD, "recall_B"), "neutral": (HELD, "neutral"), "doc_A": (HELD, "doc_A"),
               "cpu_A": (HELD, "cpu_A"), "doc_B": (HELD, "doc_B"), "cpu_B": (HELD, "cpu_B"), "doc_neutral": (HELD, "doc_neutral"),
               "cpu_neutral": (HELD, "cpu_neutral")}


def wording(out="figs/wording.png", word="results/fp32/wording_heldout_mac.npz+results/fp32/wording_discovery.npz",
            att="results/fp32/attention_all.npz", att_word="results/fp32/wording_heldout_mac.npz"):
    """Left: recall for each wording, head 21.6 working and switched off (both run sets, 1,197 runs; p: paired against the
    plain question).  Right: per question, head 21.6's attention from the answer position to the reply against recall."""
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(22, 9.6))
    ax = fig.add_axes([0.25, 0.14, 0.33, 0.68])
    base = {e: saved_logp(word, f"{e}|preA_askA") for e in ("none", "21:6*0")}
    shift = {"none": [], "21:6*0": []}
    for i, (k, lab) in enumerate(WORDING):
        y = len(WORDING) - 1 - i
        for e, col in (("21:6*0", SLATE), ("none", BLUE)):
            L, c = saved_logp(word, f"{e}|{k}")
            mu, lo, hi = mean_ci(L, c, rng)
            ax.plot([lo, hi], [y, y], color=col, lw=2.2)
            ax.scatter([mu], [y], s=150, color=col, edgecolor="white", lw=1.4, zorder=3)
            if k != "preA_askA":
                shift[e].append(mu - per_animal(base[e][0], c).mean())
                pv = p_paired(centred_logp(L, c) - centred_logp(base[e][0], c), rng)
                ax.text(hi + 0.6, y, p_text(pv), fontsize=14, va="center", color=col if pv < 0.05 else CLOUD)
    ax.set_yticks(range(len(WORDING)), [lab for _, lab in WORDING][::-1], fontsize=16)
    ax.axvline(50, color=SLATE, lw=1.4, ls=(0, (4, 3)))
    ax.set_xlim(35, 84)
    ax.set_xlabel("hidden animal ranked above another animal (%)", fontsize=19)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.text(36, len(WORDING) - 0.35, "head 21.6 working", color=BLUE, fontsize=18)
    ax.text(64, len(WORDING) - 0.35, "switched off", color=SLATE, fontsize=18)
    fig.text(0.01, 0.93, "a", fontsize=36, weight="bold")
    fig.text(0.03, 0.93, f"every edit of the plain question raises recall by {min(shift['none']):.0f}-{max(shift['none']):.0f} points;\n"
             f"with head 21.6 switched off, by at most {max(np.abs(shift['21:6*0'])):.0f}", fontsize=21, weight="bold", va="center")

    ax = fig.add_axes([0.68, 0.14, 0.3, 0.68])
    z = np.load(att)
    g = int(z["group"])
    pts = []
    labels = {"A": "\u201cWhich animal\u2026?\u201d", "B": "introspective\nquestion", "neutral": "\u201cName one\nanimal\u2026\u201d",
              "doc_A": "Janus's explainer\n+ \u201cWhich animal\u2026?\u201d", "cpu_A": "CPU explainer\n+ \u201cWhich animal\u2026?\u201d"}
    for k, (src, arm) in list(ATT_SOURCES.items()) + [(k, (att_word, f"{k}")) for k, _ in WORDING if k not in ("preA_askA", "preB_askB")]:
        L, c = saved_logp(src, f"none|{arm}")
        a = 100 * z[k][:, 0, 6 * g:7 * g].mean()
        r = per_animal(L, c).mean()
        col = GOLD if k.startswith("cpu") else (CORAL if k.startswith("doc") else BLUE)
        ax.scatter([a], [r], s=170, color=col, edgecolor="white", lw=1.4, zorder=3)
        pts.append((a, r))
        if k in labels:
            dx, dy, ha, va = {"A": (-0.15, 0, "right", "center"), "B": (0.12, -0.5, "left", "top"),
                              "neutral": (0.12, 0.5, "left", "bottom"), "doc_A": (0.12, 0.6, "left", "bottom"),
                              "cpu_A": (0.14, 0, "left", "center")}[k]
            if k in ("doc_A", "cpu_A"):                       # crowded corner: text set apart with a leader line
                tx, ty = {"doc_A": (1.4, 61.2), "cpu_A": (1.4, 64.2)}[k]
                ax.annotate(labels[k], xy=(a, r), xytext=(tx, ty), fontsize=14, color=INK, ha="left", va="center",
                            linespacing=1.0, arrowprops=dict(arrowstyle="-", color=CLOUD, lw=1.2, shrinkA=2, shrinkB=6))
            else:
                ax.text(a + dx, r + dy, labels[k], fontsize=14, color=INK, ha=ha, va=va, linespacing=1.0)
    xs, ys = np.array(pts).T
    rho = np.corrcoef(xs, ys)[0, 1]
    ax.set_xlabel("head 21.6's attention to the reply\n(% of the answer position's attention)", fontsize=19)
    ax.set_ylabel("hidden animal ranked above\nanother animal (%)", fontsize=19)
    ax.axhline(50, color=SLATE, lw=1.4, ls=(0, (4, 3)), zorder=0)
    for t, col, yy in (("question wordings", BLUE, 0.16), ("Janus's explainer first", CORAL, 0.1), ("CPU explainer first", GOLD, 0.04)):
        ax.text(0.98, yy + 0.8, t, transform=ax.transAxes, ha="right", color=col, fontsize=16)
    fig.text(0.62, 0.93, "b", fontsize=36, weight="bold")
    fig.text(0.64, 0.93, f"the less head 21.6 reads the reply, the higher\nrecall (15 questions, correlation {rho:.2f})",
             fontsize=21, weight="bold", va="center")
    fig.text(0.03, 0.995, "Qwen3-1.7B (a: both run sets; b: held-out runs)", fontsize=16, color=SLATE, va="top")
    fig.savefig(out, dpi=110, facecolor="white")


def runs(out="figs/head216_runs.png", att="results/fp32/attention_all.npz"):
    """Per run (held-out Qwen3-1.7B runs): the change in head 21.6's attention to the reply between the plain and the
    introspective question, against the change in the hidden animal's log P (each relative to that animal's mean over
    runs); line: mean log P change in deciles of the attention change."""
    from scipy.stats import spearmanr
    z = np.load(att)
    g = int(z["group"])
    a = lambda k: 100 * z[k][:, 0, 6 * g:7 * g].mean(1)
    L0, c = saved_logp(HELD, "none|retained")
    L1, _ = saved_logp(HELD, "none|recall_B")
    dx, dy = a("B") - a("A"), centred_logp(L1, c) - centred_logp(L0, c)
    fig, ax = plt.subplots(figsize=(12, 8.4))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.17, top=0.8)
    ax.scatter(dx, dy, s=26, color=BLUE, alpha=0.45, edgecolor="none")
    edges = np.percentile(dx, np.linspace(0, 100, 11))
    mids = [(dx[(dx >= lo) & (dx <= hi)].mean(), dy[(dx >= lo) & (dx <= hi)].mean()) for lo, hi in zip(edges[:-1], edges[1:])]
    ax.plot(*zip(*mids), "-o", color=INK, lw=3, ms=8)
    ax.axhline(0, color=SLATE, lw=1.3, ls=(0, (4, 3)))
    ax.set_xlabel("change in head 21.6's attention to the reply\n(percentage points)", fontsize=19)
    ax.set_ylabel("change in the hidden animal's log P (nats)", fontsize=19)
    rho = spearmanr(dx, dy).correlation
    fig.text(0.03, 0.91, f"Qwen3-1.7B, run by run: the more the introspective question lowers head 21.6's attention\n"
             f"to the reply, the more it raises the hidden animal (rank correlation {rho:.2f}, 599 runs)", fontsize=19,
             weight="bold", va="center")
    fig.savefig(out, dpi=110, facecolor="white")


if __name__ == "__main__":
    {"documents_head": documents_head, "flow": flow, "wording": wording, "runs": runs}[sys.argv[1]]()
