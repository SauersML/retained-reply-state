"""The Goodfire 4-layer figure (make_figure.sh), Goodfire's 4-layer model only, three rows.
  a-b  the test's three conditions in plain text, and in each the raise of the hidden word's log P at recall within its
       template (nats), against the same with the hidden words re-paired at random (main figure: j, and b's last column)
  c-d  where the hidden word's information flows: recall lost when one block's attention or MLP output at one position
       is set to its mean over the template's words (fourlayer/mlp_routes.py); the fewest VPD subcomponents that make
       the hidden word the top answer (main figure: i)
  e    the recall circuit: two attention steps and the VPD subcomponents measured to switch them (main figure: k)
usage: make_figure.sh
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from figure_main import BLUE, CLOUD, INK, SLATE, W, Canvas, row_circuit, row_fourlayer, strip_raise, text_schematic

ROWS = [10.6, 10.6, 12.9]                  # heights of the three rows
H = sum(ROWS)
FLOW_COLS = [("word", "the word\n“fox”"), ("later", "later\ntokens"), ("cue", "where it is\nrecalled")]


def flow_maps(fig, axes, cax, cols=FLOW_COLS):
    """Recall lost (%) when one block's attention output (axes[0]) or MLP output (axes[1]) at one position class is set
    to its mean over the template's words, layers 0-3 by position; returns the colorbar."""
    r = json.load(open("results/fourlayer/mlp_routes.json"))["blocks"]
    cmap = LinearSegmentedColormap.from_list("lost", ["white", "#c9dbf0", BLUE, INK])
    for ax, (kind, name) in zip(axes, (("attn", "attention output"), ("mlp", "MLP output"))):
        M = np.array([[max(0.0, -100 * r[f"{l}.{kind}@{c}"]["raise"]) for c, _ in cols] for l in range(4)])
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto", origin="lower")
        for l in range(4):
            for j in range(3):
                v = M[l, j]
                ax.text(j, l, f"−{v:.0f}%" if v >= 0.5 else "0", ha="center", va="center", fontsize=21,
                        color="white" if v > 55 else INK)
        ax.set_xticks(range(3), [lab for _, lab in cols], fontsize=17)
        ax.set_yticks(range(4), [f"layer {l}" for l in range(4)] if kind == "attn" else [""] * 4, fontsize=19)
        ax.set_title(name, fontsize=22, pad=12)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("recall lost without the block's\nword-specific output (%)", fontsize=17)
    cb.outline.set_visible(False)
    return cb


def row_test(cv, text4l, y0):
    """Panels a-b, occupying [y0, y0 + ROWS[0]]: the three conditions, each next to its result."""
    top = y0 + ROWS[0]
    text_schematic(cv, 0.6, top, letter="a", title="Goodfire 4-layer model: which tokens may read the hidden word",
                   key_at=(4.3, top - 9.75, 6.4, 0))
    cv.letter(17.2, top - 0.75, "b")
    cv.S.text(18.0, top - 0.75, "how much the hidden word's log P rises at recall", fontsize=25, weight="bold", va="center")
    ky = top - 9.75                                                 # on the line of panel a's key
    cv.S.scatter([18.2], [ky], s=130, color=BLUE, edgecolor="white")
    cv.S.text(18.5, ky, "this word was hidden", fontsize=22, color=BLUE, va="center")
    cv.S.add_patch(plt.Rectangle((22.2, ky - 0.2), 0.4, 0.4, facecolor=CLOUD, edgecolor="none"))
    cv.S.text(22.8, ky, "another word was hidden", fontsize=22, color=SLATE, va="center")
    for k, name in enumerate(("stripped", "retained", "visible")):
        y = top - 3.3 - 2.55 * k                                    # the schematic's row k
        ax = cv.axes(19.6, y - 0.55, 4.6, 2.05)
        strip_raise(ax, text4l, name, (-3, 16), [0, 0.1, 1, 10], name == "retained")
        ax.tick_params(labelsize=22)
        if name == "stripped":
            ax.text(1.06, 0.3, "identical in every run,\nso exactly 0", transform=ax.transAxes, fontsize=20, color=SLATE,
                    va="center")
        if name == "retained":
            ax.set_ylabel("hidden word's\nlog P raised (nats)", fontsize=20, labelpad=6)


def row_flow_edit(cv, refits, y0):
    """Panels c-d, occupying [y0, y0 + ROWS[1]]."""
    top = y0 + ROWS[1]
    cv.letter(0.1, top - 0.75, "c")
    cv.S.text(0.9, top - 0.75, "where the hidden word's information flows", fontsize=25, weight="bold", va="center")
    h = ROWS[1] - 4.9
    cb = flow_maps(cv.fig, [cv.axes(2.3, y0 + 2.3, 4.3, h), cv.axes(6.85, y0 + 2.3, 4.3, h)], cv.axes(11.45, y0 + 2.3, 0.22, h))
    cb.ax.tick_params(labelsize=17)
    row_fourlayer(cv, refits, y0, x0=13.6, rh=ROWS[1], letter="d",
                  title="rescaling {k} VPD subcomponents makes the hidden\nword the top answer {top:.0f}% of the time "
                        "(from {base:.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text4l", nargs="+", required=True, help="fourlayer/hidden_span.py outputs")
    ap.add_argument("--refit", nargs="+", required=True, help="fourlayer/optimize_edit.py --support outputs")
    ap.add_argument("--parts", required=True, help="fourlayer/attention_gates.py output")
    ap.add_argument("--out", default="figs/goodfire.png")
    a = ap.parse_args()
    fig = plt.figure(figsize=(W, H))
    cv = Canvas(fig, W, H)
    row_test(cv, a.text4l, sum(ROWS[1:]))
    row_flow_edit(cv, a.refit, ROWS[2])
    row_circuit(cv, a.parts, 0, rh=ROWS[2], letter="e",
                title="the recall circuit (recall −{share:.0f}% with its two main attention steps blocked)")
    fig.savefig(a.out, dpi=100)


if __name__ == "__main__":
    main()
