"""Figure 1.  Left: what turn 2's cache holds in each arm.  Right: the measured distributions, per model -- every
chosen animal's own effect (colored) against every mismatched animal pair (gray)."""
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, BLUE, PALE_BLUE, CORAL, PALE_CORAL = "#1d1d1f", "#2f6db5", "#dce8f6", "#e8684a", "#fbe3dc"
SLATE, PALE_SLATE, RED = "#8e959c", "#eceef0", "#c8102e"
plt.rcParams.update({"font.family": "Avenir Next", "font.size": 26, "figure.facecolor": "white",
                     "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.6, "xtick.major.width": 1.6, "ytick.major.width": 1.6, "hatch.linewidth": 2})


def block(ax, x, y, w, h, face, edge, text="", color=INK, style="-", fs=24, hatch=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", facecolor=face,
                                edgecolor=edge, lw=2.5, linestyle=style, hatch=hatch))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color)


def schematic(ax):
    ax.set_xlim(0, 13.2)
    ax.set_ylim(-0.2, 9.4)
    ax.axis("off")
    h = 1.0
    # turn 1
    ax.text(0, 8.85, "turn 1", fontsize=30, weight="bold", color=INK, va="center")
    block(ax, 0, 7.2, 1.5, h, PALE_SLATE, "none", "ask", fs=22)
    block(ax, 1.65, 7.2, 4.6, h, PALE_CORAL, CORAL, "thinking:  …otter…", fs=24)
    block(ax, 6.4, 7.2, 3.6, h, PALE_BLUE, BLUE, "I understand.", fs=24)
    ax.annotate("", xy=(8.2, 7.18), xytext=(4.0, 7.18), arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=3,
                connectionstyle="arc3,rad=0.35", mutation_scale=28))
    # turn 2
    ax.text(0, 5.75, "turn 2", fontsize=30, weight="bold", color=INK, va="center")
    rows = [("stripped", SLATE, False, (PALE_SLATE, SLATE)),
            ("retained", BLUE, False, (PALE_BLUE, BLUE)),
            ("visible", CORAL, True, (PALE_BLUE, BLUE))]
    for k, (name, col, think, (rf, re)) in enumerate(rows):
        y = 4.2 - k * 1.75
        ax.text(0, y + h / 2, name, fontsize=27, color=col, weight="bold", va="center")
        x = 2.6
        block(ax, x, y, 1.0, h, PALE_SLATE, "none", "ask", fs=20)
        x += 1.15
        if think:
            block(ax, x, y, 2.2, h, PALE_CORAL, CORAL, "…otter…", fs=22)
        else:
            block(ax, x, y, 2.2, h, "white", "#e3a1a8", style=(0, (4, 3)))
            cx, cy, r = x + 1.1, y + h / 2, 0.28
            ax.plot([cx - r, cx + r], [cy - r, cy + r], color=RED, lw=5, solid_capstyle="round")
            ax.plot([cx - r, cx + r], [cy + r, cy - r], color=RED, lw=5, solid_capstyle="round")
        x += 2.35
        block(ax, x, y, 2.4, h, rf, re, "I understand.", fs=20, style="-" if name != "stripped" else (0, (4, 3)))
        x += 2.55
        block(ax, x, y, 2.6, h, PALE_SLATE, "none", "which animal?", fs=20)
    # legend for the reply's two sources
    block(ax, 2.6, -0.15, 0.55, 0.5, PALE_SLATE, SLATE, style=(0, (4, 3)))
    ax.text(3.3, 0.1, "recomputed", fontsize=21, va="center", color=SLATE)
    block(ax, 6.2, -0.15, 0.55, 0.5, PALE_BLUE, BLUE)
    ax.text(6.9, 0.1, "kept from turn 1", fontsize=21, va="center", color=BLUE)


def own_vs_other(path, arm):
    d = json.load(open(path))
    L = np.array(d["arms"][arm])
    c = np.array([d["animals"].index(x) for x in d["chosen"]])
    present = np.unique(c)
    M = np.array([L[c == a].mean(0)[present] for a in present])
    M = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()
    return np.diag(M), M[~np.eye(len(M), dtype=bool)], d["model"].split("-")[-1]


def distributions(ax, paths):
    rng = np.random.default_rng(0)
    for k, p in enumerate(paths):
        own, other, name = own_vs_other(p, "retained")
        parts = ax.violinplot([other], positions=[k], widths=0.85, showextrema=False, bw_method=0.35)
        for b in parts["bodies"]:
            b.set_facecolor("#d6dade")
            b.set_edgecolor("none")
            b.set_alpha(1)
        col = BLUE
        ax.scatter(k + rng.uniform(-0.17, 0.17, own.size), own, s=70, color=col, alpha=0.85, edgecolor="white", lw=0.8, zorder=3)
        ax.plot([k - 0.3, k + 0.3], [own.mean()] * 2, color=INK, lw=4, zorder=4)
    ax.axhline(0, color=SLATE, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(range(len(paths)), [own_vs_other(p, "retained")[2] for p in paths])
    ax.set_xlim(-0.6, len(paths) - 0.4)
    lim = 0.32
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("shift in log P at recall (nats)")
    ax.set_title("retained", color=BLUE, fontsize=27, weight="bold", loc="left")
    ax.text(len(paths) - 0.45, lim * 0.92, "chosen animal", color=BLUE, ha="right", fontsize=22)
    ax.text(len(paths) - 0.45, lim * 0.80, "every other animal", color=SLATE, ha="right", fontsize=22)


def main(paths):
    fig = plt.figure(figsize=(26, 10.5))
    ax0 = fig.add_axes([0.01, 0.04, 0.53, 0.92])
    ax1 = fig.add_axes([0.62, 0.12, 0.36, 0.78])
    schematic(ax0)
    distributions(ax1, paths)
    fig.savefig("paper/figs/design.png", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1:])
