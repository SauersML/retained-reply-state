"""Figure 1.  Each row: what turn 2's cache holds in one condition (left) and what the hidden animal does to the recall
answer in that condition, per model (right): every animal's log P at recall in the runs where it was the hidden
animal (blue) or where another animal was hidden (gray), minus its average over all runs."""
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from stats import animal_level

INK, BLUE, PALE_BLUE, CORAL, PALE_CORAL = "#1d1d1f", "#2f6db5", "#dce8f6", "#e8684a", "#fbe3dc"
SLATE, PALE_SLATE, RED, CLOUD = "#8e959c", "#eceef0", "#c8102e", "#cdd2d7"
plt.rcParams.update({"font.family": "Avenir Next", "font.size": 26, "figure.facecolor": "white",
                     "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.6, "xtick.major.width": 1.6, "ytick.major.width": 1.6})

W, H = 28, 14.5
fig = plt.figure(figsize=(W, H))
S = fig.add_axes([0, 0, 1, 1])
S.set_xlim(0, W)
S.set_ylim(0, H)
S.axis("off")


def block(x, y, w, h, face, edge, text="", color=INK, style="-", fs=24, weight="normal"):
    S.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.14", facecolor=face,
                               edgecolor=edge, lw=2.5, linestyle=style))
    if text:
        S.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight=weight)


def cross(x, y, w, h):
    block(x, y, w, h, "white", "#e3a1a8", style=(0, (4, 3)))
    cx, cy, r = x + w / 2, y + h / 2, min(0.3, h / 3)
    S.plot([cx - r, cx + r], [cy - r, cy + r], color=RED, lw=5, solid_capstyle="round")
    S.plot([cx - r, cx + r], [cy + r, cy - r], color=RED, lw=5, solid_capstyle="round")


def matrix(path, arm):
    d = json.load(open(path))
    L = np.array(d["arms"][arm])
    c = np.array([d["animals"].index(x) for x in d["chosen"]])
    present = np.unique(c)
    M = np.array([L[c == a].mean(0)[present] for a in present]) - L.mean(0)[present]
    return M, L, c


def strip(ax, paths, arm, lim, ticks, pvals):
    rng = np.random.default_rng(0)
    for k, p in enumerate(paths):
        M, L, c = matrix(p, arm)
        own, other = np.diag(M), M[~np.eye(len(M), dtype=bool)]
        sub = rng.choice(other, min(600, other.size), replace=False)
        ax.scatter(k - 0.3 + rng.uniform(0, 0.24, sub.size), sub, s=10, color=CLOUD, alpha=0.9, edgecolor="none",
                   zorder=2)
        ax.scatter(k + 0.06 + rng.uniform(0, 0.22, own.size), own, s=60, color=BLUE, alpha=0.9, edgecolor="white",
                   lw=0.6, zorder=3, clip_on=False)
        ax.plot([k + 0.03, k + 0.31], [own.mean()] * 2, color=INK, lw=5, zorder=4, solid_capstyle="round")
        if pvals:
            z, pv = animal_level(L, c, np.random.default_rng(0), 20000, two_sided=True)
            ax.text(k, lim[1] * 0.97, f"p={pv:.1g}" if pv >= 1e-3 else "p<0.001", ha="center", va="top",
                    fontsize=21, color=INK if pv < 0.05 else SLATE, weight="bold" if pv < 0.05 else "normal")
    ax.axhline(0, color=SLATE, lw=1.4, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(-0.55, len(paths) - 0.45)
    ax.set_ylim(*lim)
    ax.set_yticks(ticks)
    ax.set_xticks(range(len(paths)), [])


def main(paths):
    h = 1.15
    # turn 1
    S.text(0.5, 13.55, "turn 1", fontsize=32, weight="bold", va="center")
    block(0.5, 11.75, 3.3, h, PALE_SLATE, "none", "pick an animal\nin your thinking", fs=20)
    block(4.0, 11.75, 4.9, h, PALE_CORAL, CORAL, "thinking:  …otter…", fs=25)
    S.text(6.45, 13.1, "we write a random animal here", ha="center", fontsize=21, color=CORAL, style="italic")
    block(9.1, 11.75, 3.4, h, PALE_BLUE, BLUE, "I understand.", fs=25)
    S.text(10.8, 13.1, "the model writes this", ha="center", fontsize=21, color=BLUE, style="italic")
    # turn 2 rows
    S.text(0.5, 10.25, "turn 2", fontsize=32, weight="bold", va="center")
    rows = [("stripped", SLATE), ("retained", BLUE), ("visible", CORAL)]
    ys = [8.0, 5.0, 2.0]
    for (name, col), y in zip(rows, ys):
        S.text(0.5, y + h / 2, name, fontsize=29, color=col, weight="bold", va="center")
        x = 3.0
        block(x, y, 1.15, h, PALE_SLATE, "none", "ask", fs=20)
        x += 1.3
        if name == "visible":
            block(x, y, 2.3, h, PALE_CORAL, CORAL, "…otter…", fs=23)
        else:
            cross(x, y, 2.3, h)
        x += 2.45
        if name == "stripped":
            block(x, y, 2.5, h, PALE_SLATE, SLATE, "I understand.", fs=21, style=(0, (4, 3)))
        else:
            block(x, y, 2.5, h, PALE_BLUE, BLUE, "I understand.", fs=21)
        x += 2.65
        block(x, y, 2.5, h, PALE_SLATE, "none", "which animal?", fs=21)
        S.annotate("", xy=(13.9, y + h / 2), xytext=(13.0, y + h / 2),
                   arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.5, mutation_scale=26))
    # what the boxes mean
    cross(3.0, 0.35, 1.0, 0.6)
    S.text(4.2, 0.65, "deleted", fontsize=21, va="center", color=RED)
    block(6.0, 0.35, 1.0, 0.6, PALE_SLATE, SLATE, style=(0, (4, 3)))
    S.text(7.2, 0.65, "recomputed", fontsize=21, va="center", color=SLATE)
    block(9.6, 0.35, 1.0, 0.6, PALE_BLUE, BLUE)
    S.text(10.8, 0.65, "kept from turn 1", fontsize=21, va="center", color=BLUE)

    # results, one strip per row, aligned with it
    specs = {"stripped": ((-0.25, 0.25), [-0.2, 0, 0.2], False),
             "retained": ((-0.25, 0.34), [-0.2, 0, 0.2], True),
             "visible": ((-5, 45), [0, 20, 40], False)}
    x0, w = 16.2 / W, 11.0 / W
    for (name, col), y in zip(rows, ys):
        ax = fig.add_axes([x0, (y - 0.55) / H, w, (h + 1.1) / H])
        lim, ticks, pv = specs[name]
        strip(ax, paths, name, lim, ticks, pv)
        ax.set_ylabel("nats", fontsize=22)
        if name == "visible":
            ax.set_xticks(range(len(paths)), [json.load(open(p))["model"].split("-")[-1] for p in paths])
        if name == "stripped":
            ax.text(len(paths) / 2 - 0.5, 0.12, "identical in every run, so exactly 0", ha="center", fontsize=21, color=SLATE)
    S.text(16.2, 13.55, "log P(animal) at recall, above its average", fontsize=29, weight="bold", va="center")
    S.scatter([16.4], [12.75], s=130, color=BLUE, edgecolor="white")
    S.text(16.7, 12.75, "this animal was hidden", fontsize=22, color=BLUE, va="center")
    S.add_patch(plt.Rectangle((22.0, 12.55), 0.4, 0.4, facecolor=CLOUD, edgecolor="none"))
    S.text(22.6, 12.75, "another animal was hidden", fontsize=22, color=SLATE, va="center")
    # direction: up on the top plot, down on the bottom plot
    S.annotate("", xy=(27.7, 9.9), xytext=(27.7, 7.7), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=4, mutation_scale=34))
    S.annotate("", xy=(27.7, 1.0), xytext=(27.7, 3.2), arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=4, mutation_scale=34))
    fig.savefig("paper/figs/design.png", dpi=140)


if __name__ == "__main__":
    main(sys.argv[1:])
