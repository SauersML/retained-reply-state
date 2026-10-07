"""The main figure, four rows.
  a-b  the experiment, and per model the share of other animals the hidden animal is ranked above at recall (each
       animal relative to its mean over runs; 50% = chance) in each condition; last column: the same three conditions in
       plain text on Goodfire's 4-layer model
  c-e  Qwen3: a linear readout of the reply's state against the model's answer (probe.py); layer 21's value maps from
       the weights alone (overlap.py, copying.py); layer 21's attention to the reply at the answer (attention.py)
  f-g  Qwen3-1.7B: each recall question against each turn-1 wording, unedited and with one head deleted from the
       weights; the deletion on other data, and other heads deleted (edit_heads.py)
  h-i  the 4-layer model: the circuit and three VPD subcomponents that set it (fourlayer/why4l.py, why4l_b.py); weight
       edits of those subcomponents against damage on Pile text (fourlayer/edit_eval.py)
usage: figure_main.py --results R.json ... --text4l T.json --probes P.json --overlap O.json --why W.json WB.json
       --sweep S.json --attention A1.json A2.json A3.json A4.json --cross figs/cross_cells.json --flip figs/flip_rows.json
"""
import argparse
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch

from stats import animal_level

INK, BLUE, PALE_BLUE, CORAL, PALE_CORAL = "#1d1d1f", "#2f6db5", "#dce8f6", "#e8684a", "#fbe3dc"
SLATE, PALE_SLATE, RED, CLOUD = "#8e959c", "#eceef0", "#c8102e", "#cdd2d7"
CMAP = LinearSegmentedColormap.from_list("cb", [CORAL, "#f7f7f5", BLUE])
plt.rcParams.update({"font.family": "Avenir Next", "font.size": 26, "figure.facecolor": "white",
                     "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.6, "xtick.major.width": 1.6, "ytick.major.width": 1.6})
W = 28.0                                   # figure width in drawing units (inches)
ROWS = [14.5, 12.5, 12.5, 12.5]            # heights of the four rows
H = sum(ROWS)


class Canvas:
    """Drawing in inch units over the whole figure; y counts up from the bottom."""

    def __init__(self, fig):
        self.fig = fig
        self.S = fig.add_axes([0, 0, 1, 1], zorder=-1)
        self.S.set_xlim(0, W)
        self.S.set_ylim(0, H)
        self.S.axis("off")

    def axes(self, x, y, w, h):
        return self.fig.add_axes([x / W, y / H, w / W, h / H])

    def block(self, x, y, w, h, face, edge, text="", color=INK, style="-", fs=24):
        self.S.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.14", facecolor=face,
                                        edgecolor=edge, lw=2.5, linestyle=style))
        if text:
            self.S.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color)

    def cross(self, x, y, w, h):
        self.block(x, y, w, h, "white", "#e3a1a8", style=(0, (4, 3)))
        cx, cy, r = x + w / 2, y + h / 2, min(0.3, h / 3)
        self.S.plot([cx - r, cx + r], [cy - r, cy + r], color=RED, lw=5, solid_capstyle="round")
        self.S.plot([cx - r, cx + r], [cy + r, cy - r], color=RED, lw=5, solid_capstyle="round")

    def letter(self, x, y, s):
        self.S.text(x, y, s, fontsize=40, weight="bold", va="center", color=INK)


def per_animal(L, c, perm=None):
    """Per hidden animal: the share of the other animals it is ranked above at recall in its runs, each animal's log P
    taken relative to its mean over all runs (50% = chance).  With perm, the hidden animals are re-paired at random."""
    D = L - L.mean(0)
    present = np.unique(c)
    lab = c if perm is None else perm[np.searchsorted(present, c)]
    own = D[np.arange(len(c)), lab]
    diff = own[:, None] - D                                     # differences under 1e-9 nats are ties (rounding)
    acc = ((diff > 1e-9).sum(1) + 0.5 * ((np.abs(diff) <= 1e-9).sum(1) - 1)) / (D.shape[1] - 1)
    return np.array([acc[c == a].mean() for a in present]) * 100


def strip(ax, paths, arm, lim, ticks, pvals):
    rng = np.random.default_rng(0)
    for k, p in enumerate(paths):
        d = json.load(open(p))
        L = np.array(d["arms"][arm])
        c = np.array([d["animals"].index(x) for x in d["chosen"]])
        own = per_animal(L, c)
        present = np.unique(c)
        null = np.concatenate([per_animal(L, c, rng.permutation(present)) for _ in range(12)])
        ax.scatter(k - 0.3 + rng.uniform(0, 0.24, null.size), null, s=10, color=CLOUD, alpha=0.9, edgecolor="none", zorder=2)
        ax.scatter(k + 0.06 + rng.uniform(0, 0.22, own.size), own, s=60, color=BLUE, alpha=0.9, edgecolor="white",
                   lw=0.6, zorder=3, clip_on=False)
        ax.plot([k + 0.03, k + 0.31], [own.mean()] * 2, color=INK, lw=5, zorder=4, solid_capstyle="round")
        if pvals:
            nulls = np.array([per_animal(L, c, rng.permutation(present)).mean() for _ in range(2000)])
            pv = (1 + np.sum(np.abs(nulls - 50) >= abs(own.mean() - 50))) / 2001
            col = SLATE if pv >= 0.05 else (BLUE if own.mean() > 50 else CORAL)
            ax.text(k, lim[1] * 0.985, f"p={pv:.1g}" if pv >= 1e-3 else "p<0.001", ha="center", va="top",
                    fontsize=21, color=col, weight="bold" if pv < 0.05 else "normal")
    ax.axhline(50, color=SLATE, lw=1.4, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(-0.55, len(paths) - 0.45)
    ax.set_ylim(*lim)
    ax.set_yticks(ticks)
    ax.set_xticks(range(len(paths)), [])


def row_experiment(cv, paths, y0, text4l=None):
    """Rows a-b, occupying [y0, y0 + ROWS[0]]."""
    h = 1.15
    top = y0 + ROWS[0]
    cv.letter(0.1, top - 0.95, "a")
    cv.S.text(0.9, top - 0.95, "turn 1", fontsize=32, weight="bold", va="center")
    cv.block(0.9, top - 2.75, 3.3, h, PALE_SLATE, "none", "pick an animal\nin your thinking", fs=20)
    cv.block(4.4, top - 2.75, 4.9, h, PALE_CORAL, CORAL, "thinking:  …otter…", fs=25)
    cv.S.text(6.85, top - 1.4, "we write a random animal here", ha="center", fontsize=21, color=CORAL, style="italic")
    cv.block(9.5, top - 2.75, 3.4, h, PALE_BLUE, BLUE, "I understand.", fs=25)
    cv.S.text(11.2, top - 1.4, "the model writes this", ha="center", fontsize=21, color=BLUE, style="italic")
    cv.S.text(0.9, top - 4.25, "turn 2", fontsize=32, weight="bold", va="center")
    rows = [("stripped", SLATE), ("retained", BLUE), ("visible", CORAL)]
    ys = [y0 + 8.0, y0 + 5.0, y0 + 2.0]
    for (name, col), y in zip(rows, ys):
        label = {"stripped": "reply\nrecomputed", "retained": "reply cache\nkept", "visible": "thinking\nkept"}[name]
        cv.S.text(0.9, y + h / 2, label, fontsize=25, color=col, weight="bold", va="center", linespacing=1.05)
        x = 3.4
        cv.block(x, y, 1.15, h, PALE_SLATE, "none", "ask", fs=20)
        x += 1.3
        if name == "visible":
            cv.block(x, y, 2.3, h, PALE_CORAL, CORAL, "…otter…", fs=23)
        else:
            cv.cross(x, y, 2.3, h)
        x += 2.45
        if name == "stripped":
            cv.block(x, y, 2.5, h, PALE_SLATE, SLATE, "I understand.", fs=21, style=(0, (4, 3)))
        else:
            cv.block(x, y, 2.5, h, PALE_BLUE, BLUE, "I understand.", fs=21)
        x += 2.65
        cv.block(x, y, 2.5, h, PALE_SLATE, "none", "which animal?", fs=21)
        cv.S.annotate("", xy=(14.3, y + h / 2), xytext=(13.4, y + h / 2),
                      arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.5, mutation_scale=26))
    cv.cross(3.4, y0 + 0.35, 1.0, 0.6)
    cv.S.text(4.6, y0 + 0.65, "deleted", fontsize=21, va="center", color=RED)
    cv.block(6.4, y0 + 0.35, 1.0, 0.6, PALE_SLATE, SLATE, style=(0, (4, 3)))
    cv.S.text(7.6, y0 + 0.65, "recomputed", fontsize=21, va="center", color=SLATE)
    cv.block(10.0, y0 + 0.35, 1.0, 0.6, PALE_BLUE, BLUE)
    cv.S.text(11.2, y0 + 0.65, "kept from turn 1", fontsize=21, va="center", color=BLUE)

    cv.letter(15.4, top - 0.95, "b")
    cv.S.text(16.2, top - 0.95, "the hidden animal, ranked above another animal at recall", fontsize=29, weight="bold", va="center")
    cv.S.scatter([16.4], [top - 1.75], s=130, color=BLUE, edgecolor="white")
    cv.S.text(16.7, top - 1.75, "this animal was hidden", fontsize=22, color=BLUE, va="center")
    cv.S.add_patch(plt.Rectangle((22.0, top - 1.95), 0.4, 0.4, facecolor=CLOUD, edgecolor="none"))
    cv.S.text(22.6, top - 1.75, "another animal was hidden", fontsize=22, color=SLATE, va="center")
    specs = {"stripped": ((18, 80), [30, 50, 70], False),
             "retained": ((18, 80), [30, 50, 70], True),
             "visible": ((18, 105), [50, 100], False)}
    specs4 = specs
    # direction: an up arrow above zero on the top plot, a down arrow below zero on the bottom plot
    arrows = {"stripped": (51, 78, BLUE, "toward"), "visible": (49, 27, CORAL, "away")}
    for (name, _), y in zip(rows, ys):
        if text4l:
            ax4 = cv.axes(24.4, y - 0.55, 3.2, h + 1.1)
            lim4, ticks4, pv4 = specs4[name]
            strip(ax4, text4l, name, lim4, ticks4, pv4)
            if name == "visible":
                ax4.set_xticks(range(len(text4l)), ["unedited", "key #224\n×8"][:len(text4l)], fontsize=20)
                ax4.set_xlabel("Goodfire 4-layer", fontsize=23, labelpad=6)
        ax = cv.axes(16.2, y - 0.55, 7.4, h + 1.1)
        lim, ticks, pv = specs[name]
        strip(ax, paths, name, lim, ticks, pv)
        if name not in arrows:
            ax.set_ylabel("%", fontsize=22)
        if name == "visible":
            ax.set_xticks(range(len(paths)), ["Qwen3\n" + json.load(open(p))["model"].split("-")[-1] for p in paths])
        if name == "stripped":
            ax.text(len(paths) / 2 - 0.5, 68, "identical in every run, so exactly chance", ha="center", fontsize=20, color=SLATE)
        if name in arrows:
            y_from, y_to, col, lab = arrows[name]
            ax.annotate("", xy=(-0.105, y_to), xytext=(-0.105, y_from), xycoords=("axes fraction", "data"),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=3.5, mutation_scale=26), annotation_clip=False)
            ax.text(-0.135, (y_from + y_to) / 2, lab, transform=ax.get_yaxis_transform(), rotation=90, ha="center",
                    va="center", fontsize=21, color=col)


def panel_flip(cv, x, y, w, h, groups):
    """Qwen3: discrimination before (gray dot) and after (arrow head) a weight edit, one line per dataset or edit.
    groups: [{"group": title, "rows": [{"label", "path" (edit_heads.py output), "spec", "arm" (default retained)}]}];
    rows whose result is not there yet are left out."""
    lines = []
    for g in groups:
        got = []
        for r in g["rows"]:
            try:
                e = json.load(open(r["path"]))["edits"]
            except FileNotFoundError:
                continue
            arm = r.get("arm", "retained")
            pick = lambda v: v[arm]["discrimination"] if arm in v else (v["discrimination"] if arm == "retained" else None)
            if r["spec"] in e and "none" in e and pick(e["none"]) is not None and pick(e[r["spec"]]) is not None:
                got.append((r["label"], 100 * pick(e["none"]), 100 * pick(e[r["spec"]]), g.get("control", False)))
        if got:
            lines.append((g["group"], None, None, None))
            lines += got
    ax = cv.axes(x, y, w, h)
    yy, ticks, labels = 0.0, [], []
    for label, a, b, control in lines:
        if a is None:
            yy -= 0.35 if ticks else 0
            ax.text(-0.025, yy, label, transform=ax.get_yaxis_transform(), fontsize=20, color=SLATE, weight="bold",
                    va="center", ha="right")
            yy -= 1.0
            continue
        col = CLOUD if control else (BLUE if b > 50 else CORAL)
        ax.annotate("", xy=(b, yy), xytext=(a, yy), arrowprops=dict(arrowstyle="-|>", color=col, lw=3.5,
                                                                      mutation_scale=24, shrinkA=0, shrinkB=0))
        ax.scatter([a], [yy], s=90, color=SLATE, zorder=3)
        ticks.append(yy)
        labels.append(label)
        yy -= 1.0
    ax.axvline(50, color=SLATE, lw=1.5, ls=(0, (4, 3)))
    ax.set_yticks(ticks, labels, fontsize=21)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(yy + 0.4, 0.6)
    ax.set_xlim(35, 80)
    ax.set_xticks([40, 50, 60, 70, 80])
    ax.set_xlabel("hidden animal ranked above\nanother animal (%)")
    return ax


def row_qwen(cv, y0, results, probes, overlap, attention):
    """Panels c-e, occupying [y0, y0 + ROWS[1]]."""
    top = y0 + ROWS[1]
    # c: what the reply's cache holds against what the answer uses
    cv.letter(0.1, top - 0.7, "c")
    cv.S.text(0.9, top - 0.7, "Qwen3: the reply's state holds the animal;\nthe answer barely uses it", fontsize=25,
              weight="bold", va="center", linespacing=1.15)
    ax = cv.axes(2.2, y0 + 1.9, 6.2, 8.4)
    pr = json.load(open(probes))
    names = []
    for k, p in enumerate(results):
        d = json.load(open(p))
        size = d["model"].split("-")[-1]
        names.append(size)
        L = np.array(d["arms"]["retained"])
        c = np.array([d["animals"].index(x) for x in d["chosen"]])
        top1 = 100 * np.mean(L.argmax(1) == c)
        probe = 100 * max(v["accuracy"] for v in pr[size.lower()].values())
        ax.plot([k, k], [top1, probe], color=CLOUD, lw=3, zorder=1)
        ax.scatter([k], [probe], s=320, color=BLUE, zorder=3, edgecolor="white", lw=2)
        ax.scatter([k], [top1], s=320, color=CORAL, zorder=3, edgecolor="white", lw=2)
    ax.axhline(2, color=SLATE, lw=1.5, ls=(0, (4, 3)))
    ax.text(0.5, 5, "chance", color=SLATE, fontsize=18, ha="center")
    ax.set_xticks(range(len(results)), names)
    ax.set_xlim(-0.5, len(results) - 0.5)
    ax.set_ylim(-4, 105)
    ax.set_ylabel("hidden animal named first (%)")
    ax.text(1.5, 70, "a linear readout of\nthe reply's state", color=BLUE, fontsize=20, ha="center", va="center")
    ax.text(1.5, 20, "the model,\nasked at recall", color=CORAL, fontsize=20, ha="center", va="center")

    # d: the weights of layer 21 (Qwen3-1.7B): three heads read the same directions, with opposite signs
    cv.letter(9.2, top - 0.7, "d")
    cv.S.text(10.0, top - 0.7, "Qwen3-1.7B layer 21, weights only: three heads\nread the same directions, write opposite signs",
              fontsize=25, weight="bold", va="center", linespacing=1.15)
    ov = json.load(open(overlap))
    O, g = np.array(ov["overlap"]), np.array(ov["copying_gain"])
    n = len(O)
    ax = cv.axes(10.6, y0 + 1.9, 6.0, 7.6)
    M = O.copy()
    np.fill_diagonal(M, np.nan)
    ax.imshow(M, cmap=LinearSegmentedColormap.from_list("w", ["#f7f7f5", INK]), vmin=0, vmax=0.6)
    for a in range(n):
        for b in range(n):
            if a != b and M[a, b] > 0.3:
                ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=15, color="white")
    ax.set_xticks(range(n), [str(h) for h in range(n)])
    ax.set_yticks(range(n), [str(h) for h in range(n)])
    ax.set_xlabel("key/value head")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    gmax = np.abs(g).max()
    for h in range(n):
        ax.scatter([n - 0.1], [h], s=60 + 500 * abs(g[h]) / gmax, color=BLUE if g[h] > 0 else CORAL, clip_on=False, zorder=5)
    ax.text(n - 0.1, -0.95, "copying\ngain", ha="center", va="bottom", fontsize=16, color=SLATE)
    for h in (0, 5, 6):
        ax.get_yticklabels()[h].set_weight("bold")
        ax.get_xticklabels()[h].set_weight("bold")
    ax.set_xlim(-0.5, n + 0.4)

    # e: where layer 21 looks at the answer: attention to the reply's "." under each recall question (attention.py)
    cv.letter(18.6, top - 0.7, "e")
    cv.S.text(19.4, top - 0.7, "Qwen3-1.7B layer 21 at the answer:\nattention to the reply's \".\"", fontsize=25,
              weight="bold", va="center", linespacing=1.15)
    att = [json.load(open(p)) for p in attention]
    ax = cv.axes(20.6, y0 + 1.9, 6.6, 7.6)
    heads = [0, 5, 6]
    for k, h in enumerate(heads):
        for j, (rw, col) in enumerate((("A", CORAL), ("B", BLUE))):
            vals = [100 * d["kv_heads"][f"21:{h}"][d["reply_tokens"].index(".")] for d in att if d["recall_wording"] == rw]
            ax.bar(k + (j - 0.5) * 0.36, np.mean(vals), width=0.34, color=col, zorder=2)
            ax.scatter([k + (j - 0.5) * 0.36] * len(vals), vals, s=40, color=INK, zorder=3)
    ax.set_xticks(range(len(heads)), ["0\ncopies", "5", "6\nsuppresses"])
    ax.set_xlabel("key/value head")
    ax.set_ylabel("attention to \".\" (%)")
    ax.text(2.0, 1.32, "recall question A", color=CORAL, fontsize=20, ha="right", va="bottom")
    ax.text(2.0, 1.18, "recall question B", color=BLUE, fontsize=20, ha="right", va="bottom")
    ax.set_ylim(0, 1.5)


def panel_cross(cv, x, y, w, h, cells):
    """Discrimination for each recall question (and the any-animal question), the turn-1 cache written under either
    wording, unedited and with one head deleted.  cells: {"A": {question: [path, arm]}, "B": {...}}, the edit spec "edit"."""
    ax = cv.axes(x, y, w, h)
    qs = ["A", "B", "neutral"]
    for t1, marker in (("A", "o"), ("B", "s")):
        for state, col in (("none", INK), (cells["edit"], BLUE)):
            ys = []
            for q in qs:
                path, arm = cells[t1][q]
                e = json.load(open(path))["edits"]
                ys.append(100 * e[state][arm]["discrimination"])
            ax.plot(range(3), ys, "-", color=col, lw=3.5, zorder=2)
            ax.scatter(range(3), ys, s=230, marker=marker, color=col, edgecolor="white", lw=2, zorder=3)
    ax.axhline(50, color=SLATE, lw=1.5, ls=(0, (4, 3)))
    ax.set_xticks(range(3), ["\u201cWhich animal did\nyou choose?\u201d", "\u201cRecall, introspect,\nor reconstruct it\u201d",
                             "\u201cName any animal\nfrom the list\u201d"], fontsize=21)
    ax.set_xlim(-0.35, 2.75)
    ax.set_ylim(35, 80)
    ax.set_ylabel("hidden animal ranked above\nanother animal (%)")
    return ax


def row_wording(cv, y0, cells, flip_groups):
    """Panels f-g, occupying [y0, y0 + ROWS[2]]."""
    top = y0 + ROWS[2]
    cv.letter(0.1, top - 0.7, "f")
    cv.S.text(0.9, top - 0.7, "Qwen3-1.7B: the recall question sets the direction,\nthrough one head (layer 21, head 6)",
              fontsize=25, weight="bold", va="center", linespacing=1.15)
    ax = panel_cross(cv, 2.2, y0 + 2.1, 9.6, 7.9, cells)
    ax.text(2.72, 72.5, "head 6 deleted\nfrom the weights", color=BLUE, fontsize=21, ha="right", va="center")
    ax.text(2.2, 45.5, "unedited", color=INK, fontsize=21, ha="left", va="center")
    cv.S.scatter([2.6], [top - 1.9], s=200, marker="o", color=SLATE, edgecolor="white")
    cv.S.text(2.85, top - 1.9, "turn 1 in wording A", fontsize=20, color=SLATE, va="center")
    cv.S.scatter([6.6], [top - 1.9], s=200, marker="s", color=SLATE, edgecolor="white")
    cv.S.text(6.85, top - 1.9, "turn 1 in wording B", fontsize=20, color=SLATE, va="center")

    cv.letter(13.6, top - 0.7, "g")
    cv.S.text(14.4, top - 0.7, "the same deletion on other data,\nand deleting other heads", fontsize=25, weight="bold",
              va="center", linespacing=1.15)
    cv.S.scatter([14.6], [top - 1.75], s=90, color=SLATE)
    cv.S.text(14.85, top - 1.75, "unedited", fontsize=19, color=SLATE, va="center")
    cv.S.annotate("", xy=(17.6, top - 1.75), xytext=(16.4, top - 1.75),
                  arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=3.5, mutation_scale=24))
    cv.S.text(17.8, top - 1.75, "edited", fontsize=19, color=BLUE, va="center")
    panel_flip(cv, 19.4, y0 + 1.9, 8.0, 8.6, flip_groups)


def row_fourlayer(cv, why_paths, sweep_paths, y0):
    """Panels h-i, occupying [y0, y0 + ROWS[3]]."""
    top = y0 + ROWS[3]
    # f: the 4-layer circuit, three VPD subcomponents that set it (attention measured by fourlayer/why4l.py)
    cv.letter(0.1, top - 0.7, "h")
    cv.S.text(0.9, top - 0.7, "Goodfire 4-layer model: three VPD subcomponents\nset how much of the word reaches the cue",
              fontsize=25, weight="bold", va="center", linespacing=1.15)
    why, why_b = json.load(open(why_paths[0])), json.load(open(why_paths[1]))["H2"]
    ax = cv.axes(0.6, y0 + 0.4, 14.5, 9.8)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")
    hb, yb = 1.1, 1.3
    def tok(x, w, text, face, edge, color=INK, style="-"):
        ax.add_patch(FancyBboxPatch((x, yb), w, hb, boxstyle="round,pad=0,rounding_size=0.12", facecolor=face,
                                    edgecolor=edge, lw=2.5, linestyle=style))
        ax.text(x + w / 2, yb + hb / 2, text, ha="center", va="center", fontsize=24, color=color)
    tok(0.0, 2.2, "first token", PALE_SLATE, "none")
    tok(2.6, 2.0, "otter", "white", "#e3a1a8", color="#d98b93", style=(0, (4, 3)))
    tok(5.0, 5.0, ". Nobody else knows.", PALE_BLUE, BLUE)
    tok(10.5, 3.7, "My pet is a", PALE_SLATE, "none")
    ax.text(3.6, 0.6, "not kept", ha="center", fontsize=22, color=RED)
    ax.text(7.5, 0.6, "cache kept", ha="center", fontsize=22, color=BLUE)
    ax.text(12.35, 0.6, "cue", ha="center", fontsize=22, color=SLATE)
    from matplotlib.patches import FancyArrowPatch
    def arr(a, b, width, color, rad):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=28, lw=width, color=color,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=3, shrinkB=3))
    copy = lambda d: float(np.mean([d["write_layer2"][2], d["write_layer2"][3]]))
    top_y = yb + hb + 0.05
    arr((6.8, top_y), (3.9, top_y), 2 + 30 * copy(why["none"]), BLUE, 0.55)
    arr((11.6, top_y), (8.4, top_y), 2 + 30 * why_b["with #334"]["later tokens"], INK, 0.55)
    arr((12.8, top_y), (1.1, top_y), 2 + 30 * why_b["with #334"]["first token"], CLOUD, 0.42)
    ax.text(5.35, 3.45, "layer 2 copies", ha="center", va="bottom", fontsize=23, color=BLUE, weight="bold")
    ax.text(10.0, 3.45, "layer 3 reads", ha="center", va="bottom", fontsize=23, color=INK, weight="bold")
    ax.text(6.95, 4.15, "or the first token", ha="center", va="center", fontsize=21, color=SLATE)
    # the three dials, each at the token where it acts
    dials = [(3.6, 6.0, BLUE, "key #224", "stronger: more copying", "left"),
             (3.6, 7.9, CORAL, "MLP out #1320", "deleted: more copying", "left"),
             (12.35, 6.0, CORAL, "query #334", "deleted: reads the\nlater tokens more", "right")]
    for dx, dy, col, name, what, side in dials:
        ax.plot([dx, dx], [yb + hb + 0.1, dy - 0.4], color=col, lw=1.5, ls=(0, (2, 2)))
        ax.scatter([dx], [dy], s=420, color=col, zorder=5)
        tx, ha = (dx + 0.45, "left") if side == "left" else (dx - 0.45, "right")
        ax.text(tx, dy + 0.1, name, fontsize=24, color=col, weight="bold", va="center", ha=ha)
        ax.text(tx, dy - 0.7, what, fontsize=20, color=SLATE, va="center", ha=ha, linespacing=0.95)

    # g: what the edits buy, against damage on other text (fourlayer/edit_eval.py)
    cv.letter(16.4, top - 0.7, "i")
    cv.S.text(17.2, top - 0.7, "same model: turn the three subcomponents\nin the weights, at a cost elsewhere", fontsize=25,
              weight="bold", va="center", linespacing=1.15)
    ax = cv.axes(18.6, y0 + 1.9, 8.6, 8.4)
    pts = [(r["pile_kl"], 100 * r["discrimination"]) for path in sweep_paths for spec, r in json.load(open(path)).items()
           if "discrimination" in r]
    xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    ax.scatter(xs, ys, s=60, color=CLOUD, zorder=2)
    best, front = -1, []
    for i in np.argsort(xs):
        if ys[i] > best:
            best, front = ys[i], front + [i]
    ax.plot(xs[front], ys[front], "-o", color=BLUE, lw=3.5, ms=10, zorder=3)
    ax.scatter([0], [ys[np.argmin(xs)]], s=220, color=INK, zorder=4)
    ax.annotate("unedited", (0, ys[np.argmin(xs)]), xytext=(12, -14), textcoords="offset points", fontsize=19)
    ax.axhline(50, color=SLATE, lw=1.5, ls=(0, (4, 3)))
    ax.set_ylim(35, 85)
    ax.set_xlabel("damage on other text\n(KL, nats per token)")
    ax.set_ylabel("hidden word ranked above\nanother word (%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--text4l", nargs="*", default=[],
                    help="fourlayer/hidden_span.py outputs: the same three conditions in text, 4-layer model")
    ap.add_argument("--probes", required=True, help="probe accuracies per model (probe.py)")
    ap.add_argument("--overlap", required=True, help="overlap.py output for the layer of the circuit")
    ap.add_argument("--why", nargs=2, required=True, help="fourlayer/why4l.py and why4l_b.py outputs")
    ap.add_argument("--sweep", nargs="+", required=True, help="fourlayer/edit_eval.py outputs")
    ap.add_argument("--attention", nargs=4, required=True, help="attention.py outputs, both turn-1 by both recall wordings")
    ap.add_argument("--cross", required=True, help="JSON: the cells of panel f (see panel_cross)")
    ap.add_argument("--flip", required=True, help="JSON: groups of rows for panel g (see panel_flip)")
    ap.add_argument("--out", default="figs/main.png")
    a = ap.parse_args()
    fig = plt.figure(figsize=(W, H))
    cv = Canvas(fig)
    row_experiment(cv, a.results, ROWS[1] + ROWS[2] + ROWS[3], a.text4l)
    row_qwen(cv, ROWS[2] + ROWS[3], a.results, a.probes, a.overlap, a.attention)
    row_wording(cv, ROWS[3], json.load(open(a.cross)), json.load(open(a.flip)))
    row_fourlayer(cv, a.why, a.sweep, 0)
    for y in (ROWS[3] + ROWS[2] + ROWS[1], ROWS[3] + ROWS[2], ROWS[3]):
        cv.S.plot([0.3, W - 0.3], [y, y], color="#e3e5e8", lw=2)
    fig.savefig(a.out, dpi=100)


if __name__ == "__main__":
    main()
