"""The main figure, three rows.
  a-b  the experiment, and per model the share of other animals the hidden animal is ranked above at recall (each
       animal relative to its mean over runs; 50% = chance) in each condition; last column: the same three conditions in
       plain text on Goodfire's 4-layer model
  c-e  Qwen3-1.7B: switching off layer 21, head 6 (its output weights set to zero) on several datasets, with other
       heads switched off as controls (edit_heads.py); each turn-2 question with and without the head; the head's
       attention on the reply while answering, under each recall question (attention.py)
  f    Goodfire's 4-layer model: the cheapest weight edits of its VPD subcomponents against damage on Pile text
       (fourlayer/edit_eval.py, fourlayer/optimize_edit.py)
usage: figure_main.py --results R.json ... --text4l T.json --flip figs/flip_rows.json --questions figs/questions.json
       --attention A1.json A2.json A3.json A4.json --sweep S.json [--optimized O.json]
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
ROWS = [14.5, 11.5, 11.0]                  # heights of the three rows
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
             "visible": ((0, 105), [0, 50, 100], False)}
    specs4 = specs
    # direction: an up arrow above zero on the top plot, a down arrow below zero on the bottom plot
    arrows = {"stripped": (51, 78, BLUE, "toward"), "visible": (49, 6, CORAL, "away")}
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


def direction_axis(ax, lo, hi):
    """Discrimination axis with chance marked and the two directions named at its ends."""
    ax.axvline(50, color=SLATE, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(lo, hi)
    ax.text(49, 1.0, "away from\nthe hidden animal", transform=ax.get_xaxis_transform(), ha="right", va="bottom",
            fontsize=19, color=CORAL, linespacing=1.0)
    ax.text(51, 1.0, "toward\nthe hidden animal", transform=ax.get_xaxis_transform(), ha="left", va="bottom",
            fontsize=19, color=BLUE, linespacing=1.0)


def panel_switch(cv, x, y, w, h, groups):
    """Discrimination with the original model (gray dot) and with an attention head switched off (arrow head), one
    line per dataset or head.  groups: [{"group", "control" (bool), "rows": [{"label", "path" (edit_heads.py output),
    "spec", "arm" (default retained)}]}]; lines whose result is not there yet are left out."""
    lines = []
    for g in groups:
        got = []
        for r in g["rows"]:
            try:
                e = json.load(open(r["path"]))["edits"]
            except FileNotFoundError:
                continue
            arm = r.get("arm", "retained")
            if r["spec"] in e and "none" in e and arm in e["none"] and arm in e[r["spec"]]:
                got.append((r["label"], 100 * e["none"][arm]["discrimination"], 100 * e[r["spec"]][arm]["discrimination"],
                            g.get("control", False)))
        if got:
            lines.append((g["group"], None, None, None))
            lines += got
    ax = cv.axes(x, y, w, h)
    yy, ticks, labels = 0.0, [], []
    for label, a, b, control in lines:
        if a is None:
            yy -= 0.3 if ticks else 0
            ax.text(-0.02, yy, label, transform=ax.get_yaxis_transform(), fontsize=20, color=INK, weight="bold",
                    va="center", ha="right")
            yy -= 1.0
            continue
        col = CLOUD if control else (BLUE if b > 50 else CORAL)
        ax.annotate("", xy=(b, yy), xytext=(a, yy), arrowprops=dict(arrowstyle="-|>", color=col, lw=4,
                                                                      mutation_scale=26, shrinkA=0, shrinkB=0))
        ax.scatter([a], [yy], s=110, color=SLATE, zorder=3)
        ticks.append(yy)
        labels.append(label)
        yy -= 1.0
    direction_axis(ax, 35, 80)
    ax.set_yticks(ticks, labels, fontsize=20)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(yy + 0.4, 0.7)
    ax.set_xticks([40, 50, 60, 70, 80])
    ax.set_xlabel("hidden animal ranked above another animal (%)", fontsize=22)
    return ax


def panel_questions(cv, x, y, w, h, spec):
    """Discrimination for each question asked in turn 2, original model (outline) and head switched off (filled);
    bars start at chance.  spec: {"edit", "questions": [{"label", "path", "arm"}]}."""
    ax = cv.axes(x, y, w, h)
    for k, q in enumerate(spec["questions"]):
        e = json.load(open(q["path"]))["edits"]
        for j, state in enumerate(("none", spec["edit"])):
            v = 100 * e[state][q["arm"]]["discrimination"]
            col = BLUE if v > 50 else CORAL
            xpos = k + (j - 0.5) * 0.38
            ax.bar(xpos, v - 50, bottom=50, width=0.34, color=col if j else "white", edgecolor=col, lw=3, zorder=2)
            ax.text(xpos, v + (1.0 if v > 50 else -1.0), f"{v:.0f}", ha="center", va="bottom" if v > 50 else "top",
                    fontsize=19, color=col)
    ax.axhline(50, color=SLATE, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(range(len(spec["questions"])), [q["label"] for q in spec["questions"]], fontsize=18)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(spec["questions"]) - 0.4)
    ax.set_ylim(35, 75)
    ax.set_yticks([40, 50, 60, 70])
    ax.set_ylabel("hidden animal ranked above\nanother animal (%)")
    ax.text(len(spec["questions"]) - 0.45, 74, "toward the\nhidden animal", ha="right", va="top", fontsize=18, color=BLUE)
    ax.text(len(spec["questions"]) - 0.45, 36, "away from the\nhidden animal", ha="right", va="bottom", fontsize=18, color=CORAL)
    return ax


def row_head(cv, y0, flip_groups, questions, attention):
    """Panels c-e, Qwen3-1.7B, occupying [y0, y0 + ROWS[1]]."""
    top = y0 + ROWS[1]
    cv.letter(0.1, top - 0.75, "c")
    cv.S.text(0.9, top - 0.75, "Qwen3-1.7B: switch off one attention head\n(layer 21, head 6) and the answer turns\ntoward the hidden animal",
              fontsize=25, weight="bold", va="center", linespacing=1.15)
    cv.S.scatter([1.1], [top - 2.35], s=110, color=SLATE)
    cv.S.text(1.35, top - 2.35, "original model", fontsize=20, color=SLATE, va="center")
    cv.S.annotate("", xy=(5.3, top - 2.35), xytext=(4.1, top - 2.35),
                  arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=4, mutation_scale=26))
    cv.S.text(5.5, top - 2.35, "head switched off (its output weights set to 0)", fontsize=20, color=BLUE, va="center")
    panel_switch(cv, 5.4, y0 + 1.1, 5.6, 6.9, flip_groups)

    cv.letter(12.2, top - 0.75, "d")
    cv.S.text(13.0, top - 0.75, "the wording of the turn-2 question sets the\ndirection, and only while head 21.6 works",
              fontsize=25, weight="bold", va="center", linespacing=1.15)
    cv.S.add_patch(plt.Rectangle((13.15, top - 2.55), 0.4, 0.4, facecolor="white", edgecolor=SLATE, lw=2.5))
    cv.S.text(13.75, top - 2.35, "original model", fontsize=20, color=SLATE, va="center")
    cv.S.add_patch(plt.Rectangle((16.85, top - 2.55), 0.4, 0.4, facecolor=SLATE, edgecolor=SLATE, lw=2.5))
    cv.S.text(17.45, top - 2.35, "head 21.6 switched off", fontsize=20, color=SLATE, va="center")
    panel_questions(cv, 14.6, y0 + 2.6, 6.2, 6.3, questions)

    cv.letter(21.3, top - 0.75, "e")
    cv.S.text(22.1, top - 0.75, "why: “Which animal did you\nchoose?” makes head 21.6\nlook at the reply more",
              fontsize=25, weight="bold", va="center", linespacing=1.15)
    att = [json.load(open(p)) for p in attention]
    ax = cv.axes(23.0, y0 + 2.6, 4.6, 6.3)
    for k, h in enumerate((6, 0)):
        for j, (rw, col) in enumerate((("A", CORAL), ("B", BLUE))):
            vals = [100 * sum(d["kv_heads"][f"21:{h}"]) for d in att if d["recall_wording"] == rw]
            ax.bar(k + (j - 0.5) * 0.38, np.mean(vals), width=0.34, color=col, zorder=2)
    ax.set_xticks([0, 1], ["head 21.6\n(pushes away)", "head 21.0\n(pushes toward)"], fontsize=19)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("attention on the reply\nwhile answering (%)")
    ax.text(1.55, 3.15, "asked “Which animal\ndid you choose?”", color=CORAL, fontsize=18, ha="right", va="top")
    ax.text(1.55, 2.45, "asked “Recall, introspect,\nor reconstruct…”", color=BLUE, fontsize=18, ha="right", va="top")
    ax.set_ylim(0, 3.2)
    ax.set_xlim(-0.55, 1.55)


def panel_words(cv, x, y, w, h, path):
    """Discrimination for turn-2 questions that differ only in their final request, over the same turn-1 caches
    (edit_heads.py --question); bars start at chance."""
    e = json.load(open(path))["edits"]["none"]
    rows = [("retained", "\u201cWhich animal did you choose?\u201d"),
            ("A_recall", "\u201cRecall which animal you chose\u201d"),
            ("A_introspect", "\u201cIntrospect on which animal you chose\u201d"),
            ("A_reconstruct", "\u201cReconstruct which animal you chose\u201d"),
            ("A_all_three", "\u201cRecall, introspect, or reconstruct which\nanimal you chose\u201d"),
            (None, None),
            ("B_original", "question B as written"),
            ("B_which", "question B ending \u201cWhich animal did\nyou choose?\u201d")]
    ax = cv.axes(x, y, w, h)
    yy, ticks, labels = 0.0, [], []
    for arm, label in rows:
        if arm is None:
            yy -= 0.5
            continue
        if arm not in e:
            continue
        v = 100 * e[arm]["discrimination"]
        col = BLUE if v > 50 else CORAL
        ax.barh(yy, v - 50, left=50, height=0.66, color=col, zorder=2)
        ax.text(v + (0.6 if v > 50 else -0.6), yy, f"{v:.0f}", va="center", ha="left" if v > 50 else "right", fontsize=19, color=col)
        ticks.append(yy)
        labels.append(label)
        yy -= 1.0
    direction_axis(ax, 35, 70)
    ax.set_yticks(ticks, labels, fontsize=19)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(yy + 0.4, 0.6)
    ax.set_xticks([40, 50, 60, 70])
    ax.set_xlabel("hidden animal ranked above another animal (%)", fontsize=22)


def row_fourlayer(cv, sweep_paths, optimized, words, y0):
    """Panel f, occupying [y0, y0 + ROWS[2]]: the cheapest VPD-subcomponent weight edits of Goodfire's 4-layer model."""
    top = y0 + ROWS[2]
    cv.letter(0.1, top - 0.75, "f")
    cv.S.text(0.9, top - 0.75, "Goodfire 4-layer model: the cheapest weight edits that raise\nrecall, built from its VPD subcomponents (rank-one parts\nof the weights)",
              fontsize=25, weight="bold", va="center", linespacing=1.15)
    ax = cv.axes(2.4, y0 + 1.6, 8.6, 6.9)
    names = {"h.3.attn.q_proj#334": "query #334", "h.2.attn.k_proj#224": "key #224", "h.1.mlp.down_proj#1320": "MLP out #1320"}
    word = lambda f: {"0": "removed", "0.5": "halved", "2": "doubled", "4": "×4", "8": "×8"}.get(f, f"×{f}")
    pts = []
    for path in sweep_paths:
        for spec, r in json.load(open(path)).items():
            if "discrimination" in r and r["pile_kl"] <= 0.1:
                lab = "unedited" if spec == "unedited" else ",\n".join(f"{names[p.split(':')[0]]} {word(p.split(':')[1])}"
                                                                       for p in spec.split(","))
                pts.append((r["pile_kl"], 100 * r["discrimination"], lab))
    pts.sort()
    best, front = -1, []
    for kl, d, lab in pts:
        if d > best:
            best, front = d, front + [(kl, d, lab)]
    ax.plot([p[0] for p in front], [p[1] for p in front], "-o", color=BLUE, lw=3.5, ms=12, zorder=3,
            label="hand-picked subcomponents")
    for kl, d, lab in front:
        ax.annotate(lab, (kl, d), xytext=(10, -6), textcoords="offset points", fontsize=17, color=INK if lab == "unedited" else BLUE,
                    va="top")
    if optimized:
        o = json.load(open(optimized))["frontier"]
        o = sorted((r["kl"], 100 * r["discrimination"]) for r in o if r["kl"] <= 0.1)
        ax.plot([p[0] for p in o], [p[1] for p in o], "-s", color=INK, lw=3.5, ms=11, zorder=3,
                label="all subcomponents, fitted on other templates")
        ax.legend(frameon=False, fontsize=18, loc="lower right")
    ax.axhline(50, color=SLATE, lw=1.5, ls=(0, (4, 3)))
    ax.set_xlim(-0.003, 0.1)
    ax.set_ylim(48, 75)
    ax.set_xlabel("damage to the model on ordinary Pile text (KL, nats per token)")
    ax.set_ylabel("hidden word ranked above\nanother word (%)")
    if words:
        cv.letter(13.6, top - 0.75, "g")
        cv.S.text(14.4, top - 0.75, "Qwen3-1.7B: which words in the recall question turn the\nanswer toward the hidden animal (same caches, original model)",
                  fontsize=25, weight="bold", va="center", linespacing=1.15)
        panel_words(cv, 20.6, y0 + 1.6, 6.6, 6.9, words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--text4l", nargs="*", default=[],
                    help="fourlayer/hidden_span.py outputs: the same three conditions in text, 4-layer model")
    ap.add_argument("--flip", required=True, help="JSON: groups of lines for panel c (see panel_switch)")
    ap.add_argument("--questions", required=True, help="JSON: the questions of panel d (see panel_questions)")
    ap.add_argument("--attention", nargs=4, required=True, help="attention.py outputs, both turn-1 by both recall wordings")
    ap.add_argument("--sweep", nargs="+", required=True, help="fourlayer/edit_eval.py outputs")
    ap.add_argument("--optimized", default=None, help="fourlayer/optimize_edit.py output, if there is one")
    ap.add_argument("--words", default=None, help="edit_heads.py --question output: recall questions differing in their request")
    ap.add_argument("--out", default="figs/main.png")
    a = ap.parse_args()
    fig = plt.figure(figsize=(W, H))
    cv = Canvas(fig)
    row_experiment(cv, a.results, ROWS[1] + ROWS[2], a.text4l)
    row_head(cv, ROWS[2], json.load(open(a.flip)), json.load(open(a.questions)), a.attention)
    row_fourlayer(cv, a.sweep, a.optimized, a.words, 0)
    for y in (ROWS[2] + ROWS[1], ROWS[2]):
        cv.S.plot([0.3, W - 0.3], [y, y], color="#e3e5e8", lw=2)
    fig.savefig(a.out, dpi=100)


if __name__ == "__main__":
    main()
