"""The main figure (make_figure.sh), four rows.
  a-b  the experiment, and per model the share of other animals the hidden animal is ranked above at recall (each
       animal relative to its mean over runs; 50% = chance) in each condition; last column: the same three conditions in
       text on Goodfire's 4-layer model, as the raise of the hidden word's log P within its template (nats)
  c-e  Qwen3-1.7B: key-value head 21.6 switched off across datasets, with other heads as controls; per hidden animal,
       each question with the head and without it (held-out runs); each question alone and after Janus's LLM explainer
       (and Qwen3-0.6B)
  f-g  the same with a CPU explainer of the same length and style between the two; the Qwen3-1.7B head circuit
  h-i  Goodfire's 4-layer model: the fewest VPD subcomponents that make the hidden word the top answer; the attention
       route of recall and the subcomponents that switch it (Graphviz, circuit_dot.py)
Paired p-values: sign flips (t statistic) of each run's paired difference in the hidden animal's log P relative to
that animal's mean over runs; p against chance: re-pairing hidden animals.
usage: make_figure.sh
"""
import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch

from stats import animal_level

INK, BLUE, PALE_BLUE, CORAL, PALE_CORAL = "#1d1d1f", "#2f6db5", "#dce8f6", "#e8684a", "#fbe3dc"
SLATE, PALE_SLATE, RED, CLOUD = "#8e959c", "#eceef0", "#c8102e", "#cdd2d7"
GOLD = "#c99a1e"
CMAP = LinearSegmentedColormap.from_list("cb", [CORAL, "#f7f7f5", BLUE])
FC_CMAP = LinearSegmentedColormap.from_list("fc", [CORAL, "#dcdcdc", BLUE])   # log2 fold change of a paired value
FC_NORM = matplotlib.colors.Normalize(-1, 1)
plt.rcParams.update({"font.family": "Avenir Next", "font.size": 26, "figure.facecolor": "white",
                     "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.6, "xtick.major.width": 1.6, "ytick.major.width": 1.6})
W = 28.0                                   # figure width in drawing units (inches)
ROWS = [14.5, 11.5, 11.0, 12.5]            # heights of the four rows
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


def per_run(L, c):
    """Per run, the share of other animals its hidden animal is ranked above, each animal relative to its mean over runs."""
    D = L - L.mean(0)
    diff = D[np.arange(len(c)), c][:, None] - D
    return ((diff > 1e-9).sum(1) + 0.5 * ((np.abs(diff) <= 1e-9).sum(1) - 1)) / (D.shape[1] - 1) * 100


def saved_logp(spec, key):
    """log P [runs, animals] and the hidden animals of one saved edit|arm, pooled over sets of runs ("a.npz+b.npz"),
    each set centered on its own mean per animal."""
    Ls, cs = [], []
    for path in spec.split("+"):
        if not os.path.exists(path):
            continue
        z = np.load(path)
        if key not in z.files:
            return None, None
        Ls.append(z[key] - z[key].mean(0))
        cs.append(z["chosen"])
    return (np.concatenate(Ls), np.concatenate(cs)) if Ls else (None, None)


def pooled(spec, arm, per_template=False):
    """Log-probabilities and hidden animals of one condition, pooled over independent sets of runs ("a.json+b.json").
    Each set is centered on its own mean per animal first, so the comparison stays within a set (sets computed on
    different hardware differ by rounding).  per_template: the 4-layer model's runs come as templates, each with every
    word hidden once in order; each word is then centered within its template (templates differ in their preferences
    among the words far more than one hidden word moves them)."""
    Ls, cs = [], []
    for path in spec.split("+"):
        d = json.load(open(path))
        L = np.array(d["arms"][arm])
        c = np.array([d["animals"].index(x) for x in d["chosen"]])
        if per_template:
            n = len(d["animals"])
            assert (c == np.arange(len(c)) % n).all(), "runs are not templates of every word in order"
            L = (L.reshape(-1, n, L.shape[1]) - L.reshape(-1, n, L.shape[1]).mean(1, keepdims=True)).reshape(L.shape)
        Ls.append(L - L.mean(0))
        cs.append(c)
    return np.concatenate(Ls), np.concatenate(cs)


def strip(ax, paths, arm, lim, ticks, pvals, per_template=False):
    strip_data(ax, [pooled(p, arm, per_template) for p in paths], lim, ticks, pvals)


def p_chance(L, c, own, rng, B=10000):
    """Two-sided animal-level permutation p of the mean per-animal discrimination against chance."""
    present = np.unique(c)
    nulls = np.array([per_animal(L, c, rng.permutation(present)).mean() for _ in range(B)])   # re-pairing animals: the design's own randomization
    return (1 + np.sum(np.abs(nulls - 50) >= abs(own.mean() - 50))) / (B + 1)


def p_text(ax, x, y, pv, sign, fontsize=21):
    col = SLATE if pv >= 0.05 else (BLUE if sign > 0 else CORAL)
    ax.text(x, y, f"p={pv:.1g}" if pv >= 1e-3 else "p<0.001", ha="center", va="top", fontsize=fontsize, color=col,
            weight="bold" if pv < 0.05 else "normal")


def strip_data(ax, data, lim, ticks, pvals):
    """Per hidden animal, the share of other animals it is ranked above (blue), against the same with the hidden animals
    re-paired at random (gray); one column per (log P [runs, animals], hidden animal per run)."""
    rng = np.random.default_rng(0)
    for k, (L, c) in enumerate(data):
        own = per_animal(L, c)
        present = np.unique(c)
        null = np.concatenate([per_animal(L, c, rng.permutation(present)) for _ in range(12)])
        ax.scatter(k - 0.3 + rng.uniform(0, 0.24, null.size), null, s=10, color=CLOUD, alpha=0.9, edgecolor="none", zorder=2)
        ax.scatter(k + 0.06 + rng.uniform(0, 0.22, own.size), own, s=60, color=BLUE, alpha=0.9, edgecolor="white",
                   lw=0.6, zorder=3, clip_on=False)
        ax.plot([k + 0.03, k + 0.31], [own.mean()] * 2, color=INK, lw=5, zorder=4, solid_capstyle="round")
        if pvals:
            p_text(ax, k, lim[1] * 0.985, p_chance(L, c, own, rng), own.mean() - 50)
    ax.axhline(50, color=SLATE, lw=1.4, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(-0.55, len(data) - 0.45)
    ax.set_ylim(*lim)
    ax.set_yticks(ticks)
    ax.set_xticks(range(len(data)), [])


def strip_raise(ax, paths, arm, lim, ticks, pvals):
    """Goodfire's 4-layer model: per hidden word, the raise of its log P relative to its mean within the template
    (nats; blue), against the same with the hidden words re-paired at random (gray).  Within a template the runs differ
    only in the hidden word, so the share of other words it is ranked above saturates near 100% for any consistent
    raise; the raise itself shows how strong recall is."""
    rng = np.random.default_rng(0)
    for k, path in enumerate(paths):
        L, c = pooled(path, arm, per_template=True)
        present = np.unique(c)
        raise_of = lambda lab: np.array([L[np.arange(len(c)), lab][c == a].mean() for a in present])
        own = raise_of(c)
        null = np.concatenate([raise_of(rng.permutation(present)[np.searchsorted(present, c)]) for _ in range(12)])
        ax.scatter(k - 0.3 + rng.uniform(0, 0.24, null.size), null, s=10, color=CLOUD, alpha=0.9, edgecolor="none", zorder=2)
        ax.scatter(k + 0.06 + rng.uniform(0, 0.22, own.size), own, s=60, color=BLUE, alpha=0.9, edgecolor="white",
                   lw=0.6, zorder=3, clip_on=False)
        ax.plot([k + 0.03, k + 0.31], [own.mean()] * 2, color=INK, lw=5, zorder=4, solid_capstyle="round")
        if pvals:
            nulls = np.array([raise_of(rng.permutation(present)[np.searchsorted(present, c)]).mean() for _ in range(10000)])
            pv = (1 + np.sum(np.abs(nulls) >= abs(own.mean()))) / 10001
            p_text(ax, k, lim[1] * 0.95, pv, own.mean())
    ax.axhline(0, color=SLATE, lw=1.4, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(-0.55, len(paths) - 0.45)
    ax.set_yscale("symlog", linthresh=0.05, linscale=0.6)        # log above 0.05 nats, linear through zero
    ax.set_ylim(*lim)
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xticks(range(len(paths)), [])


def row_experiment(cv, paths, y0, text4l=None, retained=None):
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
             "retained": ((14, 88), [30, 50, 70], True),
             "visible": ((0, 105), [0, 50, 100], False)}
    specs4 = {k: ((-3, 16), [0, 0.1, 1, 10], k == "retained") for k in ("stripped", "retained", "visible")}
    # direction: an up arrow above zero on the top plot, a down arrow below zero on the bottom plot
    arrows = {"stripped": (51, 78, BLUE, "toward"), "visible": (49, 6, CORAL, "away")}
    for (name, _), y in zip(rows, ys):
        if text4l:
            ax4 = cv.axes(24.6, y - 0.55, 3.0, h + 1.1)
            lim4, ticks4, pv4 = specs4[name]
            strip_raise(ax4, text4l, name, lim4, ticks4, pv4)
            if name == "retained":
                ax4.set_ylabel("hidden word's\nlog P raised (nats)", fontsize=17, labelpad=4)
            if name == "visible":
                ax4.set_xticks(range(len(text4l)), [""] * len(text4l), fontsize=20)
                ax4.set_xlabel("Goodfire 4-layer", fontsize=23, labelpad=6)
        ax = cv.axes(16.2, y - 0.55, 6.9, h + 1.1)
        lim, ticks, pv = specs[name]
        if name == "retained" and retained:          # float32 rescoring of the same runs (edit_heads.py)
            strip_data(ax, [saved_logp(spec, "none|retained") for spec in retained], lim, ticks, pv)
        else:
            strip(ax, paths, name, lim, ticks, pv)
        if name not in arrows:
            ax.set_ylabel("%", fontsize=22)
        if name == "visible":
            ax.set_xticks(range(len(paths)), ["Qwen3\n" + json.load(open(p.split("+")[0]))["model"].split("-")[-1] for p in paths])
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
    direction_axis(ax, 32, 88)
    ax.set_yticks(ticks, labels, fontsize=20)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(yy + 0.4, 0.7)
    ax.set_xticks([40, 50, 60, 70, 80])
    ax.set_xlabel("hidden animal ranked above another animal (%)", fontsize=22)
    return ax


def p_paired(diff, rng, B=100000):
    """Two-sided sign-flip permutation p of paired differences (one per run), with the t statistic: exact when, with no
    effect, each difference is as likely positive as negative.  No difference at all gives p = 1."""
    if not np.any(diff):
        return 1.0
    t = lambda d: d.mean(-1) / (d.std(-1, ddof=1) / np.sqrt(d.shape[-1]))
    t0, hits = abs(t(diff)), 0
    for _ in range(B // 10000):                      # in chunks of 10,000 sign patterns (memory)
        hits += int(np.sum(np.abs(t(rng.choice([-1.0, 1.0], size=(10000, len(diff))) * diff)) >= t0))
    return (1 + hits) / (B + 1)


def centred_logp(L, c):
    """Per run, the hidden animal's log P minus that animal's mean log P over all runs of the same condition."""
    return (L - L.mean(0))[np.arange(len(c)), c]


def panel_paired(ax, pairs, lim):
    """Per hidden animal, discrimination in two conditions over the same runs (gray, then blue), joined by a line colored
    by its log2 fold change; pairs: [(log P first, log P second, hidden animal per run)]; p of the paired difference
    over runs (sign flips of each run's paired difference in the hidden animal's centered log P)."""
    rng = np.random.default_rng(0)
    for k, (L0, L1, c) in enumerate(pairs):
        x0, x1 = 3 * k, 3 * k + 1.2
        o0 = per_animal(L0, c)
        j = rng.uniform(-0.18, 0.18, o0.size)
        if L1 is None:                            # the edited model's runs are not in yet
            ax.scatter(x0 + j, o0, s=55, color=SLATE, edgecolor="white", lw=0.6, zorder=3)
            ax.plot([x0 - 0.3, x0 + 0.3], [o0.mean()] * 2, color=INK, lw=5, zorder=4, solid_capstyle="round")
            continue
        o1 = per_animal(L1, c)
        fc = np.log2(o1 / o0)
        for a_, b_, jj, f in zip(o0, o1, j, fc):
            ax.plot([x0 + jj, x1 + jj], [a_, b_], color=FC_CMAP(FC_NORM(f)), lw=1.6, zorder=1)
        ax.scatter(x0 + j, o0, s=55, color=SLATE, edgecolor="white", lw=0.6, zorder=3)
        ax.scatter(x1 + j, o1, s=55, color=BLUE, edgecolor="white", lw=0.6, zorder=3)
        for x, o in ((x0, o0), (x1, o1)):
            ax.plot([x - 0.3, x + 0.3], [o.mean()] * 2, color=INK, lw=5, zorder=4, solid_capstyle="round")
        d = centred_logp(L1, c) - centred_logp(L0, c)          # the paired test uses each run's log P, not its rank
        p_text(ax, (x0 + x1) / 2, lim[1] - 1, p_paired(d, rng), d.mean())
    ax.axhline(50, color=SLATE, lw=1.4, ls=(0, (4, 3)), zorder=0)
    ax.set_ylim(*lim)
    ax.set_xlim(-0.8, 3 * len(pairs) - 0.6)


def panel_triple(ax, groups, lim):
    """Per hidden animal, discrimination under three conditions over the same runs (gray, gold, blue), joined by
    segments colored by their log2 fold change; groups: [(log P first, second, third, hidden animal per run)].
    Brackets: p of each condition against the one to its left (sign flips of each run's paired difference in the
    hidden animal's centered log P)."""
    rng = np.random.default_rng(0)
    xs = (0.0, 1.15, 2.3)
    for k, (Ls, c) in enumerate(groups):
        x = [4 * k + d for d in xs]
        o = [per_animal(L, c) for L in Ls]
        j = rng.uniform(-0.16, 0.16, o[0].size)
        for i in range(2):
            for a_, b_, jj, f in zip(o[i], o[i + 1], j, np.log2(o[i + 1] / o[i])):
                ax.plot([x[i] + jj, x[i + 1] + jj], [a_, b_], color=FC_CMAP(FC_NORM(f)), lw=1.3, zorder=1)
        for xi, oi, col in zip(x, o, (SLATE, GOLD, BLUE)):
            ax.scatter(xi + j, oi, s=40, color=col, edgecolor="white", lw=0.5, zorder=3)
            ax.plot([xi - 0.28, xi + 0.28], [oi.mean()] * 2, color=INK, lw=4.5, zorder=4, solid_capstyle="round")
        g = [centred_logp(L, c) for L in Ls]
        for i, yb in ((0, lim[1] - 6), (1, lim[1] - 18)):
            d = g[i + 1] - g[i]
            ax.plot([x[i], x[i], x[i + 1], x[i + 1]], [yb - 1.2, yb, yb, yb - 1.2], color=SLATE, lw=1.2, zorder=2)
            p_text(ax, (x[i] + x[i + 1]) / 2, yb + 6.5, p_paired(d, rng), d.mean(), fontsize=16)
    ax.axhline(50, color=SLATE, lw=1.4, ls=(0, (4, 3)), zorder=0)
    ax.set_ylim(*lim)
    ax.set_xlim(-0.7, 4 * len(groups) - 1.0)


def paired_legend(cv, x, y, left, right):
    cv.S.scatter([x], [y], s=110, color=SLATE)
    cv.S.text(x + 0.25, y, left, fontsize=20, color=SLATE, va="center")
    x2 = x + 0.55 + 0.13 * len(left)
    cv.S.scatter([x2], [y], s=110, color=BLUE)
    cv.S.text(x2 + 0.25, y, right, fontsize=20, color=BLUE, va="center")


def explainer_questions(npz, other):
    """(npz, CPU npz, question arm, CPU arm, explainer arm, label) for each column of panels e and f; npz: Qwen3-1.7B,
    other: Qwen3-0.6B (edit_heads.py --save-logp outputs holding every question's arms)."""
    cpu, other_cpu = npz, other
    qs = [(npz, cpu, "retained", "cpu_A", "doc_A", "1.7B:\n\u201cWhich\nanimal\u2026?\u201d"),
          (npz, cpu, "recall_B", "cpu_B", "doc_B", "1.7B:\n\u201cRecall,\nintrospect\u2026\u201d"),
          (npz, cpu, "neutral", "cpu_neutral", "doc_neutral", "1.7B control:\n\u201cName one\nanimal\u2026\u201d"),
          (other, other_cpu, "retained", "cpu_A", "doc_A", "0.6B:\n\u201cWhich\nanimal\u2026?\u201d")]
    return [q for q in qs if saved_logp(q[0], f"none|{q[4]}")[0] is not None]


def row_head(cv, y0, flip_groups, npz, other):
    """Panels c-e, Qwen3-1.7B, occupying [y0, y0 + ROWS[1]].
    c: layer 21 head 6 switched off on several datasets, other heads as controls (edit_heads.py)
    d: per hidden animal, each question, original model and head 21.6 switched off (edit_heads.py --save-logp)
    e: per hidden animal, each question alone and after Janus's LLM explainer (edit_heads.py --question)"""
    top = y0 + ROWS[1]
    cv.letter(0.1, top - 0.75, "c")
    cv.S.text(0.9, top - 0.75, "Qwen3-1.7B: switching off head 21.6\n(layer 21, key-value head 6)\nraises introspection on all data",
              fontsize=25, weight="bold", va="center", linespacing=1.15)
    cv.S.scatter([1.1], [top - 2.35], s=110, color=SLATE)
    cv.S.text(1.35, top - 2.35, "original model", fontsize=20, color=SLATE, va="center")
    cv.S.annotate("", xy=(5.3, top - 2.35), xytext=(4.1, top - 2.35),
                  arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=4, mutation_scale=26))
    cv.S.text(5.5, top - 2.35, "head switched off", fontsize=20, color=BLUE, va="center")
    panel_switch(cv, 5.2, y0 + 1.1, 4.7, 6.9, flip_groups)

    cv.letter(10.6, top - 0.75, "d")
    cv.S.text(11.4, top - 0.75, "head 21.6 reduces introspection", fontsize=25, weight="bold", va="center")
    paired_legend(cv, 11.6, top - 1.75, "original model", "head 21.6 switched off")
    cv.S.text(11.6, top - 2.45, "each dot: one hidden animal; lines join the same animal", fontsize=18, color=SLATE, va="center")
    held = npz.split("+")[0]                     # the held-out runs: head 21.6 was chosen on the other set
    edited = all(saved_logp(held, f"21:6*0|{q}")[0] is not None for q in ("retained", "recall_B", "neutral"))
    qs = [("retained", "\u201cWhich\nanimal did\nyou choose?\u201d"), ("recall_B", "\u201cRecall,\nintrospect, or\nreconstruct\u2026\u201d"),
          ("neutral", "control:\n\u201cName one\nanimal\u2026\u201d")]
    ax = cv.axes(12.9, y0 + 2.6, 5.6, 5.9)
    panel_paired(ax, [(saved_logp(held, f"none|{q}")[0], saved_logp(held, f"21:6*0|{q}")[0] if edited else None,
                       saved_logp(held, f"none|{q}")[1]) for q, _ in qs], (15, 102))
    ax.set_xticks([3 * k + 0.6 for k in range(len(qs))], [lab for _, lab in qs], fontsize=18)
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([30, 50, 70, 90])
    ax.set_ylabel("hidden animal ranked above\nanother animal (%)")

    cv.letter(19.0, top - 0.75, "e")
    qs = explainer_questions(npz, other)
    ax = cv.axes(20.9, y0 + 2.6, 5.75, 5.5)
    cv.S.text(19.8, top - 0.75, "Janus's LLM explainer before the\nquestion: up in Qwen3-1.7B,\ndown in Qwen3-0.6B",
              fontsize=25, weight="bold", va="center", linespacing=1.15)
    paired_legend(cv, 20.0, top - 2.35, "question alone", "Janus's LLM explainer, then the question")
    panel_paired(ax, [(saved_logp(sp, f"none|{q0}")[0], saved_logp(sp, f"none|{qd}")[0], saved_logp(sp, f"none|{q0}")[1])
                      for sp, _, q0, _, qd, _ in qs], (0, 102))
    ax.set_xticks([3 * k + 0.6 for k in range(len(qs))], [lab for *_, lab in qs], fontsize=17)
    ax.set_ylabel("hidden animal ranked above\nanother animal (%)")
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([10, 30, 50, 70, 90])
    cb = cv.fig.colorbar(matplotlib.cm.ScalarMappable(norm=FC_NORM, cmap=FC_CMAP), cax=cv.axes(26.8, y0 + 2.6, 0.2, 5.5))
    cb.set_ticks([-1, 0, 1], labels=["½×", "1×", "2×"])
    cb.set_label("change per line (d-f)", fontsize=16, labelpad=2)
    cb.ax.tick_params(labelsize=17)
    cb.outline.set_visible(False)


SITES = [("h.1.mlp.down_proj", "layer 1\nMLP out"), ("h.2.attn.q_proj", "layer 2\nquery"), ("h.2.attn.k_proj", "layer 2\nkey"),
         ("h.2.attn.v_proj", "layer 2\nvalue"), ("h.2.attn.o_proj", "layer 2\noutput"), ("h.3.attn.q_proj", "layer 3\nquery"),
         ("h.3.attn.k_proj", "layer 3\nkey"), ("h.3.attn.v_proj", "layer 3\nvalue"), ("h.3.attn.o_proj", "layer 3\noutput")]


def row_controls(cv, y0, npz, other):
    """Panels f-g, occupying [y0, y0 + ROWS[2]].
    f: per hidden animal, each question alone, after a CPU explainer of the same length and style, and after Janus's LLM
       explainer; brackets: paired p of each against the one to its left (edit_heads.py --question).
    g: the Qwen3-1.7B circuit of two layer-21 key-value heads (edit_heads.py, weights.py, attention_runs.py), drawn by
       Graphviz (circuit_dot.py)."""
    import circuit_dot
    top = y0 + ROWS[2]
    qs = [q for q in explainer_questions(npz, other) if saved_logp(q[1], f"none|{q[3]}")[0] is not None]
    cv.letter(0.1, top - 0.75, "f")
    cv.S.text(0.9, top - 0.75, "control: a CPU explainer of the same length", fontsize=25, weight="bold", va="center",
              linespacing=1.15)
    for x, col, lab in ((1.1, SLATE, "question alone"), (4.1, GOLD, "CPU explainer first"), (7.6, BLUE, "Janus's LLM explainer first")):
        cv.S.scatter([x], [top - 2.0], s=110, color=col)
        cv.S.text(x + 0.25, top - 2.0, lab, fontsize=19, color=col, va="center")
    ax = cv.axes(2.0, y0 + 1.75, 9.6, 6.3)
    panel_triple(ax, [((saved_logp(sp, f"none|{q0}")[0], saved_logp(cp, f"none|{qc}")[0], saved_logp(sp, f"none|{qd}")[0]),
                       saved_logp(sp, f"none|{q0}")[1]) for sp, cp, q0, qc, qd, _ in qs], (0, 120))
    ax.set_xticks([4 * k + 1.15 for k in range(len(qs))], [lab.replace("\n", " ", 1) for *_, lab in qs], fontsize=19)
    ax.set_ylabel("hidden animal ranked above\nanother animal (%)")
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([10, 30, 50, 70, 90])

    cv.letter(14.6, top - 0.75, "g")
    cv.S.text(15.4, top - 0.75, "Qwen3-1.7B: the head circuit", fontsize=25, weight="bold", va="center")
    image(cv, circuit_dot.render(circuit_dot.qwen_dot(), "figs/circuit_qwen"), 14.8, top - 1.7, 12.8, 9.0, middle=True)


def row_fourlayer(cv, refits, parts, y0):
    """Panels h-i, occupying [y0, y0 + ROWS[3]].
    h: Goodfire's 4-layer model, the fewest VPD subcomponents that make the hidden word the top answer: for each K, the
       scales of only the K largest subcomponents of a fitted edit fitted again on the training templates; on held-out
       templates, how often the hidden word is the model's first choice among the 50 words, against K, colored by KL on
       held-out Pile text (fourlayer/optimize_edit.py --support).
    i: the recall circuit, two attention steps and the VPD subcomponents measured to switch them on or off
       (fourlayer/attention_gates.py, subcomponent_circuit.py), drawn by Graphviz."""
    import circuit_dot
    top = y0 + ROWS[3]
    rows = sorted((o["support_k"], o["metrics"]) for path in refits if os.path.exists(path)
                  for o in json.load(open(path)).get("optimized", []))
    base = 100 * json.load(open(refits[0]))["unedited"]["top1"]
    ks = [k for k, _ in rows]
    t1 = [100 * m["top1"] for _, m in rows]
    kl = np.array([m["pile_kl"] for _, m in rows])
    k90 = next(k for k, t in zip(ks, t1) if t >= 0.9 * max(t1))           # the fewest that reach 90% of the best
    t90 = t1[ks.index(k90)]
    cv.letter(0.1, top - 0.75, "h")
    cv.S.text(0.9, top - 0.75, f"Goodfire 4-layer model: rescaling {k90}\nVPD subcomponents makes the hidden word\n"
              f"the top answer {t90:.0f}% of the time (from {base:.0f}%)",
              fontsize=25, weight="bold", va="center", linespacing=1.15)
    ax = cv.axes(2.4, y0 + 1.6, 5.4, 6.2)
    ax.plot(ks, t1, "-", color=CLOUD, lw=2.5, zorder=2)
    sc = ax.scatter(ks, t1, c=kl, cmap=LinearSegmentedColormap.from_list("kl", ["#d6e4f5", BLUE, INK]),
                    vmin=0, vmax=max(0.05, float(kl.max())), s=110, edgecolor="white", lw=1.2, zorder=3)
    cb = cv.fig.colorbar(sc, cax=cv.axes(8.2, y0 + 1.6, 0.22, 6.2))
    cb.set_label("damage on held-out Pile text\n(KL, nats per token)", fontsize=17)
    cb.ax.tick_params(labelsize=15)
    cb.outline.set_visible(False)
    ax.axhline(base, color=INK, lw=2, ls=(0, (2, 2)))
    ax.text(max(ks) ** 0.75, base + 1.2, f"unedited: {base:.0f}%", fontsize=17, color=INK, va="bottom", ha="center")
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100], ["1", "10", "100"])
    ax.set_ylim(0, 5 * np.ceil(max(t1) / 5) + 5)
    ax.set_xlabel("subcomponents rescaled\n(of 9,728 searched)")
    ax.set_ylabel("hidden word is the top answer\namong 50 words (%),\nheld-out sentences")

    if not parts or not os.path.exists(parts):
        return
    g = json.load(open(parts))
    cv.letter(10.4, top - 0.75, "i")
    stress = json.load(open(os.path.join(os.path.dirname(parts), "stress_circuit.json")))
    share = 1 - stress["routes"]["circuit blocked"]["raise"] / stress["unedited"]["raise"]    # blocking both steps
    cv.S.text(11.2, top - 0.75, f"Goodfire 4-layer model: the attention route behind {100 * share:.0f}% of recall",
              fontsize=25, weight="bold", va="center", linespacing=1.15)
    key(cv, 11.4, top - 2.3)
    image(cv, circuit_dot.render(*circuit_dot.gates_dot(g)[:1], "figs/circuit_gates", snap=circuit_dot.gates_dot(g)[1]),
          10.6, top - 3.1, 17.2, 9.1, middle=True)


def key(cv, x, y):
    """Visual key of the circuit diagrams: wire kinds and widths."""
    S = cv.S
    S.annotate("", xy=(x + 0.9, y), xytext=(x, y), arrowprops=dict(arrowstyle="-|>", lw=3.5, color=BLUE, mutation_scale=22))
    S.text(x + 1.05, y, "needed for recall", fontsize=22, color=BLUE, va="center")
    x2 = x + 4.1
    S.plot([x2, x2 + 0.85], [y, y], color=CORAL, lw=3.5, solid_capstyle="butt")
    S.plot([x2 + 0.85, x2 + 0.85], [y - 0.17, y + 0.17], color=CORAL, lw=4.5, solid_capstyle="butt")
    S.text(x2 + 1.05, y, "holds recall back", fontsize=22, color=CORAL, va="center")
    x3 = x2 + 4.1
    S.annotate("", xy=(x3 + 0.9, y), xytext=(x3, y), arrowprops=dict(arrowstyle="-|>", lw=2, color=CLOUD,
               ls=(0, (4, 2)), mutation_scale=16))
    S.text(x3 + 1.05, y, "input from that token", fontsize=22, color=SLATE, va="center")
    y2 = y - 0.6
    S.annotate("", xy=(x + 0.9, y2), xytext=(x, y2), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=BLUE, mutation_scale=16))
    S.text(x + 1.05, y2, "small effect", fontsize=20, color=SLATE, va="center")
    S.annotate("", xy=(x2 + 0.9, y2), xytext=(x2, y2), arrowprops=dict(arrowstyle="-|>", lw=6, color=BLUE, mutation_scale=26))
    S.text(x2 + 1.05, y2, "large effect", fontsize=20, color=SLATE, va="center")
    for dy in (-0.12, -0.04, 0.04, 0.12):
        S.annotate("", xy=(x3 + 0.9, y2 + dy), xytext=(x3, y2 + dy), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=SLATE,
                   mutation_scale=7))
    S.text(x3 + 1.05, y2, "the word's identity, spread over\nmany value subcomponents", fontsize=19, color=SLATE, va="center",
           linespacing=1.0)


def image(cv, img, x, top, w, h, middle=False):
    """The image scaled to fit w x h drawing units, its top at `top` (or centered in the h band), centered across w."""
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    ax = cv.axes(x + (w - dw) / 2, top - dh - ((h - dh) / 2 if middle else 0), dw, dh)
    ax.imshow(img, interpolation="lanczos")
    ax.axis("off")
    return ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, help="hidden_choice.py outputs, one per model; join sets of runs with +")
    ap.add_argument("--text4l", nargs="*", default=[],
                    help="fourlayer/hidden_span.py outputs: the same three conditions in text, 4-layer model")
    ap.add_argument("--flip", required=True, help="JSON: groups of lines for panel c (see panel_switch)")
    ap.add_argument("--questions", required=True,
                    help="Qwen3-1.7B edit_heads.py --save-logp outputs (.npz, held-out set first, sets joined with +) for d-f")
    ap.add_argument("--questions06", required=True, help="the same for Qwen3-0.6B (e, f)")
    ap.add_argument("--retained", nargs="*", default=[],
                    help="float32 rescoring of the reply-cache-kept condition, one npz spec per --results entry (b)")
    ap.add_argument("--refit", nargs="+", required=True, help="fourlayer/optimize_edit.py --support outputs")
    ap.add_argument("--parts", default=None, help="fourlayer/attention_gates.py output")
    ap.add_argument("--out", default="figs/main.png")
    a = ap.parse_args()
    fig = plt.figure(figsize=(W, H))
    cv = Canvas(fig)
    row_experiment(cv, a.results, ROWS[1] + ROWS[2] + ROWS[3], a.text4l, a.retained)
    row_head(cv, ROWS[2] + ROWS[3], json.load(open(a.flip)), a.questions, a.questions06)
    row_controls(cv, ROWS[3], a.questions, a.questions06)
    row_fourlayer(cv, a.refit, a.parts, 0)
    fig.savefig(a.out, dpi=100)


if __name__ == "__main__":
    main()
