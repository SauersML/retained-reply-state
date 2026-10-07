"""The main figure.
  a-b  the experiment: what turn 2's cache holds in each condition, and log P(animal) at recall above its average, per
       model, for the hidden animal (blue) and for every other animal (gray); last column: the same three conditions in
       plain text on the 4-layer model (a word, then later tokens, then a cue that repeats the word's frame)
  c-d  a 4-layer model with a published parameter decomposition: the share of the effect removed by deleting each single
       subcomponent (fourlayer/pd4l.py), and the effect when one subcomponent's weight is scaled (fourlayer/dose_sub.py)
usage: figure_main.py --results R.json ... --text4l T.json --pd4l P.json --dosesub S.json
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
ROWS = [14.5, 11.5]                        # heights of the two rows
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


def own_matrix(path, arm):
    d = json.load(open(path))
    L = np.array(d["arms"][arm])
    c = np.array([d["animals"].index(x) for x in d["chosen"]])
    present = np.unique(c)
    M = np.array([L[c == a].mean(0)[present] for a in present]) - L.mean(0)[present]
    return M, L, c


def strip(ax, paths, arm, lim, ticks, pvals):
    rng = np.random.default_rng(0)
    for k, p in enumerate(paths):
        M, L, c = own_matrix(p, arm)
        own, other = np.diag(M), M[~np.eye(len(M), dtype=bool)]
        sub = rng.choice(other, min(600, other.size), replace=False)
        ax.scatter(k - 0.3 + rng.uniform(0, 0.24, sub.size), sub, s=10, color=CLOUD, alpha=0.9, edgecolor="none", zorder=2)
        ax.scatter(k + 0.06 + rng.uniform(0, 0.22, own.size), own, s=60, color=BLUE, alpha=0.9, edgecolor="white",
                   lw=0.6, zorder=3, clip_on=False)
        ax.plot([k + 0.03, k + 0.31], [own.mean()] * 2, color=INK, lw=5, zorder=4, solid_capstyle="round")
        if pvals:
            pv = animal_level(L, c, np.random.default_rng(0), 20000, two_sided=True)[1]
            col = SLATE if pv >= 0.05 else (BLUE if own.mean() > 0 else CORAL)
            ax.text(k, lim[1] * 0.97, f"p={pv:.1g}" if pv >= 1e-3 else "p<0.001", ha="center", va="top",
                    fontsize=21, color=col, weight="bold" if pv < 0.05 else "normal")
    ax.axhline(0, color=SLATE, lw=1.4, ls=(0, (4, 3)), zorder=1)
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
        cv.S.text(0.9, y + h / 2, name, fontsize=29, color=col, weight="bold", va="center")
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
    cv.S.text(16.2, top - 0.95, "log P(animal) at recall, above its average", fontsize=29, weight="bold", va="center")
    cv.S.scatter([16.4], [top - 1.75], s=130, color=BLUE, edgecolor="white")
    cv.S.text(16.7, top - 1.75, "this animal was hidden", fontsize=22, color=BLUE, va="center")
    cv.S.add_patch(plt.Rectangle((22.0, top - 1.95), 0.4, 0.4, facecolor=CLOUD, edgecolor="none"))
    cv.S.text(22.6, top - 1.75, "another animal was hidden", fontsize=22, color=SLATE, va="center")
    specs = {"stripped": ((-0.25, 0.25), [-0.2, 0, 0.2], False),
             "retained": ((-0.25, 0.34), [-0.2, 0, 0.2], True),
             "visible": ((-20, 45), [0, 20, 40], False)}
    # direction: an up arrow above zero on the top plot, a down arrow below zero on the bottom plot
    arrows = {"stripped": (0.01, 0.24, BLUE, "toward"), "visible": (-1, -19, CORAL, "away")}
    specs4 = {"stripped": ((-0.25, 0.25), [-0.2, 0, 0.2], False),
              "retained": ((-0.4, 0.75), [0, 0.4], True),
              "visible": ((-4, 9), [0, 4, 8], False)}
    for (name, _), y in zip(rows, ys):
        if text4l:
            ax4 = cv.axes(25.6, y - 0.55, 1.9, h + 1.1)
            lim4, ticks4, pv4 = specs4[name]
            strip(ax4, [text4l], name, lim4, ticks4, pv4)
            ax4.set_xlim(-0.5, 0.5)
            if name == "visible":
                ax4.set_xticks([0], ["Goodfire\n4-layer"])
        ax = cv.axes(16.2, y - 0.55, 8.4, h + 1.1)
        lim, ticks, pv = specs[name]
        strip(ax, paths, name, lim, ticks, pv)
        if name not in arrows:
            ax.set_ylabel("nats", fontsize=22)
        if name == "visible":
            ax.set_xticks(range(len(paths)), ["Qwen3\n" + json.load(open(p))["model"].split("-")[-1] for p in paths])
        if name == "stripped":
            ax.text(len(paths) / 2 - 0.5, 0.12, "identical in every run, so exactly 0", ha="center", fontsize=20, color=SLATE)
        if name in arrows:
            y_from, y_to, col, lab = arrows[name]
            ax.annotate("", xy=(-0.105, y_to), xytext=(-0.105, y_from), xycoords=("axes fraction", "data"),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=3.5, mutation_scale=26), annotation_clip=False)
            ax.text(-0.135, (y_from + y_to) / 2, lab, transform=ax.get_yaxis_transform(), rotation=90, ha="center",
                    va="center", fontsize=21, color=col)


def row_mechanism(cv, pd_path, dosesub_path, y0):
    """Panels c-d, occupying [y0, y0 + ROWS[1]]."""
    top = y0 + ROWS[1]

    # c: delete one subcomponent of the decomposition (4-layer model)
    d = json.load(open(pd_path))
    base = d.get("screen_base", 0.2712)
    order = ["h.2.attn.k_proj", "h.2.attn.v_proj", "h.2.attn.q_proj", "h.2.attn.o_proj",
             "h.3.attn.k_proj", "h.3.attn.v_proj", "h.3.attn.q_proj", "h.3.attn.o_proj"]
    names = {"k_proj": "key", "v_proj": "value", "q_proj": "query", "o_proj": "output"}
    cv.letter(0.1, top - 0.7, "c")
    cv.S.text(0.9, top - 0.7, "Goodfire's 4-layer Pile model: remove one VPD subcomponent where its matrix\nacts in this circuit (each dot is one rank-one piece of an attention matrix)",
              fontsize=25, weight="bold", va="center", linespacing=1.15)
    ax = cv.axes(2.2, y0 + 1.9, 13.6, 7.4)
    rng = np.random.default_rng(0)
    for k, site in enumerate(order):
        sd = d["sites"][site]
        idx = [i for i, x in enumerate(sd["screen_drops"]) if x is not None]
        drops = np.array([sd["screen_drops"][i] for i in idx]) / base * 100
        top2 = [j for j in np.argsort(drops)[::-1][:2] if drops[j] > 8]
        lo = int(np.argmin(drops))
        marked4 = set(top2) | ({lo} if drops[lo] < -8 else set())
        rest = np.array([x for j, x in enumerate(drops) if j not in marked4])
        ax.scatter(k + rng.uniform(-0.18, 0.18, rest.size), rest, s=30, color=CLOUD, zorder=2)
        for j in top2:
            ax.scatter([k], [drops[j]], s=200, color=BLUE, zorder=3, edgecolor="white", lw=2)
        if drops[lo] < -8:
            ax.scatter([k], [drops[lo]], s=200, color=CORAL, zorder=3, edgecolor="white", lw=2)
    ax.axhline(0, color=SLATE, lw=1.4, ls=(0, (4, 3)))
    ax.axvline(3.5, color=SLATE, lw=1.2)
    ax.set_xticks(range(len(order)), [names[s_.split(".")[-1]] for s_ in order], fontsize=21)
    ax.text(1.5, 106, "layer 2", ha="center", fontsize=23, weight="bold")
    ax.text(5.5, 106, "layer 3", ha="center", fontsize=23, weight="bold")
    ax.set_ylim(-110, 118)
    ax.set_yticks([-100, -50, 0, 50, 100])
    ax.set_ylabel("% of the effect removed")

    # d: dose-response of single subcomponents, as weight edits W + (a - 1) u v^T everywhere (4-layer model)
    cv.letter(17.4, top - 0.7, "d")
    cv.S.text(18.2, top - 0.7, "same model: scale one VPD subcomponent\nin the weights (0 = deleted, 1 = unchanged)",
              fontsize=25, weight="bold", va="center", linespacing=1.15)
    ax = cv.axes(19.6, y0 + 1.9, 6.6, 7.4)
    ds = json.load(open(dosesub_path))
    show = {"h.3.attn.q_proj#334": (CORAL, "layer-3 query #334"), "h.2.attn.k_proj#224": (BLUE, "layer-2 key #224")}
    for key, rows in ds.items():
        a = [r["dose"] for r in rows]
        y = [r["raise"] for r in rows]
        col, lab = show.get(key, (CLOUD, None))
        ax.plot(a, y, "-o", color=col, lw=4 if lab else 2.5, ms=9 if lab else 5, zorder=3 if lab else 2)
        if lab:
            ax.text(a[-1] + 0.3, y[-1], lab.replace(" #", "\n#"), color=col, fontsize=20, va="center", weight="bold")
    ax.axvline(1, color=SLATE, lw=1.2, ls=(0, (4, 3)))
    ax.axhline(0, color=SLATE, lw=1.2)
    ax.set_xlim(-0.4, 11.8)
    ax.set_xticks([0, 1, 4, 8])
    ax.set_xlabel("strength of the subcomponent")
    ax.set_ylabel("effect of the hidden word (nats)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--pd4l", required=True)
    ap.add_argument("--text4l", help="fourlayer/hidden_span.py output: the same three conditions in text, 4-layer model")
    ap.add_argument("--dosesub", required=True, help="fourlayer/dose_sub.py output")
    ap.add_argument("--out", default="paper/figs/main.png")
    a = ap.parse_args()
    fig = plt.figure(figsize=(W, H))
    cv = Canvas(fig)
    row_experiment(cv, a.results, ROWS[1], a.text4l)
    row_mechanism(cv, a.pd4l, a.dosesub, 0)
    cv.S.plot([0.3, W - 0.3], [ROWS[1], ROWS[1]], color="#e3e5e8", lw=2)
    fig.savefig(a.out, dpi=110)


if __name__ == "__main__":
    main()
