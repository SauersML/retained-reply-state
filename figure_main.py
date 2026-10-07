"""The main figure.
  a-b  the experiment: what turn 2's cache holds in each condition, and log P(animal) at recall above its average, per
       model, for the hidden animal (blue) and for every other animal (gray)
  c-e  Qwen3-1.7B: the own-animal effect of each key/value head alone (localize.py), the weights-only copying gain
       against that effect (copying.py), and the dose-response of the selected heads (dose.py)
  f-g  a 4-layer model with a published parameter decomposition: the share of the effect removed by deleting each single
       subcomponent (fourlayer/pd4l.py), and top-1 recall when the carried state is amplified (fourlayer/dose4l.py)
usage: figure_main.py --results R.json ... --localize L.json --copying C.json --dose D.json --pd4l P.json --dose4l D4.json
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
ROWS = [14.5, 11.0, 10.5]                  # heights of the three rows
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
            ax.text(k, lim[1] * 0.97, f"p={pv:.1g}" if pv >= 1e-3 else "p<0.001", ha="center", va="top",
                    fontsize=21, color=INK if pv < 0.05 else SLATE, weight="bold" if pv < 0.05 else "normal")
    ax.axhline(0, color=SLATE, lw=1.4, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(-0.55, len(paths) - 0.45)
    ax.set_ylim(*lim)
    ax.set_yticks(ticks)
    ax.set_xticks(range(len(paths)), [])


def row_experiment(cv, paths, y0):
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
             "visible": ((-5, 45), [0, 20, 40], False)}
    for (name, _), y in zip(rows, ys):
        ax = cv.axes(16.2, y - 0.55, 11.0, h + 1.1)
        lim, ticks, pv = specs[name]
        strip(ax, paths, name, lim, ticks, pv)
        if name != "retained":
            ax.set_ylabel("nats", fontsize=22)
        if name == "visible":
            ax.set_xticks(range(len(paths)), [json.load(open(p))["model"].split("-")[-1] for p in paths])
        if name == "stripped":
            ax.text(len(paths) / 2 - 0.5, 0.12, "identical in every run, so exactly 0", ha="center", fontsize=21, color=SLATE)
        if name == "retained":
            for y_from, y_to, col, lab in ((0.02, 0.25, BLUE, "toward"), (-0.02, -0.25, CORAL, "away")):
                ax.annotate("", xy=(-0.105, y_to), xytext=(-0.105, y_from), xycoords=("axes fraction", "data"),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=3.5, mutation_scale=26), annotation_clip=False)
                ax.text(-0.135, (y_from + y_to) / 2, lab, transform=ax.get_yaxis_transform(), rotation=90, ha="center",
                        va="center", fontsize=21, color=col)


def row_mechanism(cv, loc_path, copy_path, dose_path, y0):
    """Rows c-e, occupying [y0, y0 + ROWS[1]]."""
    top = y0 + ROWS[1]
    loc = json.load(open(loc_path))
    cp = json.load(open(copy_path))["kv_heads"]
    dose = json.load(open(dose_path))["groups"]
    model = loc["model"].split("/")[-1]
    heads = {(int(k.split()[1]), int(k.split()[3])): v for k, v in loc["arms"].items()
             if k.startswith("layer ") and "kv-head" in k}
    layers = sorted({l for l, _ in heads})
    nkv = max(h for _, h in heads) + 1
    E = np.full((nkv, len(layers)), np.nan)
    for (l, h), v in heads.items():
        E[h, layers.index(l)] = v["raise"]
    marked = sorted(heads, key=lambda k: -abs(heads[k]["raise"]))[:3]
    sign = {k: BLUE if heads[k]["raise"] > 0 else CORAL for k in marked}

    cv.letter(0.1, top - 0.6, "c")
    cv.S.text(0.9, top - 0.6, f"{model}: one key/value head at a time", fontsize=29, weight="bold", va="center")
    ax = cv.axes(1.8, y0 + 1.6, 9.2, 7.6)
    lim = np.nanmax(np.abs(E))
    ax.imshow(E, cmap=CMAP, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto", origin="lower")
    for (l, h) in marked:
        ax.add_patch(plt.Rectangle((layers.index(l) - 0.5, h - 0.5), 1, 1, fill=False, edgecolor=INK, lw=3))
    step = max(1, len(layers) // 7)
    ax.set_xticks(range(0, len(layers), step), [layers[i] for i in range(0, len(layers), step)])
    ax.set_yticks(range(nkv))
    ax.set_xlabel("layer")
    ax.set_ylabel("key/value head")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cv.letter(11.9, top - 0.6, "d")
    cv.S.text(12.7, top - 0.6, "the weights predict the sign", fontsize=29, weight="bold", va="center")
    ax = cv.axes(13.4, y0 + 1.6, 6.3, 7.6)
    xs = np.array([cp[f"{l}:{h}"]["embedding"] for l, h in heads])
    ys = np.array([v["raise"] for v in heads.values()])
    ax.scatter(xs, ys, s=50, color=CLOUD, zorder=2)
    for k in marked:
        x, y = cp[f"{k[0]}:{k[1]}"]["embedding"], heads[k]["raise"]
        ax.scatter([x], [y], s=340, color=sign[k], zorder=3, edgecolor="white", lw=2)
        ax.annotate(f"L{k[0]}·{k[1]}", (x, y), xytext=(14, 0), textcoords="offset points", va="center", fontsize=23,
                    color=sign[k], weight="bold")
    ax.axhline(0, color=SLATE, lw=1.2)
    ax.axvline(0, color=SLATE, lw=1.2)
    ax.set_xlabel("copying gain, weights only")
    ax.set_ylabel("effect alone (nats)")

    cv.letter(20.6, top - 0.6, "e")
    cv.S.text(21.4, top - 0.6, "a dial", fontsize=29, weight="bold", va="center")
    ax = cv.axes(22.2, y0 + 1.6, 4.0, 7.6)
    for g, rows in dose.items():
        if g == "all":
            col, lab = INK, "whole state"
        else:
            hs = [tuple(map(int, p.split(":"))) for p in g.split(",")]
            vals = [heads.get(k, {"raise": 0})["raise"] for k in hs]
            col, lab = (BLUE, "promoter") if all(v > 0 for v in vals) else (CORAL, "suppressors") if all(v < 0 for v in vals) else (SLATE, g)
        a = [r["dose"] for r in rows]
        ax.plot(a, [r["raise"] for r in rows], "-o", color=col, lw=4, ms=10)
        ax.text(a[-1] + 0.4, rows[-1]["raise"], lab, color=col, fontsize=22, va="center", weight="bold")
    ax.axhline(0, color=SLATE, lw=1.2)
    ax.axvline(1, color=SLATE, lw=1.2, ls=(0, (4, 3)))
    ax.set_xlabel("dose (1 = as retained)")
    ax.set_ylabel("effect (nats)")


def row_fourlayer(cv, pd_path, dose4_path, y0):
    """Rows f-g, occupying [y0, y0 + ROWS[2]]."""
    top = y0 + ROWS[2]
    d = json.load(open(pd_path))
    base = d.get("screen_base", 0.2712)
    order = ["h.2.attn.k_proj", "h.2.attn.v_proj", "h.2.attn.q_proj", "h.2.attn.o_proj",
             "h.3.attn.k_proj", "h.3.attn.v_proj", "h.3.attn.q_proj", "h.3.attn.o_proj"]
    names = {"k_proj": "key", "v_proj": "value", "q_proj": "query", "o_proj": "output"}
    where = {"word": "at word", "middle": "after it", "cue": "at cue"}
    cv.letter(0.1, top - 0.6, "f")
    cv.S.text(0.9, top - 0.6, "4-layer model: delete one decomposition subcomponent", fontsize=29, weight="bold",
              va="center")
    ax = cv.axes(2.2, y0 + 1.9, 15.6, 7.0)
    rng = np.random.default_rng(0)
    for k, site in enumerate(order):
        s = d["sites"][site]
        idx = [i for i, x in enumerate(s["screen_drops"]) if x is not None]
        drops = np.array([s["screen_drops"][i] for i in idx]) / base * 100
        top2 = [j for j in np.argsort(drops)[::-1][:2] if drops[j] > 8]
        lo = int(np.argmin(drops))
        marked = set(top2) | ({lo} if drops[lo] < -8 else set())
        rest = np.array([x for j, x in enumerate(drops) if j not in marked])
        ax.scatter(k + rng.uniform(-0.18, 0.18, rest.size), rest, s=40, color=CLOUD, zorder=2)
        for j in top2:
            ax.scatter([k], [drops[j]], s=230, color=BLUE, zorder=3, edgecolor="white", lw=2)
        if drops[lo] < -8:
            ax.scatter([k], [drops[lo]], s=230, color=CORAL, zorder=3, edgecolor="white", lw=2)
    ax.axhline(0, color=SLATE, lw=1.4, ls=(0, (4, 3)))
    ax.axvline(3.5, color=SLATE, lw=1.2)
    ax.set_xticks(range(len(order)), [f"{names[s.split('.')[-1]]}\n{where[d['sites'][s]['positions']]}" for s in order],
                  fontsize=19)
    ax.text(1.5, 108, "layer 2: copy forward", ha="center", fontsize=24, weight="bold")
    ax.text(5.5, 108, "layer 3: read at cue", ha="center", fontsize=24, weight="bold")
    ax.set_ylim(-110, 118)
    ax.set_yticks([-100, -50, 0, 50, 100])
    ax.set_ylabel("% of effect removed")

    cv.letter(19.0, top - 0.6, "g")
    cv.S.text(19.8, top - 0.6, "amplify the carried state", fontsize=29, weight="bold", va="center")
    ax = cv.axes(20.6, y0 + 1.9, 6.6, 7.0)
    dd = json.load(open(dose4_path))
    for arm, col, lab in (("state", INK, "whole state"), ("readers", BLUE, "2 reader heads")):
        rows = [r for r in dd[arm] if r["dose"] >= 0]
        ax.plot([r["dose"] for r in rows], [100 * r["top1"] for r in rows], "-o", color=col, lw=4, ms=10)
        ax.text(rows[-1]["dose"] + 0.3, 100 * rows[-1]["top1"], lab, color=col, fontsize=22, va="center", weight="bold")
    ax.axhline(2, color=SLATE, lw=1.6, ls=(0, (4, 3)))
    ax.text(8, 4, "chance", color=SLATE, fontsize=20, ha="right")
    ax.axvline(1, color=SLATE, lw=1.2, ls=(0, (4, 3)))
    ax.set_xlim(-0.4, 12.5)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_xlabel("dose (1 = as retained)")
    ax.set_ylabel("hidden word named first (%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--localize", required=True)
    ap.add_argument("--copying", required=True)
    ap.add_argument("--dose", required=True)
    ap.add_argument("--pd4l", required=True)
    ap.add_argument("--dose4l", required=True)
    ap.add_argument("--out", default="paper/figs/main.png")
    a = ap.parse_args()
    fig = plt.figure(figsize=(W, H))
    cv = Canvas(fig)
    row_experiment(cv, a.results, ROWS[2] + ROWS[1])
    row_mechanism(cv, a.localize, a.copying, a.dose, ROWS[2])
    row_fourlayer(cv, a.pd4l, a.dose4l, 0)
    for y in (ROWS[2] + ROWS[1], ROWS[2]):
        cv.S.plot([0.3, W - 0.3], [y, y], color="#e3e5e8", lw=2)
    fig.savefig(a.out, dpi=110)


if __name__ == "__main__":
    main()
