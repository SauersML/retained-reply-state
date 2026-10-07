"""The mechanism figure.  (a) every key/value head's own-animal effect when only it keeps the retained reply state;
(b) weights-only copying gain against that effect; (c) dose-response of the selected heads.
usage: fig_mechanism.py LOCALIZE.json COPYING.json DOSE.json OUT.png"""
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

INK, BLUE, CORAL, SLATE = "#1d1d1f", "#2f6db5", "#e8684a", "#8e959c"
plt.rcParams.update({"font.family": "Avenir Next", "font.size": 26, "figure.facecolor": "white",
                     "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.6, "xtick.major.width": 1.6, "ytick.major.width": 1.6})
CMAP = LinearSegmentedColormap.from_list("cb", [CORAL, "#f7f7f5", BLUE])


def main(loc_path, copy_path, dose_path, out):
    loc = json.load(open(loc_path))["arms"]
    cp = json.load(open(copy_path))["kv_heads"]
    dose = json.load(open(dose_path))["groups"]
    heads = {}
    for k, v in loc.items():
        if k.startswith("layer ") and "kv-head" in k:
            heads[(int(k.split()[1]), int(k.split()[3]))] = v
    layers = sorted({l for l, _ in heads})
    nkv = max(h for _, h in heads) + 1
    E = np.full((len(layers), nkv), np.nan)
    for (l, h), v in heads.items():
        E[layers.index(l), h] = v["raise"]
    fig = plt.figure(figsize=(30, 10))
    # (a) head map
    ax = fig.add_axes([0.04, 0.12, 0.2, 0.78])
    lim = np.nanmax(np.abs(E))
    ax.imshow(E, cmap=CMAP, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto", origin="lower")
    ax.set_xticks(range(nkv))
    ax.set_yticks(range(0, len(layers), max(1, len(layers) // 6)), [layers[i] for i in range(0, len(layers), max(1, len(layers) // 6))])
    ax.set_xlabel("key/value head")
    ax.set_ylabel("layer")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("one head at a time", fontsize=27, loc="left", weight="bold")
    top = sorted(heads, key=lambda k: -abs(heads[k]["raise"]))[:3]
    for l, h in top:
        ax.add_patch(plt.Rectangle((h - 0.5, layers.index(l) - 0.5), 1, 1, fill=False, edgecolor=INK, lw=3))
    # (b) weights vs effect
    ax = fig.add_axes([0.32, 0.12, 0.28, 0.78])
    xs = np.array([cp[f"{l}:{h}"]["embedding"] for l, h in heads])
    ys = np.array([v["raise"] for v in heads.values()])
    ax.scatter(xs, ys, s=60, color="#c9cdd1", zorder=2)
    for l, h in top:
        x, y = cp[f"{l}:{h}"]["embedding"], heads[(l, h)]["raise"]
        ax.scatter([x], [y], s=320, color=BLUE if y > 0 else CORAL, zorder=3, edgecolor="white", lw=2)
        ax.annotate(f"L{l}·{h}", (x, y), xytext=(14, 0), textcoords="offset points", va="center", fontsize=24,
                    color=BLUE if y > 0 else CORAL, weight="bold")
    ax.axhline(0, color=SLATE, lw=1.2)
    ax.axvline(0, color=SLATE, lw=1.2)
    ax.set_xlabel("copying gain, from the weights alone")
    ax.set_ylabel("effect on the hidden animal (nats)")
    ax.set_title("weights predict the sign", fontsize=27, loc="left", weight="bold")
    # (c) dose-response
    ax = fig.add_axes([0.68, 0.12, 0.3, 0.78])
    names = list(dose)
    style = {}
    for g in names:
        if g == "all":
            style[g] = (INK, "whole reply state")
        elif all(heads.get(tuple(map(int, p.split(":"))), {"raise": 0})["raise"] > 0 for p in g.split(",")):
            style[g] = (BLUE, "promoter")
        elif all(heads.get(tuple(map(int, p.split(":"))), {"raise": 0})["raise"] < 0 for p in g.split(",")):
            style[g] = (CORAL, "suppressors")
        else:
            style[g] = (SLATE, g)
    for g in names:
        col, lab = style[g]
        a = [r["dose"] for r in dose[g]]
        r = [r["raise"] for r in dose[g]]
        ax.plot(a, r, "-o", color=col, lw=4, ms=12)
        ax.text(a[-1] + 0.25, r[-1], lab, color=col, fontsize=24, va="center", weight="bold")
    ax.axhline(0, color=SLATE, lw=1.2)
    ax.axvline(1, color=SLATE, lw=1.2, ls=(0, (4, 3)))
    ax.set_xlabel("dose  (0 = removed, 1 = as retained)")
    ax.set_ylabel("effect on the hidden animal (nats)")
    ax.set_xlim(right=ax.get_xlim()[1] + 3.2)
    ax.set_title("a dial", fontsize=27, loc="left", weight="bold")
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main(*sys.argv[1:5])
