"""Figures from results/*.json (hidden_choice.py outputs).

own_vs_other.png   per model and arm: the chosen animal's raise against every other animal's, one point per animal
rank_curves.png    fraction of animals whose own raise ranks in the top k of their row, per model (retained arm)
"""
import glob
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 18, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
GRAY = "#8a8984"


def animal_matrix(d, arm):
    L = np.array(d["arms"][arm])
    c = np.array([d["animals"].index(x) for x in d["chosen"]])
    present = np.unique(c)
    M = np.array([L[c == a].mean(0)[present] for a in present])
    return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()


def main(paths):
    runs = sorted((json.load(open(p)) for p in paths), key=lambda d: d["model"])
    arms = [a for a in ("retained", "neutral") if all(a in d["arms"] for d in runs)]
    panels = [(d, a) for d in runs for a in arms]
    fig, ax = plt.subplots(figsize=(max(10, 3.2 * len(panels)), 7.5))
    rng = np.random.default_rng(0)
    for k, (d, a) in enumerate(panels):
        M = animal_matrix(d, a)
        off, diag = M[~np.eye(len(M), dtype=bool)], np.diag(M)
        ax.scatter(k - 0.18 + rng.uniform(-0.12, 0.12, off.size), off, s=3, color=GRAY, alpha=0.2)
        ax.scatter(k + 0.18 + rng.uniform(-0.08, 0.08, diag.size), diag, s=36,
                   color=COLORS[runs.index(d) % len(COLORS)], alpha=0.85, edgecolor="white", lw=0.5)
        ax.plot([k + 0.05, k + 0.31], [diag.mean()] * 2, color="black", lw=3)
        ax.plot([k - 0.31, k - 0.05], [off.mean()] * 2, color="black", lw=3)
    ax.axhline(0, color=GRAY, ls="--", lw=1.5)
    ax.set_xticks(range(len(panels)), [f"{d['model'].split('/')[-1]}\n{a}" for d, a in panels], fontsize=15)
    ax.set_ylabel("log P at recall vs that animal's\naverage over contexts (nats)")
    fig.suptitle("Chosen animal (colored, one point per animal) versus every other animal (gray)", x=0.02, ha="left", fontsize=20)
    fig.tight_layout()
    fig.savefig("results/own_vs_other.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 7))
    for d in runs:
        if "retained" not in d["arms"]:
            continue
        M = animal_matrix(d, "retained")
        n = len(M)
        rank = np.array([(M[i] > M[i, i]).sum() + 1 for i in range(n)])
        ks = np.arange(1, n + 1)
        ax.plot(ks, [np.mean(rank <= k) for k in ks], lw=3, color=COLORS[runs.index(d) % len(COLORS)],
                label=d["model"].split("/")[-1])
    ax.plot(ks, ks / n, color=GRAY, ls="--", lw=2, label="chance")
    ax.set_xlabel("k (rank of the chosen animal among the 50 at recall)")
    ax.set_ylabel("Animals ranked in the top k")
    ax.legend(frameon=False)
    fig.suptitle("Retained reply state: where the chosen animal ranks at recall", x=0.02, ha="left", fontsize=20)
    fig.tight_layout()
    fig.savefig("results/rank_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:] or glob.glob("results/*.json"))
