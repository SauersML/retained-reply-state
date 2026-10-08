"""Discrimination accuracy: from one run's recall answer, can the hidden animal be told apart from another animal?

For run i with hidden animal c_i, each animal's log P at recall is taken relative to that animal's mean over all runs,
D[i, b] = L[i, b] - mean_j L[j, b].  The run's accuracy is the share of the other 49 animals b with D[i, c_i] > D[i, b]
(ties count half); the reported accuracy is the mean over animals of their runs' mean (50% = chance, 100% = always
ranked first, below 50% = the answer points away from the hidden animal).  With groups, each animal is centered within
its group of runs instead (the 4-layer model's templates).  Test: the same statistic with the hidden
animals re-paired at random among the runs' animal groups (exact under the null, since the hidden animal is drawn
uniformly at random), 1,000 re-pairings.
"""
import json
import sys

import numpy as np


def centred(L, groups=None):
    """Each animal's log P relative to its mean over all runs, or over the runs of the same group (groups: one id per
    run; used where runs come in balanced groups, e.g. the 4-layer model's templates, each with every word once)."""
    if groups is None:
        return L - L.mean(0)
    D = np.empty_like(L, dtype=float)
    for g in np.unique(groups):
        D[groups == g] = L[groups == g] - L[groups == g].mean(0)
    return D


def accuracy(L, c, perm=None, groups=None):
    D = centred(L, groups)
    present = np.unique(c)
    lab = c if perm is None else perm[np.searchsorted(present, c)]
    own = D[np.arange(len(c)), lab]
    diff = own[:, None] - D                                     # differences under 1e-9 nats are ties (rounding)
    acc = ((diff > 1e-9).sum(1) + 0.5 * ((np.abs(diff) <= 1e-9).sum(1) - 1)) / (D.shape[1] - 1)
    return np.mean([acc[c == a].mean() for a in present])


def main(paths):
    rng = np.random.default_rng(0)
    for p in paths:
        d = json.load(open(p))
        c = np.array([d["animals"].index(x) for x in d["chosen"]])
        present = np.unique(c)
        name = d.get("model", p.split("/")[-1])
        for arm, L in d["arms"].items():
            L = np.array(L)
            if np.allclose(L, L[:1]):
                print(f"{name:42s} {arm:17s} 50.0% (identical contexts)")
                continue
            obs = accuracy(L, c)
            null = np.array([accuracy(L, c, rng.permutation(present)) for _ in range(1000)])
            pv = (1 + np.sum(np.abs(null - 0.5) >= abs(obs - 0.5))) / 1001
            print(f"{name:42s} {arm:17s} {100 * obs:5.1f}%  (null {100 * null.mean():.1f} +- {100 * null.std():.1f}, two-sided p {pv:.3g})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
