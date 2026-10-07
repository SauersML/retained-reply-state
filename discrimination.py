"""Discrimination accuracy: from one run's recall answer, can the hidden animal be told apart from another animal?

For run i with hidden animal c_i, each animal's log P at recall is taken relative to that animal's mean over all runs,
D[i, b] = L[i, b] - mean_j L[j, b].  The run's accuracy is the share of the other 49 animals b with D[i, c_i] > D[i, b]
(ties count half); the reported accuracy is the mean over animals of their runs' mean (50% = chance, 100% = always
ranked first, below 50% = the answer points away from the hidden animal).  Test: the animal-level permutation null of
stats.py on the same statistic (hidden animals re-paired with runs' animal groups).
"""
import json
import sys

import numpy as np


def accuracy(L, c, perm=None):
    D = L - L.mean(0)
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
