"""Does a recall context raise the probability of the animal chosen in its own run?

L[i, b] = log P(animal b | recall context of run i); c_i = the animal chosen in run i.
Raise of run i: L[i, c_i] minus the mean of L[j, c_i] over all runs j (how much this context favours its own animal
compared with the average context).

Two one-sided permutation tests of a positive mean raise:
  run level     units = runs; re-pairs contexts with chosen animals.  Valid when contexts are exchangeable given
                no information, but anti-conservative when several runs share near-identical contexts.
  animal level  units = the 50 animals; M[a, b] = mean of L[i, b] over runs that chose a, double-centred; the
                statistic is the mean of the diagonal, the null re-pairs rows with columns.  Treats each animal's
                effect as a random draw, so it stays valid when contexts cluster by animal.
Calibration (--calibrate): the same tests on placebo labels that carry no own-animal information, in two forms:
runs' labels shuffled, and animals relabelled as a whole (which keeps the clustering by animal).
"""
import argparse
import json

import numpy as np


def raises(L, c):
    return L[np.arange(len(c)), c] - L.mean(0)[c]


def run_level(L, c, rng, B=20000):
    n = len(c)
    S = L[:, c]
    rows, cols = S.sum(1), S.sum(0)

    def stat(p):
        d = S[np.arange(n), p]
        return np.mean(d - (rows - d) / (n - 1) - (cols[p] - d) / (n - 1))

    obs = stat(np.arange(n))
    null = np.array([stat(rng.permutation(n)) for _ in range(B)])
    return (obs - null.mean()) / null.std(), (1 + np.sum(null >= obs)) / (B + 1)


def animal_level(L, c, rng, B=20000):
    present = np.unique(c)
    M = np.array([L[c == a].mean(0)[present] for a in present])
    M = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()
    k = len(present)
    obs = np.mean(np.diag(M))
    null = np.array([np.mean(M[np.arange(k), rng.permutation(k)]) for _ in range(B)])
    return (obs - null.mean()) / null.std(), (1 + np.sum(null >= obs)) / (B + 1)


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def calibrate(L, c, rng, K=200, B=1000):
    out = {}
    for name in ("runs shuffled", "animals relabelled"):
        hits = []
        for _ in range(K):
            pc = c[derangement(len(c), rng)] if name == "runs shuffled" else derangement(L.shape[1], rng)[c]
            hits.append((run_level(L, pc, rng, B)[1] < 0.05, animal_level(L, pc, rng, B)[1] < 0.05))
        out[name] = np.mean(hits, axis=0)
    return out


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.empty_like(p)
    run = 1.0
    for rank in range(len(p), 0, -1):
        run = min(run, p[order[rank - 1]] * len(p) / rank)
        q[order[rank - 1]] = run
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for path in a.results:
        d = json.load(open(path))
        c = np.array([d["animals"].index(x) for x in d["chosen"]])
        distinct = len(set(zip(d["chosen"], d["thinking"])))
        for arm, L in d["arms"].items():
            L = np.array(L)
            if np.allclose(L, L[:1]):
                rows.append((d["model"], arm, len(c), distinct, 0.0, np.nan, np.nan, np.nan, np.nan))
                continue
            zr, pr = run_level(L, c, rng)
            za, pa = animal_level(L, c, rng)
            rows.append((d["model"], arm, len(c), distinct, raises(L, c).mean(), zr, pr, za, pa))
            if a.calibrate and arm in ("retained", "neutral"):
                for name, (fr, fa) in calibrate(L, c, rng).items():
                    print(f"calibration {d['model']} {arm} ({name}): p < 0.05 in {fr:.3f} (run level), {fa:.3f} (animal level)")
    tested = [i for i, r in enumerate(rows) if r[1] not in ("stripped", "visible") and np.isfinite(r[8])]
    q = dict(zip(tested, bh([rows[i][8] for i in tested])))
    print(f"{'model':18s} {'arm':9s} {'runs':>5s} {'distinct':>8s} {'raise':>9s} {'run z':>7s} {'run p':>8s} {'animal z':>9s} {'animal p':>9s} {'BH q':>7s}")
    for i, r in enumerate(rows):
        print(f"{r[0]:18s} {r[1]:9s} {r[2]:5d} {r[3]:8d} {r[4]:+9.4f} {r[5]:+7.2f} {r[6]:8.2g} {r[7]:+9.2f} {r[8]:9.2g} "
              f"{q.get(i, np.nan):7.2g}")


if __name__ == "__main__":
    main()
