"""Is the chosen animal linearly decodable from the reply's residual stream?

Multinomial logistic regression (standardised, PCA to 64 dimensions) predicts the chosen animal from the residual
stream at the reply's last text token, per layer.  Each distinct thinking text appears once, and the 5 folds hold out
whole contexts, so no test context was seen in training.  Null: the same pipeline with shuffled labels."""
import argparse
import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def accuracy(X, y, seed=0):
    pred = np.empty_like(y)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        clf = make_pipeline(StandardScaler(), PCA(n_components=min(64, len(tr) - 1), random_state=seed),
                            LogisticRegression(max_iter=3000, C=0.1))
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return float(np.mean(pred == y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--null", type=int, default=10)
    a = ap.parse_args()
    d = json.load(open(a.result))
    H = np.load(a.result.replace(".json", "_states.npy")).astype(np.float32)
    first = {}
    for i, key in enumerate(zip(d["chosen"], d["thinking"])):
        first.setdefault(key, i)
    keep = np.array(sorted(first.values()))
    X, y = H[keep], np.array([d["animals"].index(d["chosen"][i]) for i in keep])
    rng = np.random.default_rng(0)
    print(f"{d['model']}: {len(keep)} distinct contexts, chance {1 / len(np.unique(y)):.3f}")
    for layer in range(0, H.shape[1], 4):
        acc = accuracy(X[:, layer], y)
        null = [accuracy(X[:, layer], rng.permutation(y), seed=s) for s in range(a.null)]
        print(f"layer {layer:2d}: {acc:.3f} (shuffled labels: mean {np.mean(null):.3f}, max {np.max(null):.3f})", flush=True)


if __name__ == "__main__":
    main()
