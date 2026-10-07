"""Which residual-stream directions each key/value head of a layer reads, from the weights alone: for every pair of
heads, the overlap of their value maps' input subspaces (the top right-singular vectors of W_V^h diag(g) holding 90% of
its energy; overlap = mean squared cosine of the principal angles).  Saved with each head's copying gain (copying.py).
usage: overlap.py MODEL LAYER COPYING.json OUT.json"""
import itertools
import json
import sys

import numpy as np
import scipy.linalg
import torch
from transformers import AutoModelForCausalLM


def value_basis(Wv):
    """Orthonormal rows spanning the top right-singular vectors of one head's W_V diag(g) that hold 90% of its energy
    (SciPy in float64: the Accelerate LAPACK behind torch.linalg on macOS is not trusted)."""
    _, S, Vt = scipy.linalg.svd(np.asarray(Wv, dtype=np.float64), full_matrices=False)
    return Vt[:int(np.searchsorted(np.cumsum(S ** 2) / np.sum(S ** 2), 0.9)) + 1]


def subspace_overlap(A, B):
    """Mean squared cosine of the principal angles between the row spaces of A and B (orthonormal rows)."""
    s = scipy.linalg.svdvals(A @ B.T)
    return float((s ** 2).sum() / min(len(A), len(B)))


def main(model_name, layer, copying, out):
    m = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, device_map="cpu")
    cfg = m.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kvh = cfg.num_key_value_heads
    L = m.model.layers[layer]
    Wv = (L.self_attn.v_proj.weight.detach().float() * L.input_layernorm.weight.detach().float()).numpy()
    basis = {h: value_basis(Wv[h * hd:(h + 1) * hd]) for h in range(kvh)}
    O = np.eye(kvh)
    for a, b in itertools.combinations(range(kvh), 2):
        O[a, b] = O[b, a] = subspace_overlap(basis[a], basis[b])
    cp = json.load(open(copying))["kv_heads"]
    gains = [cp[f"{layer}:{h}"]["embedding"] for h in range(kvh)]
    json.dump({"model": model_name, "layer": layer, "overlap": O.tolist(), "copying_gain": gains}, open(out, "w"), indent=1)
    print(np.round(O, 2))
    print("copying gains:", np.round(gains, 1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
