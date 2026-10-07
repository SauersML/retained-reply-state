"""Key/value cache operations: moving cached keys to new positions, assembling caches, scoring answers."""
import torch
from transformers import DynamicCache


def layers_of(cache):
    """(keys, values) per layer, each [batch, kv_heads, positions, head_dim]."""
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def shift_keys(keys, delta, inv_freq):
    """Move rotary-encoded keys by `delta` positions: rotations compose, so R(p + delta) k = R(delta) R(p) k.
    Values carry no position and need no change."""
    angle = delta * inv_freq.to(torch.float64)
    cos = torch.cat([angle.cos(), angle.cos()]).to(torch.float32)
    sin = torch.cat([angle.sin(), angle.sin()]).to(torch.float32)
    k = keys.to(torch.float32)
    return (k * cos + rotate_half(k) * sin).to(keys.dtype)


def make_cache(parts, device):
    """Concatenate per-layer (keys, values) pieces along positions into one cache for a batch of one."""
    layers = []
    for pieces in zip(*parts):
        layers.append((torch.cat([k for k, _ in pieces], 2).to(device), torch.cat([v for _, v in pieces], 2).to(device)))
    return DynamicCache.from_legacy_cache(tuple(layers))


def score_candidates(model, cache_layers, last_logp, candidates, device):
    """Summed log-probability of each candidate continuation of a cached context, in one forward pass.

    Candidates are packed into one sequence.  A 4-D mask lets each candidate token attend to the cached context and to
    the earlier tokens of its own candidate; every candidate starts at the position after the context.  The cache is
    used once, without per-candidate copies."""
    P = cache_layers[0][0].shape[2]
    ids, pos, owner = [], [], []
    for c, toks in enumerate(candidates):
        for t, tok in enumerate(toks):
            ids.append(tok)
            pos.append(P + t)
            owner.append(t)
    T = len(ids)
    dtype = cache_layers[0][0].dtype
    mask = torch.full((1, 1, T, P + T), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask[..., :P] = 0
    for q, t in enumerate(owner):
        mask[0, 0, q, P + q - t:P + q + 1] = 0
    cache = DynamicCache.from_legacy_cache(tuple((k.to(device), v.to(device)) for k, v in cache_layers))
    logits = model(torch.tensor([ids], device=device), position_ids=torch.tensor([pos], device=device),
                   attention_mask=mask, past_key_values=cache, use_cache=False).logits[0]
    logp = torch.log_softmax(logits.float(), -1)
    scores, q = [], 0
    for toks in candidates:
        s = float(last_logp[toks[0]])
        for t in range(1, len(toks)):
            s += float(logp[q + t - 1, toks[t]])
        scores.append(s)
        q += len(toks)
    return scores
