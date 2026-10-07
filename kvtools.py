"""Key/value cache operations: moving cached keys to new positions, assembling caches, scoring answers."""
import numpy as np
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
    used once, without per-candidate copies.  With a batch of contexts of equal length (cache batch B, last_logp
    [B, vocab]) it returns one row of scores per context; with a single context, one list."""
    single = last_logp.dim() == 1
    last_logp = last_logp[None] if single else last_logp
    B, P = cache_layers[0][0].shape[0], cache_layers[0][0].shape[2]
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
    logits = model(torch.tensor([ids] * B, device=device), position_ids=torch.tensor([pos] * B, device=device),
                   attention_mask=mask.expand(B, 1, T, P + T), past_key_values=cache, use_cache=False).logits
    first = torch.tensor([toks[0] for toks in candidates])
    rows, cols, start = [], [], []
    q = 0
    for toks in candidates:
        start.append(len(rows))
        for t in range(1, len(toks)):
            rows.append(q + t - 1)
            cols.append(toks[t])
        q += len(toks)
    out = []
    for b in range(B):
        rest = (logits[b, rows].float().log_softmax(-1)[torch.arange(len(rows)), torch.tensor(cols)]).cpu().numpy() if rows else []
        lp = last_logp[b].float().cpu()[first].numpy()
        out.append([float(lp[c]) + float(sum(rest[start[c]:start[c] + len(toks) - 1])) for c, toks in enumerate(candidates)])
    return out[0] if single else out


def recall_logp(model, prefix_kv, kv_batch, suffix, forms, names, device):
    """log P(each name) after `suffix` for a batch of caches of equal length: prefix_kv (per-layer keys/values,
    batch 1) is shared, kv_batch[b] holds run b's per-layer keys/values that follow it.  forms are (name, token ids)
    pairs; a name's score is the log-sum-exp over its forms.  Returns an array [len(kv_batch), len(names)]."""
    B = len(kv_batch)
    layers = []
    for l, (pk, pv) in enumerate(prefix_kv):
        k = torch.cat([pk.expand(B, -1, -1, -1), torch.cat([kv[l][0] for kv in kv_batch], 0)], 2)
        v = torch.cat([pv.expand(B, -1, -1, -1), torch.cat([kv[l][1] for kv in kv_batch], 0)], 2)
        layers.append((k.to(device), v.to(device)))
    out = model(torch.tensor([suffix] * B, device=device), past_key_values=DynamicCache.from_legacy_cache(tuple(layers)),
                use_cache=True)
    last = torch.log_softmax(out.logits[:, -1].float(), -1)
    scores = score_candidates(model, layers_of(out.past_key_values), last, [t for _, t in forms], device)
    res = np.full((B, len(names)), -np.inf)
    col = {n: i for i, n in enumerate(names)}
    for b in range(B):
        for (n, _), s in zip(forms, scores[b]):
            res[b, col[n]] = np.logaddexp(res[b, col[n]], s)
    return res
