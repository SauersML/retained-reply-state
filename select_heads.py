"""Pick the heads to test, from a localize.py run on the discovery data: key/value heads whose retained state alone
raises the chosen animal (promoters) or lowers it (suppressors) with |animal-level z| > 5, at most three of each,
strongest first.  Prints  promoters;suppressors  as layer:kv_head lists (for dose.py and weights.py)."""
import json
import sys

arms = json.load(open(sys.argv[1]))["arms"]
heads = [(k.split()[1], k.split()[3], v) for k, v in arms.items() if k.startswith("layer ") and "kv-head" in k]
pro = sorted([h for h in heads if h[2]["z"] > 5], key=lambda h: -h[2]["raise"])[:3]
sup = sorted([h for h in heads if h[2]["z"] < -5], key=lambda h: h[2]["raise"])[:3]
print(",".join(f"{l}:{h}" for l, h, _ in pro) + ";" + ",".join(f"{l}:{h}" for l, h, _ in sup))
