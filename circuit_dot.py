"""Circuit diagrams of Goodfire's 4-layer model laid out by Graphviz (dot), for figure_main.py.

gates_dot: the recall circuit as parts and wires: the two attention steps (head 2.3 copies the word into the later
  tokens; heads 3.4 and 3.5 read them where the word is recalled) and the VPD subcomponents measured to switch them on or off
  (fourlayer/attention_gates.py, fourlayer/subcomponent_circuit.py).
render(dot, path, snap): dot -> PNG; label nodes moved up under their wires after layout.
"""
import json
import subprocess

import matplotlib.image as mpimg

BLUE, CORAL, INK, SLATE, CLOUD, PALE_BLUE, PALE_SLATE = "#2f6db5", "#e8684a", "#1d1d1f", "#8e959c", "#b9bfc6", "#dce8f6", "#eceef0"
FONT = "Avenir Next"
DOT, NEATO = "/opt/homebrew/bin/dot", "/opt/homebrew/bin/neato"


def render(dot, path, dpi=220, snap=()):
    """dot -> PNG; returns the image array.  snap: (label node, tail, head) -- after layout, each label node is moved up
    until its top sits just under the lowest wire from tail to head (dot leaves a full node gap there)."""
    with open(path + ".dot", "w") as f:
        f.write(dot)
    if not snap:
        subprocess.run([DOT, "-Tpng", f"-Gdpi={dpi}", "-o", path + ".png", path + ".dot"], check=True)
        return mpimg.imread(path + ".png")
    laid = subprocess.run([DOT, "-Tdot", path + ".dot"], check=True, capture_output=True, text=True).stdout
    js = json.loads(subprocess.run([DOT, "-Tjson", path + ".dot"], check=True, capture_output=True, text=True).stdout)
    obj = {o["_gvid"]: o for o in js["objects"]}
    by_name = {o["name"]: o for o in js["objects"]}
    moves = []
    for lab, a, b in snap:
        o = by_name[lab]
        x, y = map(float, o["pos"].split(","))
        w, h = 72 * float(o["width"]), 72 * float(o["height"])
        pts = [tuple(map(float, t.split(",")[-2:])) for e in js["edges"] if (obj[e["tail"]]["name"], obj[e["head"]]["name"]) == (a, b)
               for t in e["pos"].split()]
        under = [py for px, py in pts if x - w / 2 <= px <= x + w / 2] or [py for _, py in pts]
        moves.append(f'  {lab} [pos="{x:.1f},{max(y, min(under) - 4 - h / 2):.1f}"];')
    with open(path + ".dot", "w") as f:
        f.write(laid.rstrip().rstrip("}") + "\n" + "\n".join(moves) + "\n}\n")
    subprocess.run([NEATO, "-n2", "-Tpng", f"-Gdpi={dpi}", "-o", path + ".png", path + ".dot"], check=True)
    return mpimg.imread(path + ".png")


def width(d):
    return f"{1.6 + 0.9 * abs(d):.2f}"


def gates_dot(g):
    """Parts named by layer, matrix and position; wire width grows with the recall change when the part is removed.
    Returns the dot source and the bundles' (label node, tail, head) for render(snap=...)."""
    base = 100 * g["recall"]
    eff = {k: 100 * v["recall"] - base for k, v in g["gates"].items()}
    eff.update({k: -100 * v for k, v in g.get("screen", {}).items()})
    E = lambda *keys: sum(eff.get(k, 0.0) for k in keys) / max(1, sum(k in eff for k in keys))
    need = lambda d: f'color="{BLUE}", penwidth={width(d)}, arrowhead=normal, arrowsize=1.1'
    block = lambda d: f'color="{CORAL}", penwidth={width(d)}, arrowhead=tee, arrowsize=1.4'
    info = f'color="{CLOUD}", penwidth=2.2, style=dashed, arrowhead=normal, arrowsize=0.9'
    token = f'style="rounded,filled", fillcolor="{PALE_SLATE}", color="{PALE_SLATE}", fontcolor="{INK}"'
    part = f'style="rounded,filled", fillcolor="white", penwidth=2.2'
    head = f'style="rounded,filled,bold", fillcolor="{PALE_BLUE}", color="{BLUE}", fontcolor="{INK}", fontsize=33, penwidth=2.6'

    snap = []

    def bundle(a, b, label, k=10):
        """Many thin wires: information carried by many subcomponents, none of them needed alone.  The label is a node
        between the two ends, so dot sets it beside the wires and keeps other nodes clear of it."""
        snap.append((f"{a}_{b}", a, b))
        wires = [f'  {a} -> {b} [color="{SLATE}", penwidth=0.9, arrowsize=0.55];' for _ in range(k)]
        wires.append(f'  {a}_{b} [shape=plaintext, label="{label}", fontcolor="{SLATE}", fontsize=26, margin=0];')
        wires.append(f'  {a} -> {a}_{b} -> {b} [style=invis, weight=3];')
        return "\n".join(wires)
    return f"""digraph G {{
  rankdir=LR; splines=spline; nodesep=0.3; ranksep=0.45; bgcolor="white"; pad=0.1;
  node [shape=box, fontname="{FONT}", fontsize=30, margin="0.18,0.1"];
  edge [fontname="{FONT}", fontsize=26, fontcolor="{SLATE}"];
  {{ rank=same; word [label="hidden word\n“otter”", {token}]; later [label="later tokens\n“. Nobody else knows.”", {token}];
    cue [label="where the word is recalled\n“My pet is a”", {token}]; }}
  {{ rank=same; k2 [label="layer 2 keys\nat the hidden word\n(2 subcomponents)", color="{BLUE}", fontcolor="{BLUE}", {part}];
    q2 [label="layer 2 query\nat the later tokens", color="{BLUE}", fontcolor="{BLUE}", {part}]; }}
  h2 [label="layer 2, head 3\ncopies the word\ninto the later tokens", {head}];
  {{ rank=same; k3 [label="layer 3 key\nat the later tokens", color="{BLUE}", fontcolor="{BLUE}", {part}];
    q3 [label="layer 3 query\nwhere the word\nis recalled", color="{BLUE}", fontcolor="{BLUE}", {part}];
    b3 [label="layer 3 queries\nwhere the word\nis recalled\n(2 subcomponents)", color="{CORAL}", fontcolor="{CORAL}", {part}]; }}
  h3 [label="layer 3, heads 4 and 5\nread the later tokens\nwhere the word is recalled", {head}];
  answer [label="answer:\n“otter”", style="rounded,filled,bold", fillcolor="{PALE_SLATE}", color="{PALE_SLATE}", fontcolor="{INK}", fontsize=33];
  word -> k2 [{info}]; later -> q2 [{info}]; later -> k3 [{info}]; cue -> q3 [{info}]; cue -> b3 [{info}];
{bundle("word", "h2", "78 value\\nsubcomponents")}
  k2 -> h2 [{need(E("h.2.attn.k_proj#224@word", "h.2.attn.k_proj#206@word"))}];
  q2 -> h2 [{need(E("h.2.attn.q_proj#436@later"))}];
{bundle("h2", "h3", "the copied word:\\nmany output and\\nvalue subcomponents")}
  k3 -> h3 [{need(E("h.3.attn.k_proj#145@later"))}];
  q3 -> h3 [{need(E("h.3.attn.q_proj#182@cue"))}];
  b3 -> h3 [{block(E("h.3.attn.q_proj#334@cue", "h.3.attn.q_proj#60@cue"))}];
{bundle("h3", "answer", "many output\\nsubcomponents")}
}}
""", snap


def qwen_dot():
    """Qwen3-1.7B, from head removals and weight edits (edit_heads.py, attention.py): two layer-21 heads read the reply's
    "." token, where the hidden animal is held, and push the answer in opposite directions; the question's wording sets
    how much the suppressing head reads."""
    token = f'style="rounded,filled", fillcolor="{PALE_SLATE}", color="{PALE_SLATE}", fontcolor="{INK}"'
    return f"""digraph G {{
  rankdir=BT; splines=spline; nodesep=0.55; ranksep=0.6; bgcolor="white"; pad=0.2;
  node [shape=box, fontname="{FONT}", fontsize=19, margin="0.2,0.1"];
  edge [fontname="{FONT}", fontsize=16, fontcolor="{SLATE}"];
  {{ rank=same; reply [label="reply “I understand.”\\n(its cache kept from turn 1;\\nthe “.” holds the hidden animal)", {token}];
    question [label="the question\\n“Which animal did you choose?”", {token}]; }}
  {{ rank=same; h0 [label="layer 21, key-value head 0\ncopies the hidden animal", style="rounded,filled,bold", fillcolor="{PALE_BLUE}", color="{BLUE}", fontcolor="{INK}", penwidth=2.6];
    h6 [label="layer 21, key-value head 6\nwrites the hidden animal's opposite", style="rounded,filled,bold", fillcolor="#fbe3dc", color="{CORAL}", fontcolor="{INK}", penwidth=2.6]; }}
  answer [label="answer", style="rounded,filled,bold", fillcolor="{PALE_SLATE}", color="{PALE_SLATE}", fontcolor="{INK}", fontsize=21];
  reply -> h0 [color="{CLOUD}", style=dashed, penwidth=2.2, label=" reads"];
  reply -> h6 [color="{CLOUD}", style=dashed, penwidth=2.2, label=" reads"];
  question -> h6 [color="{CORAL}", penwidth=3, label=" makes it read the reply\n 2.6 times more than\n “Recall, introspect…”"];
  h0 -> answer [color="{BLUE}", penwidth=4, label=" toward the\n hidden animal"];
  h6 -> answer [color="{CORAL}", penwidth=5, arrowhead=tee, arrowsize=1.4, label=" away from the\n hidden animal"];
}}
"""
