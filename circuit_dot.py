"""Circuit diagrams of Goodfire's 4-layer model laid out by Graphviz (dot), for figure_main.py.

gates_dot: the recall circuit as parts and wires: the two attention steps (head 2.3 copies the word into the later
  tokens; heads 3.4 and 3.5 read them at the cue) and the VPD subcomponents measured to switch them on or off
  (fourlayer/attention_gates.py, fourlayer/subcomponent_circuit.py).
graph_dot: the circuit with VPD subcomponents as the only nodes, one rank per layer and matrix, links measured by
  removing one subcomponent and recording the change in a later one (subcomponent_circuit.py --nodes).
render(dot, path): dot -> PNG at the given resolution; returns the image array.
"""
import os
import subprocess

import matplotlib.image as mpimg

BLUE, CORAL, INK, SLATE, CLOUD, PALE_BLUE, PALE_SLATE = "#2f6db5", "#e8684a", "#1d1d1f", "#8e959c", "#b9bfc6", "#dce8f6", "#eceef0"
FONT = "Avenir Next"
KIND = {"q_proj": "query", "k_proj": "key", "v_proj": "value", "o_proj": "attention output", "c_fc": "MLP in", "down_proj": "MLP out"}
WHERE = {"frame": "the frame", "word": "the hidden word", "later": "the later tokens", "cue": "the cue"}


def render(dot, path, dpi=220):
    with open(path + ".dot", "w") as f:
        f.write(dot)
    subprocess.run(["/opt/homebrew/bin/dot", "-Tpng", f"-Gdpi={dpi}", "-o", path + ".png", path + ".dot"], check=True)
    return mpimg.imread(path + ".png")


def width(d):
    return f"{1.6 + 0.9 * abs(d):.2f}"


def gates_dot(g):
    """Parts named by layer, matrix and position; wire width grows with the recall change when the part is removed."""
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

    def bundle(a, b, label, k=10):
        """Many thin wires: information carried by many subcomponents, none of them needed alone."""
        wires = [f'  {a} -> {b} [color="{SLATE}", penwidth=0.9, arrowsize=0.55];' for _ in range(k - 1)]
        wires.append(f'  {a} -> {b} [color="{SLATE}", penwidth=0.9, arrowsize=0.55, xlabel="{label}", fontcolor="{SLATE}"];')
        return "\n".join(wires)
    return f"""digraph G {{
  rankdir=LR; splines=spline; nodesep=0.3; ranksep=0.45; bgcolor="white"; pad=0.1; forcelabels=true;
  node [shape=box, fontname="{FONT}", fontsize=30, margin="0.18,0.1"];
  edge [fontname="{FONT}", fontsize=26, fontcolor="{SLATE}"];
  {{ rank=same; word [label="hidden word\n“otter”", {token}]; later [label="later tokens\n“. Nobody else knows.”", {token}];
    cue [label="cue\n“My pet is a”", {token}]; }}
  {{ rank=same; k2 [label="layer 2 keys\nat the hidden word\n(2 subcomponents)", color="{BLUE}", fontcolor="{BLUE}", {part}];
    q2 [label="layer 2 query\nat the later tokens", color="{BLUE}", fontcolor="{BLUE}", {part}]; }}
  h2 [label="layer 2, head 3\ncopies the word\ninto the later tokens", {head}];
  o2 [label="layer 2 attention output\nat the later tokens", color="{BLUE}", fontcolor="{BLUE}", {part}];
  {{ rank=same; k3 [label="layer 3 key\nat the later tokens", color="{BLUE}", fontcolor="{BLUE}", {part}];
    q3 [label="layer 3 query\nat the cue", color="{BLUE}", fontcolor="{BLUE}", {part}];
    b3 [label="layer 3 queries\nat the cue\n(2 subcomponents)", color="{CORAL}", fontcolor="{CORAL}", {part}]; }}
  h3 [label="layer 3, heads 4 and 5\nread the later tokens\nat the cue", {head}];
  o3 [label="layer 3 attention output\nat the cue", color="{BLUE}", fontcolor="{BLUE}", {part}];
  answer [label="answer:\n“otter”", style="rounded,filled,bold", fillcolor="{PALE_SLATE}", color="{PALE_SLATE}", fontcolor="{INK}", fontsize=33];
  word -> k2 [{info}]; later -> q2 [{info}]; cue -> q3 [{info}]; cue -> b3 [{info}];
{bundle("word", "h2", "78 value\\nsubcomponents")}
  k2 -> h2 [{need(E("h.2.attn.k_proj#224@word", "h.2.attn.k_proj#206@word"))}];
  q2 -> h2 [{need(E("h.2.attn.q_proj#436@later"))}];
  h2 -> o2 [{need(E("h.2.attn.o_proj#735@later"))}];
  o2 -> k3 [{info}];
{bundle("o2", "h3", "many value\\nsubcomponents")}
  k3 -> h3 [{need(E("h.3.attn.k_proj#145@later"))}];
  q3 -> h3 [{need(E("h.3.attn.q_proj#182@cue"))}];
  b3 -> h3 [{block(E("h.3.attn.q_proj#334@cue", "h.3.attn.q_proj#60@cue"))}];
  h3 -> o3 [{need(E("h.3.attn.o_proj#806@cue"))}];
  o3 -> answer [{need(2.0)}];
}}
"""


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
  {{ rank=same; h0 [label="layer 21, head 0\ncopies the hidden animal", style="rounded,filled,bold", fillcolor="{PALE_BLUE}", color="{BLUE}", fontcolor="{INK}", penwidth=2.6];
    h6 [label="layer 21, head 6\nwrites the hidden animal's opposite", style="rounded,filled,bold", fillcolor="#fbe3dc", color="{CORAL}", fontcolor="{INK}", penwidth=2.6]; }}
  answer [label="answer", style="rounded,filled,bold", fillcolor="{PALE_SLATE}", color="{PALE_SLATE}", fontcolor="{INK}", fontsize=21];
  reply -> h0 [color="{CLOUD}", style=dashed, penwidth=2.2, label=" reads"];
  reply -> h6 [color="{CLOUD}", style=dashed, penwidth=2.2, label=" reads"];
  question -> h6 [color="{CORAL}", penwidth=3, label=" makes it read\n 3 times more"];
  h0 -> answer [color="{BLUE}", penwidth=4, label=" toward the\n hidden animal"];
  h6 -> answer [color="{CORAL}", penwidth=5, arrowhead=tee, arrowsize=1.4, label=" away from the\n hidden animal"];
}}
"""


def graph_dot(c, min_edge=0.2):
    base = 100 * c["recall"]
    nodes = c["circuit"]
    lost = {n: 100 * c["single"].get(n, 0.0) for n in nodes}
    stage = {"q_proj": 0, "k_proj": 0, "v_proj": 0, "o_proj": 1, "c_fc": 2, "down_proj": 3}
    rank = lambda n: 4 * int(n.split(".")[1]) + stage[n.split("#")[0].split(".")[-1]]
    nid = {n: f"n{i}" for i, n in enumerate(nodes)}
    lines = [f'digraph G {{ rankdir=BT; splines=spline; nodesep=0.3; ranksep=0.42; bgcolor="white"; pad=0.2; newrank=true;',
             f'  node [shape=box, style="rounded,filled", fontname="{FONT}", fontsize=17, margin="0.14,0.06", penwidth=0];',
             f'  edge [arrowsize=0.9];']
    ranks = sorted({rank(n) for n in nodes})
    names = {0: "query / key / value", 1: "attention output", 2: "MLP in", 3: "MLP out"}
    for r in ranks:
        members = [n for n in nodes if rank(n) == r]
        row = f'  {{ rank=same; r{r} [label="layer {r // 4}\\n{names[r % 4]}", shape=plaintext, style="", fontcolor="{SLATE}", fontsize=16];'
        for n in sorted(members, key=lambda n: list(WHERE).index(n.split("@")[1])):
            site, rest = n.split("#")
            idx, cls = rest.split("@")
            d = lost[n]
            col, fill = (BLUE, PALE_BLUE) if d > 0 else ("#c9512f", "#fbe3dc")
            row += (f' {nid[n]} [label="at {WHERE[cls]}", fillcolor="{fill}", '
                    f'fontcolor="{col}", fontsize={15 + min(8, 1.2 * abs(d)):.0f}];')
        lines.append(row + " }")
    for a, b in zip(ranks, ranks[1:]):
        lines.append(f"  r{a} -> r{b} [style=invis];")
    for e, v in sorted(c["edges"].items(), key=lambda kv: abs(kv[1])):
        src, dst = e.split(" -> ")
        if abs(v) < min_edge or src not in nid or dst not in nid:
            continue
        col = BLUE if v < 0 else CORAL
        lines.append(f'  {nid[src]} -> {nid[dst]} [color="{col}", penwidth={0.8 + 3.2 * min(abs(v), 1.0):.2f}];')
    lines.append("}")
    return "\n".join(lines)
