"""Circuit diagrams of Goodfire's 4-layer model laid out by Graphviz (dot), for figure_main.py.

gates_dot: the recall circuit as parts and wires: the two attention steps (head 2.3 copies the word into the later
  tokens; heads 3.4 and 3.5 attend to them where the word is recalled) and the VPD subcomponents measured to switch them on or off
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
    """dot -> PNG; returns the image array.  snap: (label node, tail, head) -- after layout, each label node is placed
    with its bottom just above the highest wire from tail to head (the gate wires run below the bundles)."""
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
        near = [py for px, py in pts if x - w / 2 <= px <= x + w / 2] or [py for _, py in pts]
        moves.append(f'  {lab} [pos="{x:.1f},{max(near) + 4 + h / 2:.1f}"];')
    with open(path + ".dot", "w") as f:
        f.write(laid.rstrip().rstrip("}") + "\n" + "\n".join(moves) + "\n}\n")
    subprocess.run([NEATO, "-n2", "-Tpng", f"-Gdpi={dpi}", "-o", path + ".png", path + ".dot"], check=True)
    return mpimg.imread(path + ".png")


def width(d):
    """Wire width from the change in the hidden word's raise (percent of the unedited raise) when the part is removed."""
    return f"{1.6 + min(6.0, 0.09 * abs(d)):.2f}"


DRAWN = ["h.2.attn.k_proj#224@word", "h.2.attn.k_proj#206@word", "h.2.attn.q_proj#436@later", "h.3.attn.k_proj#145@later",
         "h.3.attn.k_proj#507@later", "h.3.attn.q_proj#182@cue", "h.3.attn.q_proj#334@cue", "h.3.attn.q_proj#60@cue"]


def gates_dot(g, routes):
    """Parts named by layer, matrix and position: every query or key subcomponent whose removal moves the hidden word's
    raise (its log P relative to its mean within the template) by at least 10% (fourlayer/attention_gates.py); a part
    of two subcomponents takes the larger effect.  Wire width grows with that change.  The stages that carry the word
    (fourlayer/mlp_routes.py, routes): each labeled with the change in the raise when its word-specific output is
    removed (set to its mean over the template's words), alone; the stages are in series, so these do not add.
    Returns the dot source and the bundles' (label node, tail, head) for render(snap=...)."""
    eff = {k: 100 * v["raise_change_share"] for k, v in g["gates"].items()}
    missing = [k for k in DRAWN if k not in eff]
    assert not missing, f"no measurement for {missing}"
    E = lambda *keys: max((eff[k] for k in keys), key=abs)
    need = lambda d: f'color="{BLUE}", penwidth={width(d)}, arrowhead=normal, arrowsize=1.1'
    block = lambda d: f'color="{CORAL}", penwidth={width(d)}, arrowhead=tee, arrowsize=1.4'
    info = f'color="{CLOUD}", penwidth=2.2, style=dashed, arrowhead=normal, arrowsize=0.9'
    token = f'style="rounded,filled", fillcolor="{PALE_SLATE}", color="{PALE_SLATE}", fontcolor="{INK}"'
    part = f'style="rounded,filled", fillcolor="white", penwidth=2.2'
    head = f'style="rounded,filled,bold", fillcolor="{PALE_BLUE}", color="{BLUE}", fontcolor="{INK}", fontsize=40, penwidth=2.6'

    share = lambda key: f"\u2212{-100 * routes['blocks'][key]['raise']:.0f}%"
    combo = lambda key: f"\u2212{-100 * routes['combos'][key]['raise']:.0f}%"
    biggest = max(-v["raise"] for v in routes["parts"].values())
    stage = f'style="rounded,filled", fillcolor="white", color="{SLATE}", fontcolor="{INK}", penwidth=2.2'
    snap = []

    def bundle(a, b, label, k=10):
        """Many thin wires: information carried by many subcomponents, none of them needed alone.  The label is a node
        between the two ends, so dot sets it beside the wires and keeps other nodes clear of it."""
        snap.append((f"{a}_{b}", a, b))
        wires = [f'  {a} -> {b} [color="{SLATE}", penwidth=0.9, arrowsize=0.55];' for _ in range(k)]
        wires.append(f'  {a}_{b} [shape=plaintext, label="{label}", fontcolor="{SLATE}", fontsize=31, margin=0];')
        wires.append(f'  {a} -> {a}_{b} -> {b} [style=invis, weight=3];')
        return "\n".join(wires)
    return f"""digraph G {{
  rankdir=LR; splines=spline; nodesep=0.48; ranksep=0.45; bgcolor="white"; pad=0.1;
  node [shape=box, fontname="{FONT}", fontsize=37, margin="0.18,0.1"];
  edge [fontname="{FONT}", fontsize=31, fontcolor="{SLATE}"];
  {{ rank=same; word [label="hidden word\n“fox”", {token}]; later [label="later tokens\n“. Nobody else knows.”", {token}];
    cue [label="where the word is recalled\n“My pet is a”", {token}]; }}
  {{ rank=same; k2 [label="layer 2 keys\nat the hidden word\n(2 subcomponents)", color="{BLUE}", fontcolor="{BLUE}", {part}];
    q2 [label="layer 2 query\nat the later tokens", color="{BLUE}", fontcolor="{BLUE}", {part}]; }}
  h2 [label="layer 2, head 3\ncopies the word\ninto the later tokens", {head}];
  {{ rank=same; k3 [label="layer 3 keys\nat the later tokens\n(2 subcomponents)", color="{BLUE}", fontcolor="{BLUE}", {part}];
    q3 [label="layer 3 query\nwhere the word\nis recalled", color="{BLUE}", fontcolor="{BLUE}", {part}];
    b3 [label="layer 3 queries\nwhere the word\nis recalled\n(2 subcomponents)", color="{CORAL}", fontcolor="{CORAL}", {part}]; }}
  h3 [label="layer 3, heads 4 and 5\nwhere the word is recalled\nattend to the later tokens", {head}];
  m0 [label="layer 0 MLP at the hidden word\nwrites the word's identity\n(recall {share('0.mlp@word')} without it; spread over\nmany subcomponents, none over {100 * biggest:.0f}% alone)", {stage}];
  early [label="layers 0-1 at the later tokens:\nan earlier copy of the word\n(recall {combo('0.attn@later,0.mlp@later,1.attn@later')} without it)", {stage}];
  read2 [label="layer 2 attention\nwhere the word is recalled:\nbrings in the earlier copy\n(recall {share('2.attn@cue')} without it)", {stage}];
  m3 [label="layer 3 MLP\nwhere the word is recalled\nturns it into the answer\n(recall {share('3.mlp@cue')} without it)", {stage}];
  answer [label="answer:\n“fox”", style="rounded,filled,bold", fillcolor="{PALE_SLATE}", color="{PALE_SLATE}", fontcolor="{INK}", fontsize=40];
  word -> k2 [{info}]; word -> m0 [{info}]; later -> q2 [{info}]; later -> k3 [{info}]; cue -> q3 [{info}]; cue -> b3 [{info}];
  word -> early [{info}];
{bundle("m0", "h2", "78 value\\nsubcomponents")}
  k2 -> h2 [{need(E("h.2.attn.k_proj#224@word", "h.2.attn.k_proj#206@word"))}];
  q2 -> h2 [{need(E("h.2.attn.q_proj#436@later"))}];
{bundle("h2", "h3", "the copied word:\\nmany output and\\nvalue subcomponents")}
  k3 -> h3 [{need(E("h.3.attn.k_proj#145@later", "h.3.attn.k_proj#507@later"))}];
  q3 -> h3 [{need(E("h.3.attn.q_proj#182@cue"))}];
  b3 -> h3 [{block(E("h.3.attn.q_proj#334@cue", "h.3.attn.q_proj#60@cue"))}];
  early -> h3 [color="{SLATE}", penwidth=1.6, arrowsize=0.8];
  early -> read2 [color="{SLATE}", penwidth=1.6, arrowsize=0.8];
  read2 -> m3 [color="{SLATE}", penwidth=1.6, arrowsize=0.8];
{bundle("h3", "m3", "many output\\nsubcomponents")}
  m3 -> answer [color="{SLATE}", penwidth=2.2, arrowsize=0.9];
}}
""", snap


def qwen_dot():
    """Qwen3-1.7B, from head removals and weight edits (edit_heads.py, attention.py): two layer-21 heads read the reply's
    "." token, where the hidden animal is held, and push the answer in opposite directions; the question's wording sets
    how much the suppressing head reads."""
    token = f'style="rounded,filled", fillcolor="{PALE_SLATE}", color="{PALE_SLATE}", fontcolor="{INK}"'
    return f"""digraph G {{
  rankdir=BT; splines=spline; nodesep=0.55; ranksep=1.9; bgcolor="white"; pad=0.2;
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
