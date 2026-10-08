"""Every Qwen3 number in the figures, rescored in float32 by edit_heads.py (one GPU per group, run in sequence).

Group "1.7b": the held-out runs (q17A_confirmation) and the runs head 21.6 was found on (qwen3_1.7b), each with the
original model and layer 21 key-value head 6 switched off, asking every turn-2 question (recall A, recall B, any animal,
and each after Janus's LLM explainer and after a CPU explainer of the same length and style); the control heads on the
held-out runs; countries, turn 1 without "Do not reveal it.", and both turns in wording B, with head 21.6 switched off.
Group "other": Qwen3-0.6B (same questions, head 21.6 switched off), Qwen3-4B and Qwen3-8B (two sets each, unedited).
DOCS holds explainer.txt (Janus's LLM explainer) and cpu_explainer.txt (documents/cpu_explainer.txt).
usage: fp32_runs.py GROUP WINDOWS.u32 DOCS --device cuda
"""
import argparse
import os
import subprocess
import sys

from hidden_choice import task

CONTROLS = "21:5*0;21:1*0,21:2*0;21:3*0,21:4*0;20:5*0,20:6*0;22:5*0,22:6*0;26:4*0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group", choices=["1.7b", "other"])
    ap.add_argument("windows")
    ap.add_argument("docs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    A, B, N = task("A")["recall"], task("B")["recall"], task("A")["neutral"]
    doc = open(os.path.join(a.docs, "explainer.txt")).read().strip()
    cpu = open(os.path.join(a.docs, "cpu_explainer.txt")).read().strip()
    questions = {"recall_B": B, "doc_A": doc + "\n\n" + A, "doc_B": doc + "\n\n" + B, "doc_neutral": doc + "\n\n" + N,
                 "cpu_A": cpu + "\n\n" + A, "cpu_B": cpu + "\n\n" + B, "cpu_neutral": cpu + "\n\n" + N}
    qargs = [x for k, v in questions.items() for x in ("--question", f"{k}={v}")]
    jobs = {
        "1.7b": [
            ("results/q17A/q17A_confirmation.json", ["--arms", "retained,neutral", "--edits", "21:6*0", "--save-logp"] + qargs, "q17A_heldout"),
            ("results/qwen3_1.7b.json", ["--arms", "retained,neutral", "--edits", "21:6*0", "--save-logp"] + qargs, "q17A_discovery"),
            ("results/q17A/q17A_confirmation.json", ["--arms", "retained", "--edits", CONTROLS], "q17A_controls"),
            ("results/q17A/q17A_countries.json", ["--arms", "retained,neutral", "--edits", "21:6*0"], "q17A_countries"),
            ("results/q17A/q17A_confirmation.json", ["--arms", "retained,neutral", "--edits", "21:6*0",
                                                     "--turn1-drop", " Do not reveal it."], "q17A_noreveal"),
            ("results/q17B/q17b_div_B.json", ["--arms", "retained,neutral", "--edits", "21:6*0"], "q17B_wordingB"),
        ],
        "other": [
            ("results/qwen3_0.6b.json", ["--arms", "retained,neutral", "--edits", "21:6*0", "--save-logp"] + qargs, "q06A"),
            ("results/qwen3_4b.json", ["--arms", "retained", "--save-logp"], "q4A_set1"),
            ("results/q4A/q4A_more.json", ["--arms", "retained", "--save-logp"], "q4A_set2"),
            ("results/qwen3_8b.json", ["--arms", "retained", "--save-logp"], "q8A_set1"),
            ("results/q8A/q8A_confirmation.json", ["--arms", "retained", "--save-logp"], "q8A_set2"),
        ],
    }[a.group]
    os.makedirs("results/fp32", exist_ok=True)
    for result, extra, name in jobs:
        out = f"results/fp32/{name}.json"
        if os.path.exists(out.replace(".json", ".done")):
            continue
        cmd = [sys.executable, "edit_heads.py", result, a.windows, "--device", a.device, "--batch", str(a.batch),
               "--dtype", "float32", "--out", out] + extra
        print("running", name, flush=True)
        if subprocess.call(cmd) != 0:
            sys.exit(f"{name} failed")
        open(out.replace(".json", ".done"), "w").close()


if __name__ == "__main__":
    main()
