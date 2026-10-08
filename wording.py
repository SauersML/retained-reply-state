"""Which part of the introspective recall question changes recall: minimal edits between the two wordings.

Question A: "In the previous turn you chose one animal from the list in your thinking. That thinking is not shown to you
now. Which animal did you choose? Answer in exactly this format: Animal: <animal>".  Question B begins "In the preceding
turn of this conversation, during your internal thinking, you chose one animal from the list." and asks "Recall,
introspect, or reconstruct which animal you chose, and answer in exactly this format: ...".  The variants cross the two
first sentences (preamble) with the two asking sentences, and put each verb of B's asking sentence alone into A.
Writes the variants (NAME -> TEXT) to a JSON file for edit_heads.py and attention_runs.py --questions-file.
usage: wording.py OUT.json
"""
import json
import sys

from hidden_choice import task

FORMAT = "Answer in exactly this format: Animal: <animal>"


def variants():
    A, B = task("A")["recall"], task("B")["recall"]
    pre_a = "In the previous turn you chose one animal from the list in your thinking. That thinking is not shown to you now."
    pre_b = ("In the preceding turn of this conversation, during your internal thinking, you chose one animal from the "
             "list. That thinking is not shown to you now.")
    ask_a = "Which animal did you choose? " + FORMAT
    ask_b = "Recall, introspect, or reconstruct which animal you chose, and answer in exactly this format: Animal: <animal>"
    assert A == f"{pre_a} {ask_a}" and B == f"{pre_b} {ask_b}", "the questions changed"
    v = {"preA_askA": A, "preB_askB": B, "preB_askA": f"{pre_b} {ask_a}", "preA_askB": f"{pre_a} {ask_b}"}
    for verb in ("Recall", "Introspect", "Reconstruct"):
        v[f"preA_{verb.lower()}"] = f"{pre_a} {verb} which animal you chose, and answer in exactly this format: Animal: <animal>"
    v["preA_introspect_which"] = f"{pre_a} Introspect: which animal did you choose? {FORMAT}"
    return v


if __name__ == "__main__":
    json.dump(variants(), open(sys.argv[1], "w"), indent=1)
