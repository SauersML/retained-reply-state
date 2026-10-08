"""The Goodfire 4-layer figure (make_figure.sh), Goodfire's 4-layer model only, one panel per row.
  a  the test's three conditions in plain text (main figure: j)
  b  in each condition, the raise of the hidden word's log P at recall within its template (nats), against the same with
     the hidden words re-paired at random (main figure: b's last column)
  c  the fewest VPD subcomponents that make the hidden word the top answer (main figure: i)
  d  Claude's circuit model of recall: two attention steps and the VPD subcomponents measured to switch them (main
     figure: k)
usage: make_figure.sh
"""
import argparse

import matplotlib.pyplot as plt

from figure_main import BLUE, CLOUD, SLATE, TEXT_CONDITIONS, Canvas, row_circuit, row_fourlayer, strip_raise, text_schematic

W = 20.0                                   # figure width in drawing units (inches)
ROWS = [10.6, 9.3, 9.6, 10.0]              # heights of the four rows
H = sum(ROWS)


def row_test(cv, y0):
    """Panel a, occupying [y0, y0 + ROWS[0]], its key below the sentences."""
    top = y0 + ROWS[0]
    text_schematic(cv, 0.6, top, letter="a", title="Goodfire 4-layer model: which tokens can attend to the hidden word",
                   key_at=(4.3, top - 9.75, 6.4, 0))


def row_raise(cv, text4l, y0):
    """Panel b, occupying [y0, y0 + ROWS[1]]: one plot per condition of panel a, labeled as there."""
    top = y0 + ROWS[1]
    cv.letter(0.1, top - 0.75, "b")
    cv.S.text(0.9, top - 0.75, "how much the hidden word's log P rises at recall", fontsize=25, weight="bold", va="center")
    cv.S.scatter([1.1], [top - 1.6], s=130, color=BLUE, edgecolor="white")
    cv.S.text(1.4, top - 1.6, "this word was hidden", fontsize=22, color=BLUE, va="center")
    cv.S.add_patch(plt.Rectangle((5.1, top - 1.8), 0.4, 0.4, facecolor=CLOUD, edgecolor="none"))
    cv.S.text(5.7, top - 1.6, "another word was hidden", fontsize=22, color=SLATE, va="center")
    for k, (name, col, label) in enumerate(TEXT_CONDITIONS):
        x = 2.8 + 5.6 * k
        ax = cv.axes(x, y0 + 1.9, 5.0, ROWS[1] - 4.3)
        strip_raise(ax, text4l, name, (-3, 16), [0, 0.1, 1, 10], name == "retained")
        ax.tick_params(labelsize=22)
        cv.S.text(x + 2.5, y0 + 1.0, label, ha="center", va="center", fontsize=22, color=col, weight="bold",
                  linespacing=1.05)
        if k == 0:
            ax.set_ylabel("hidden word's log P\nraised (nats)", fontsize=22, labelpad=8)
            ax.text(0.5, 0.78, "identical in every run,\nso exactly 0", transform=ax.transAxes, ha="center", va="center",
                    fontsize=20, color=SLATE)
        else:
            ax.set_yticklabels([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text4l", nargs="+", required=True, help="fourlayer/hidden_span.py outputs")
    ap.add_argument("--refit", nargs="+", required=True, help="fourlayer/optimize_edit.py --support outputs")
    ap.add_argument("--parts", required=True, help="fourlayer/attention_gates.py output")
    ap.add_argument("--out", default="figs/goodfire.png")
    a = ap.parse_args()
    fig = plt.figure(figsize=(W, H))
    cv = Canvas(fig, W, H)
    row_test(cv, sum(ROWS[1:]))
    row_raise(cv, a.text4l, sum(ROWS[2:]))
    row_fourlayer(cv, a.refit, ROWS[3], rh=ROWS[2], letter="c", pw=13.6,
                  title="rescaling {k} VPD subcomponents makes the hidden word the top answer {top:.0f}% of the time "
                        "(from {base:.0f}%)")
    row_circuit(cv, a.parts, 0, rh=ROWS[3], letter="d",
                title="Claude's circuit model")
    fig.savefig(a.out, dpi=140)                    # 2,800 pixels wide, as main.png


if __name__ == "__main__":
    main()
