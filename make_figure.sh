#!/bin/bash
# The main figure (figs/main.png) and the Goodfire 4-layer figure (figs/goodfire.png) from the saved results: Qwen3
# numbers from the float32 rescoring (fp32_runs.py, results/fp32), Goodfire's 4-layer model from results/fourlayer.
cd "$(dirname "$0")"
F=results/fp32
python figure_main.py \
  --results results/qwen3_0.6b.json results/qwen3_1.7b.json+results/q17A/q17A_confirmation.json \
            results/qwen3_4b.json+results/q4A/q4A_more.json+results/q4A/q4A_seed4102.json+results/q4A/q4A_seed4101.json \
            results/qwen3_8b.json+results/q8A/q8A_confirmation.json \
  --retained $F/q06A.npz $F/q17A_discovery.npz+$F/q17A_heldout.npz \
             $F/q4A_set1.npz+$F/q4A_set2.npz+$F/q4A_seed4102.npz+$F/q4A_seed4101.npz $F/q8A_set1.npz+$F/q8A_set2.npz \
  --text4l results/fourlayer/hidden_span.json --flip figs/flip_rows.json \
  --questions $F/q17A_heldout.npz+$F/q17A_discovery.npz --questions06 $F/q06A.npz \
  --parts results/fourlayer/attention_gates.json --refit results/fourlayer/sparse_refit_fine.json \
  --out figs/main.png
python figure_goodfire.py --text4l results/fourlayer/hidden_span.json --refit results/fourlayer/sparse_refit_fine.json \
  --parts results/fourlayer/attention_gates.json --out figs/goodfire.png
