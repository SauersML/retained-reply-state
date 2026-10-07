#!/bin/bash
# Mac runs, one model at a time, each under a memory lease
cd ~/retained-reply-state
while pgrep -f "forced.py --model" > /dev/null; do sleep 20; done
PY=~/introspection/mlxenv/bin/python
~/.local/bin/mem-lease 16 $PY hidden_choice.py --model Qwen/Qwen3-1.7B --n 600 --batch 24 --device mps --seed 1 --out results/qwen3_1.7b.json > results/qwen3_1.7b.log 2>&1
~/.local/bin/mem-lease 8 $PY hidden_choice.py --model Qwen/Qwen3-0.6B --n 600 --batch 32 --device mps --seed 2 --out results/qwen3_0.6b.json > results/qwen3_0.6b.log 2>&1
~/.local/bin/mem-lease 24 $PY hidden_choice.py --model Qwen/Qwen3-4B --n 600 --batch 24 --device mps --seed 3 --out results/qwen3_4b.json > results/qwen3_4b.log 2>&1
echo done
