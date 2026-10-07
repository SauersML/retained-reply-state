#!/bin/bash
# One model on one GPU: a confirmation run, localization on the discovery run, and the tests of the heads it selects
# on the confirmation run.  usage: run_gpu.sh MODEL TAG DISCOVERY_SEED CONFIRMATION_SEED OUT [DISCOVERY_JSON]
set -euo pipefail
M=$1; T=$2; S1=$3; S2=$4; O=$5; D=${6:-}
cd "$(dirname "$0")/.."
mkdir -p "$O"
B=${BATCH:-48}
if [ -z "$D" ]; then
    D=$O/${T}_discovery.json
    [ -s "$D" ] || python3 hidden_choice.py --model "$M" --n 600 --batch 48 --device cuda --seed "$S1" --wording "${WORDING:-A}" --out "$D"
fi
C=$O/${T}_confirmation.json
[ -n "${CONF:-}" ] && [ ! -s "$C" ] && cp "$CONF" "$C"
[ -s "$C" ] || python3 hidden_choice.py --model "$M" --n 600 --batch 48 --device cuda --seed "$S2" --wording "${WORDING:-A}" --out "$C"
python3 stats.py "$D" "$C" $([ -n "${NOCAL:-}" ] || echo --calibrate) > "$O/${T}_stats.txt"
python3 copying.py "$M" --out "$O/${T}_copying.json"
# localization on the discovery run: layer bands and reply tokens, then every key/value head in the bands that matter
# (|animal-level z| > 3, at most 12 layers)
[ -s "$O/${T}_localize_bands.json" ] || python3 localize.py "$D" --band 4 --only layers,token --batch "$B" --device cuda \
    --out "$O/${T}_localize_bands.json"
HL=$(python3 -c "import json; a = json.load(open('$O/${T}_localize_bands.json'))['arms']; b = sorted(((abs(v['z']), k) for k, v in a.items() if k.startswith('layers ')), reverse=True); keep = [k for z, k in b if z > 3] or [k for z, k in b[:2]]; ls = [l for k in keep for l in range(int(k.split()[1].split('-')[0]), int(k.split()[1].split('-')[1]) + 1)][:12]; print(','.join(map(str, sorted(ls))))")
echo "head layers: $HL"
[ -s "$O/${T}_localize_discovery.json.done" ] || { python3 localize.py "$D" --band 4 --only head --head-layers "$HL" \
    --batch "$B" --device cuda --out "$O/${T}_localize_discovery.json"; touch "$O/${T}_localize_discovery.json.done"; }
G=$(python3 select_heads.py "$O/${T}_localize_discovery.json"); echo "$G" > "$O/${T}_heads.txt"
P=${G%;*}; S=${G#*;}
HG="all"; [ -n "$P" ] && HG="$P;$HG"; [ -n "$S" ] && HG="$S;$HG"; [ -n "$P" ] && [ -n "$S" ] && HG="$P,$S;$HG"
[ -s "$O/${T}_localize_confirmation.json" ] || python3 localize.py "$C" --band 4 --batch "$B" --device cuda --only layers,token --out "$O/${T}_localize_confirmation.json"
python3 dose.py "$C" --groups "$HG" --doses=-2,-1,0,0.5,1,2,4,8 --batch "$B" --device cuda --out "$O/${T}_dose.json"
NEC="all"; [ -n "$P" ] && NEC="$P"; [ -n "$S" ] && NEC="$NEC;$S"
python3 dose.py "$C" --groups "$NEC" --base retained --doses=0,1 --batch "$B" --device cuda --out "$O/${T}_necessity.json"
# weight edits of the selected heads (their output projections) on the confirmation run, with web-text damage
if [ -n "${WEB:-}" ]; then
    EDITS=$(python3 -c "import sys; z = lambda g: ','.join(h + '*0' for h in g.split(',') if h); print(';'.join(e for e in (z(sys.argv[1]), z(sys.argv[2])) if e))" "$S" "$P")
    [ -n "$EDITS" ] && python3 edit_heads.py "$C" "$WEB" --arms retained,neutral --edits "$EDITS" --batch "$B" --device cuda \
        --out "$O/${T}_edit_heads.json"
fi
[ -n "${SKIP_SLOW:-}" ] && { echo "pipeline done"; exit 0; }
H=$(echo "$P,$S" | sed 's/^,//; s/,$//')
[ -n "$H" ] && python3 weights.py "$C" --heads "$H" --batch "$B" --device cuda --out "$O/${T}_weights.json"
if [ -n "$P" ]; then
    RL=${P%%:*}
    python3 writers.py "$C" --read-layers "$RL-$RL" --batch "$B" --device cuda --out "$O/${T}_writers_layers.json"
    WL=$(python3 -c "import json; a=json.load(open('$O/${T}_writers_layers.json'))['arms']; b=a[[k for k in a if k.startswith('no ablation')][0]]['raise']; r=sorted(((b - v['raise'], k.split()[-1]) for k, v in a.items() if k.startswith('writer layer')), reverse=True); print(','.join(x[1] for x in r[:2]))")
    python3 writers.py "$C" --read-layers "$RL-$RL" --head-layers "$WL" --no-layer-stage --batch "$B" --device cuda --out "$O/${T}_writers_heads.json"
fi
echo "pipeline done"
