#!/bin/bash
# Post-harvest pipeline: integrate new data → re-tag → run assessment
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_DIR="$SCRIPT_DIR/training_data"
HARVEST_DIR="$TRAIN_DIR/harvested"

echo "=============================================="
echo "Step 1: Combine harvested examples"
echo "=============================================="
python3 -c "
import json, os, glob

base = '$HARVEST_DIR'
outfile = '$TRAIN_DIR/harvested_combined.jsonl'

count = 0
valid = 0
with open(outfile, 'w') as out:
    for fpath in sorted(glob.glob(os.path.join(base, '**/*.jsonl'), recursive=True)):
        if '_combined' in fpath:
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                count += 1
                try:
                    obj = json.loads(line)
                    msgs = obj.get('messages', [])
                    if len(msgs) >= 2:
                        out.write(json.dumps({'messages': msgs}, ensure_ascii=False) + '\n')
                        valid += 1
                except:
                    pass

print(f'Harvested: {valid} valid examples from {count} records')
"

echo ""
echo "=============================================="
echo "Step 2: Re-tag all training data"
echo "=============================================="
cd "$TRAIN_DIR"
python3 tag_and_sample.py

echo ""
echo "=============================================="
echo "Step 3: Run full AI-SETT assessment"
echo "=============================================="
cd "$SCRIPT_DIR"
bash run_aisett_eval.sh
