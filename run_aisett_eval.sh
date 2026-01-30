#!/bin/bash
# Full AI-SETT assessment for nemotron-30b-aisett
# Run after harvester completes and model is free
#
# Baseline: nemotron-30b-calibrated = 20.1% (116/577 criteria)
# Target:   nemotron-30b-aisett     = ???

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT="$RESULTS_DIR/nemotron-30b-aisett-${TIMESTAMP}.json"

mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "AI-SETT Full Assessment: nemotron-30b-aisett"
echo "=============================================="
echo "Probes:   $SCRIPT_DIR/probes"
echo "Output:   $OUTPUT"
echo "Model:    nemotron-30b-aisett"
echo "Provider: openai (Ollama @ localhost:11434)"
echo ""

export OPENAI_API_KEY=not-needed

cd "$SCRIPT_DIR"

python3 -m tools.assessment_runner \
  --probes probes \
  --provider openai \
  --model nemotron-30b-aisett \
  --base-url http://localhost:11434/v1 \
  --temperature 0.0 \
  --concurrency 2 \
  --output "$OUTPUT" \
  --verbose

echo ""
echo "=============================================="
echo "Assessment complete: $OUTPUT"
echo "=============================================="

# Quick summary
python3 -c "
import json
with open('$OUTPUT') as f:
    data = json.load(f)

summary = data.get('summary', {})
total = summary.get('total_criteria', 0)
demonstrated = summary.get('demonstrated', 0)
pct = (demonstrated / total * 100) if total else 0

print(f'Total criteria:  {total}')
print(f'Demonstrated:    {demonstrated} ({pct:.1f}%)')
print()
print('Per category:')
for cat, info in sorted(data.get('categories', {}).items()):
    cat_total = info.get('total', 0)
    cat_demo = info.get('demonstrated', 0)
    cat_pct = (cat_demo / cat_total * 100) if cat_total else 0
    print(f'  {cat:<25} {cat_demo:>3}/{cat_total:<3} ({cat_pct:>5.1f}%)')

print()
print('BASELINE COMPARISON:')
print('  nemotron-30b-calibrated: 20.1% (116/577)')
print(f'  nemotron-30b-aisett:    {pct:.1f}% ({demonstrated}/{total})')
diff = pct - 20.1
print(f'  Delta:                  {diff:+.1f}%')
" 2>/dev/null || echo "Could not parse results"
