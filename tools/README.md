# Tools

Assessment automation tools for AI-SETT.

## Status

🚧 Under development

## Planned tools

### assessment_runner.py
- Load probes from YAML
- Send to model via API
- Apply evaluation criteria
- Generate profile report

### profile_visualizer.py
- Visualize assessment results
- Show demonstrated vs gap by category
- Radar charts for capability profile

### gap_analyzer.py
- Identify patterns in gaps
- Suggest intervention priorities
- Track changes across assessments

## Usage (planned)

```bash
# Run assessment
python tools/assessment_runner.py \
  --model "your-model-endpoint" \
  --probes "probes/understanding/*.yaml" \
  --output "results/assessment.json"

# Generate profile
python tools/profile_visualizer.py \
  --input "results/assessment.json" \
  --output "results/profile.html"

# Analyze gaps
python tools/gap_analyzer.py \
  --input "results/assessment.json" \
  --priorities
```

## Contributing

See CONTRIBUTING.md for guidelines on tool development.

Key principles:
- Tools should output profiles, not rankings
- Never compute "total score" as primary metric
- Focus on gap identification
- Support iterative assessment (before/after intervention)
