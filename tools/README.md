# AI-SETT Assessment Tools

Diagnostic profiling tools for language models. Profiles, not scores. Gaps, not rankings.

## Requirements

```bash
pip install pyyaml httpx
```

Only two dependencies. No provider SDKs, no plotting libraries.

## Tools

### assessment_runner.py — Run probes against a model

Loads probe YAML files, sends them to a model provider, evaluates responses
against criteria, and builds a diagnostic profile.

```bash
# Run assessment against OpenAI
python -m tools.assessment_runner \
  --probes probes/understanding/ \
  --provider openai --model gpt-4o \
  --output results/gpt4o.json

# Run against Anthropic
python -m tools.assessment_runner \
  --probes probes/ \
  --provider anthropic --model claude-3-opus-20240229 \
  --output results/claude.json

# Use a local OpenAI-compatible endpoint (Ollama, vLLM, etc.)
python -m tools.assessment_runner \
  --probes probes/ \
  --provider openai --model llama3 \
  --base-url http://localhost:11434/v1 \
  --output results/local.json

# Dry run — validate probes without API calls
python -m tools.assessment_runner --dry-run --probes probes/

# List available providers
python -m tools.assessment_runner --list-providers
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--probes` | (required) | Path to probe YAML file or directory |
| `--provider` | — | Provider name (openai, anthropic, google, mistral, cohere) |
| `--model` | — | Model identifier |
| `--base-url` | — | Custom API base URL |
| `--output` | `assessment_results.json` | Output path |
| `--temperature` | `0.0` | Generation temperature |
| `--concurrency` | `2` | Max parallel API calls |
| `--aggregation` | `majority` | How to aggregate multi-probe criteria (`any`, `all`, `majority`) |
| `--dry-run` | — | Load probes only, no API calls |
| `--verbose` | — | Detailed output |

**Environment variables:**

Set the API key for your provider:
- `OPENAI_API_KEY` — OpenAI (and compatible endpoints)
- `ANTHROPIC_API_KEY` — Anthropic
- `GOOGLE_API_KEY` — Google Gemini
- `MISTRAL_API_KEY` — Mistral
- `COHERE_API_KEY` — Cohere

### profile_visualizer.py — Generate HTML reports

Creates self-contained HTML reports with SVG radar charts and heatmaps.
No JavaScript dependencies.

```bash
# Generate HTML profile report
python -m tools.profile_visualizer \
  --input results/gpt4o.json \
  --output results/profile.html

# Output to stdout
python -m tools.profile_visualizer --input results/gpt4o.json
```

### gap_analyzer.py — Analyze gap patterns

Identifies gap clusters, intervention priorities, ZPD candidates,
and supports longitudinal comparison between runs.

```bash
# Show all analysis
python -m tools.gap_analyzer --input results/gpt4o.json

# Show intervention priorities only
python -m tools.gap_analyzer --input results/gpt4o.json --priorities

# Show ZPD candidates (most productive intervention targets)
python -m tools.gap_analyzer --input results/gpt4o.json --zpd

# Show gap clusters
python -m tools.gap_analyzer --input results/gpt4o.json --clusters

# Compare two assessments (longitudinal tracking)
python -m tools.gap_analyzer \
  --input results/gpt4o-v2.json \
  --compare results/gpt4o-v1.json

# JSON output
python -m tools.gap_analyzer --input results/gpt4o.json --json
```

### probe_generator.py — Generate probe templates

Parses AI-SETT-FRAMEWORK.md to extract all criteria and generates
YAML probe templates for uncovered criteria.

```bash
# Check current probe coverage
python -m tools.probe_generator \
  --framework docs/AI-SETT-FRAMEWORK.md \
  --coverage probes/

# Generate templates for all uncovered criteria
python -m tools.probe_generator \
  --framework docs/AI-SETT-FRAMEWORK.md \
  --output probes/

# Both: report coverage and generate templates
python -m tools.probe_generator \
  --framework docs/AI-SETT-FRAMEWORK.md \
  --coverage probes/ \
  --output probes/

# JSON coverage report
python -m tools.probe_generator \
  --framework docs/AI-SETT-FRAMEWORK.md \
  --coverage probes/ --json
```

## Providers

| Provider | Endpoint | Auth Env Var | Notes |
|----------|----------|-------------|-------|
| openai | `/v1/chat/completions` | `OPENAI_API_KEY` | Use `--base-url` for Groq, vLLM, Ollama |
| anthropic | `/v1/messages` | `ANTHROPIC_API_KEY` | System msg extracted to top-level param |
| google | `generativelanguage.googleapis.com` | `GOOGLE_API_KEY` | Gemini API |
| mistral | `/v1/chat/completions` | `MISTRAL_API_KEY` | OpenAI-compatible format |
| cohere | `/v2/chat` | `COHERE_API_KEY` | Uses `message` + `chat_history` |

## Probe format

Probes are YAML files (multi-document supported with `---` separator):

```yaml
id: "probe_U_BR_001"
name: "Basic arithmetic"
criteria_tested:
  - "U.BR.01"
  - "U.BR.02"

input: "What's 2 + 2?"

expected_behaviors:
  - "Response contains '4'"
  - "Response is under 20 words"

evaluation:
  U.BR.01:
    check: "Contains answer to literal question"
    pass: "Response addresses 2+2"
  U.BR.02:
    check: "Answer is correct"
    pass: "Contains '4'"
```

### Multi-turn probes

```yaml
id: "probe_I_MT_001"
name: "Context retention"
type: "multi_turn"
criteria_tested:
  - "I.MT.01"

turns:
  - role: "user"
    content: "My name is Alex."
  - role: "user"
    content: "What's my name?"

evaluation:
  I.MT.01:
    check: "Remembers earlier context"
    pass: "Contains 'Alex'"
```

## Result format

Results follow `schemas/result_v1.json`. Key structure:

```
version: "1.0.0"
model: string
provider: string
timestamp: ISO 8601
metadata: {probe_count, criteria_tested, categories_tested, ...}
probe_results: [{probe_id, input, response, criteria_results, evidence, latency_ms}]
profile: {category: {demonstrated, total, subcategories: {subcat: {demonstrated, total, criteria: {id: bool}}}}}
```

## Design principles

- **Profiles, not scores** — the gap list is the primary output, not a count
- **httpx only** — no provider SDKs; each provider is just HTTP calls
- **SVG charts** — no matplotlib/plotly; radar + heatmap are simple geometry
- **Rule-based evaluation first** — criteria are observable/binary; LLM judge is optional fallback
- **ThreadPoolExecutor** — simpler than async, sufficient for I/O-bound API calls

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

Key rules:
- Tools output profiles, not rankings
- Never compute "total score" as primary metric
- Focus on gap identification
- Support iterative assessment (before/after intervention)
