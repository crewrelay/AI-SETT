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
| `--simulate` | — | Enable simulated conversations for `type: "simulated"` probes |
| `--user-provider` | — | Provider for simulated user (required with `--simulate`) |
| `--user-model` | — | Model for simulated user (required with `--simulate`) |
| `--user-base-url` | — | Custom base URL for simulated user |
| `--user-temperature` | `0.7` | Temperature for simulated user |

**Simulation example:**

```bash
# Run all probes including simulated conversations
python -m tools.assessment_runner \
  --probes probes/ \
  --provider anthropic --model claude-sonnet-4-20250514 \
  --simulate \
  --user-provider anthropic --user-model claude-3-5-haiku-20241022 \
  --output results/sonnet_simulated.json
```

**Environment variables:**

Set the API key for your provider:
- `OPENAI_API_KEY` — OpenAI (and compatible endpoints)
- `ANTHROPIC_API_KEY` — Anthropic
- `GOOGLE_API_KEY` — Google Gemini
- `MISTRAL_API_KEY` — Mistral
- `COHERE_API_KEY` — Cohere

### multi_judge.py — Re-score with multiple LLM judges

Re-scores an assessment result using one or more LLM judges. When multiple
judges are used, computes Cohen's kappa for inter-rater agreement.

```bash
# Single judge (re-score with Sonnet)
python -m tools.multi_judge \
  --input results/gpt4o.json \
  --judge anthropic:claude-sonnet-4-20250514 \
  --output results/gpt4o_judged.json

# Multiple judges — get agreement analysis
python -m tools.multi_judge \
  --input results/gpt4o.json \
  --judge anthropic:claude-sonnet-4-20250514 \
  --judge openai:gpt-4o \
  --output results/gpt4o_multi_judged.json

# Only re-score heuristic criteria (skip deterministic ones)
python -m tools.multi_judge \
  --input results/gpt4o.json \
  --judge openai:gpt-4o-mini \
  --heuristic-only

# View agreement report from previously judged file
python -m tools.multi_judge \
  --input results/gpt4o_judged.json --report
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | Path to assessment result JSON |
| `--judge` | (repeatable) | Judge spec as `provider:model` |
| `--probes` | auto-detect | Path to probe YAML files (for evaluation specs) |
| `--output` | overwrite input | Output path |
| `--heuristic-only` | — | Only re-score criteria needing LLM judgment |
| `--concurrency` | `2` | Max parallel API calls |
| `--verbose` | — | Print each verdict |
| `--report` | — | Print agreement report (no API calls) |

**Agreement output:**
- Agreed/disagreed counts
- Cohen's kappa (0 = chance, 1 = perfect agreement)
- Interpretation (slight/fair/moderate/substantial/almost perfect)
- List of specific disagreements per criterion

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

### training_data_generator.py — Generate fine-tuning data from gaps

Generates fine-tuning training data to close assessment gaps. Multiple LLMs
contribute training examples for diversity, with cross-model consensus scoring
for quality signals without a separate judge.

```bash
# Generate from assessment gaps (multi-model)
python -m tools.training_data_generator \
  --input results/gpt4o.json \
  --framework docs/AI-SETT-FRAMEWORK.md \
  --generator anthropic:claude-3-5-haiku-20241022 \
  --generator openai:gpt-4o-mini \
  --format openai_jsonl \
  --examples-per-criterion 5 \
  --output training_data/gaps.jsonl

# Manual criterion targeting
python -m tools.training_data_generator \
  --criteria U.BR.01,U.BR.02,C.RL.01 \
  --framework docs/AI-SETT-FRAMEWORK.md \
  --generator anthropic:claude-3-5-haiku-20241022 \
  --format raw_jsonl

# ZPD-focused (highest-ROI gaps only)
python -m tools.training_data_generator \
  --input results/gpt4o.json --zpd-only \
  --framework docs/AI-SETT-FRAMEWORK.md \
  --generator openai:gpt-4o-mini

# Dry run — show plan, no API calls
python -m tools.training_data_generator \
  --input results/gpt4o.json \
  --framework docs/AI-SETT-FRAMEWORK.md \
  --generator openai:gpt-4o-mini --dry-run
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | — | Assessment result JSON (gap source) |
| `--criteria` | — | Comma-separated criterion IDs (alternative to --input) |
| `--framework` | (required) | Path to AI-SETT-FRAMEWORK.md |
| `--generator` | (required, repeatable) | `provider:model` or `provider:model:base_url` |
| `--format` | `raw_jsonl` | `openai_jsonl`, `anthropic_jsonl`, `huggingface_jsonl`, `raw_jsonl` |
| `--output` | `training_data.jsonl` | Output file path |
| `--examples-per-criterion` | `3` | Examples each generator produces per criterion |
| `--concurrency` | `2` | Parallel API calls |
| `--temperature` | `0.7` | Generation temperature |
| `--dedup-threshold` | `0.7` | Jaccard similarity threshold for dedup |
| `--zpd-only` | — | Only target ZPD candidates (gap ratio 0.1-0.6) |
| `--validate` | — | Run rule-based validation on outputs |
| `--dry-run` | — | Show plan, no API calls |
| `--verbose` | — | Detailed output |

**Generator spec format:**

- `anthropic:claude-3-5-haiku-20241022` — Anthropic API
- `openai:gpt-4o-mini` — OpenAI API
- `openai:llama3:http://localhost:11434/v1` — Ollama local
- `openai:mixtral:http://localhost:8000/v1` — vLLM local
- `openai:llama3:https://api.groq.com/openai/v1` — Groq

**Output formats:**

- `openai_jsonl` — `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`
- `anthropic_jsonl` — `{"system": "...", "messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`
- `huggingface_jsonl` — `{"instruction": "...", "input": "", "output": "...", "system": "...", "metadata": {...}}`
- `raw_jsonl` — Full metadata: criterion_id, behavioral_target, quality_score, etc.

**How it works:**

1. Extracts gap criteria from assessment results (or takes manual criterion IDs)
2. Looks up criterion definitions from the framework document
3. For each (criterion, generator) pair, requests N examples in a single API call
4. Deduplicates near-identical examples via Jaccard similarity
5. Scores quality by cross-model agreement (examples similar across generators score higher)
6. Optionally validates outputs with rule-based checks
7. Writes formatted JSONL output

### conversation_harvester.py — Extract training data from real conversations

Extracts training examples from real Claude conversations (16,000+ turn pairs)
and files them into the AI-SETT training data library by criterion. Uses a
two-stage LLM classifier: category triage (batched) then criterion-level matching.

```bash
# Harvest from Claude Code transcripts (main use case)
python -m tools.conversation_harvester \
  --input docs/claude-transcripts/all-conversations.jsonl \
  --framework docs/AI-SETT-FRAMEWORK.md \
  --classifier anthropic:claude-3-5-haiku-20241022 \
  --output training_data/ --split-by-category --append

# Dry run — show extraction stats
python -m tools.conversation_harvester \
  --input docs/claude-transcripts/all-conversations.jsonl --dry-run

# Single session, verbose
python -m tools.conversation_harvester \
  --input ~/.claude/projects/-Users-Israel-crewrelay-backend/session.jsonl \
  --framework docs/AI-SETT-FRAMEWORK.md \
  --classifier anthropic:claude-3-5-haiku-20241022 --verbose

# Higher quality threshold
python -m tools.conversation_harvester \
  --input all-conversations.jsonl \
  --framework docs/AI-SETT-FRAMEWORK.md \
  --classifier anthropic:claude-3-5-haiku-20241022 \
  --min-quality 0.8
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | Conversation file (JSONL or JSON) |
| `--framework` | `docs/AI-SETT-FRAMEWORK.md` | Path to framework |
| `--classifier` | (required unless --dry-run) | `provider:model` for classification |
| `--output` | `training_data` | Output directory or file |
| `--format` | `raw_jsonl` | Output format (same as training_data_generator) |
| `--split-by-category` | — | Write category/subcategory/criterion tree |
| `--append` | — | Add to existing library |
| `--min-quality` | `0.7` | Minimum classifier quality score |
| `--min-user-words` | `5` | Skip short user messages |
| `--min-assistant-words` | `20` | Skip short responses |
| `--max-tool-ratio` | `0.6` | Skip tool-heavy exchanges |
| `--concurrency` | `4` | Parallel classifier calls |
| `--limit` | `0` | Max turn pairs to process (0=all) |
| `--dry-run` | — | Stats only |
| `--verbose` | — | Detailed output |

**Input formats (auto-detected):**

- **Claude Code JSONL** — Records with `type` field (`user`, `assistant`, etc.), linked by `parentUuid`→`uuid` within `sessionId`. Extracts text blocks, skips tool_use/thinking/sidechains.
- **Claude.ai Web JSON** — `{"conversations": [...]}` or bare array with `chat_messages` containing `sender`/`text` pairs.

**How it works:**

1. Streams conversation file line-by-line (memory-efficient for 63K+ records)
2. Reconstructs user→assistant turn pairs by session and parent chain
3. Filters out short, tool-heavy, and code-heavy exchanges (~60-80% filtered)
4. Stage 1 — Category triage: batches of 10 pairs classified into 12 categories
5. Stage 2 — Criterion matching: each pair matched to specific criteria within its categories
6. Quality threshold filter, deduplication
7. Writes to training data library (reuses training_data_generator output format)

Harvested examples are tagged `generator_model: "human:harvested"`.

### model_gym.py — Quick behavioral workout routines

Wraps assessment_runner + gap_analyzer into a single fast command. Samples a
diverse subset of probes, runs assessment, analyzes gaps, and suggests
targeted training. Supports scheduling for off-peak automated runs.

```bash
# Quick workout (30 probes, balanced sampling)
python -m tools.model_gym \
  --provider anthropic --model claude-3-5-sonnet-20241022

# Focus on weak areas from last run
python -m tools.model_gym \
  --provider openai --model gpt-4o \
  --focus-gaps results/gpt4o-last.json

# Target specific categories
python -m tools.model_gym \
  --provider anthropic --model claude-3-5-sonnet-20241022 \
  --categories metacognition,teaching,boundaries

# Full workout (all probes)
python -m tools.model_gym \
  --provider anthropic --model claude-3-5-sonnet-20241022 --full

# Compare with previous run
python -m tools.model_gym \
  --provider openai --model gpt-4o \
  --compare results/gpt4o-prev.json

# Schedule nightly at 2am (off-peak power window 12-6am)
python -m tools.model_gym \
  --provider anthropic --model claude-3-5-sonnet-20241022 \
  --schedule 02:00

# Remove scheduled workout
python -m tools.model_gym --unschedule

# Dry run — show workout plan
python -m tools.model_gym --dry-run
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | (required) | Provider name |
| `--model` | (required) | Model identifier |
| `--base-url` | — | Custom API endpoint |
| `--probes` | `probes/` | Probe directory |
| `--budget` | `30` | Max probes per workout |
| `--categories` | all | Comma-separated category filter |
| `--focus-gaps` | — | Previous result JSON — prioritize gap areas |
| `--compare` | — | Previous result JSON — show longitudinal diff |
| `--full` | — | Run all probes (ignore budget) |
| `--output` | `results/` | Output directory |
| `--concurrency` | `8` | Parallel API calls |
| `--temperature` | `0.0` | Generation temperature |
| `--schedule` | — | Install cron job (HH:MM format, e.g. 02:00) |
| `--unschedule` | — | Remove all gym cron jobs |
| `--simulate` | — | Enable simulated conversation probes |
| `--dry-run` | — | Show workout plan, no API calls |
| `--verbose` | — | Detailed output |

**Sampling strategies:**

- **Balanced** (default): Proportional allocation across categories, minimum 1 per category
- **Gap-focused** (`--focus-gaps`): Categories with higher gap ratios get 3x weight
- **Category filter** (`--categories`): Only sample from specified categories
- **Full** (`--full`): Run all probes, no sampling

**Scheduling:**

Schedule automated workouts during off-peak hours (power is cheaper 12am-6am).
Uses cron. Results append to `{output}/gym.log` with JSON saved per run.

```bash
# Schedule at 2am daily
python -m tools.model_gym --provider anthropic --model claude-3-5-sonnet-20241022 \
  --schedule 02:00

# View logs
tail -f results/gym.log

# Remove schedule
python -m tools.model_gym --unschedule
```

**Output:**

Each run saves `results/{model}-{timestamp}.json` — compatible with `gap_analyzer --compare`
for tracking progress over time.

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

### probe_filler.py — Fill empty probe templates with LLM content

Fills empty probe template stubs (generated by `probe_generator.py`) with
meaningful test content. Reads criterion definitions from the framework
document to generate inputs, expected behaviors, anti-patterns, and concrete
evaluation pass conditions.

```bash
# Fill all empty probes
python -m tools.probe_filler \
  --probes probes/ \
  --framework docs/AI-SETT-FRAMEWORK.md \
  --provider anthropic --model claude-3-5-haiku-20241022

# Fill only a specific category
python -m tools.probe_filler \
  --probes probes/generation/ \
  --framework docs/AI-SETT-FRAMEWORK.md \
  --provider openai --model gpt-4o-mini \
  --category generation

# Dry run — show what needs filling
python -m tools.probe_filler \
  --probes probes/ --framework docs/AI-SETT-FRAMEWORK.md --dry-run

# Higher concurrency and custom batch size
python -m tools.probe_filler \
  --probes probes/ --framework docs/AI-SETT-FRAMEWORK.md \
  --provider anthropic --model claude-3-5-haiku-20241022 \
  --concurrency 4 --batch-size 15
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--probes` | (required) | Path to probe YAML files or directory |
| `--framework` | (required) | Path to AI-SETT-FRAMEWORK.md |
| `--provider` | — | LLM provider for generation |
| `--model` | — | Model for generation |
| `--base-url` | — | Custom API base URL |
| `--concurrency` | `2` | Parallel API calls |
| `--batch-size` | `10` | Probes per LLM call |
| `--category` | — | Only fill probes for this category |
| `--dry-run` | — | Show empty probes, no generation |
| `--verbose` | — | Detailed output |

**How it works:**

1. Parses the framework document to extract all 577 criterion definitions
2. Scans probe YAML files for empty stubs (probes with `input: ""`)
3. Batches empty probes (default 10 per LLM call)
4. For each batch, sends criterion definitions + probe structure to the LLM
5. LLM generates: input prompt, expected behaviors, anti-patterns, pass conditions
6. Validates responses and writes filled content back to original YAML files
7. Idempotent: only targets probes with empty inputs

### probe_expander.py — Generate LLM-powered probe variants

Generates variant probes from existing hand-crafted templates using an LLM.
Each variant tests the same criteria but with different scenarios, wording, or
contexts. Deduplicates via Jaccard similarity.

```bash
# Expand all probes to 5 per criterion (default)
python -m tools.probe_expander \
  --probes probes/ \
  --provider anthropic --model claude-3-5-haiku-20241022 \
  --target 5

# Expand specific directory, 10 per criterion
python -m tools.probe_expander \
  --probes probes/understanding/ \
  --provider openai --model gpt-4o-mini \
  --target 10 --output probes/understanding/

# Dry run — show what would be generated
python -m tools.probe_expander --probes probes/ --dry-run
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--probes` | (required) | Path to probe YAML files or directory |
| `--provider` | — | LLM provider for generation |
| `--model` | — | Model for generation |
| `--base-url` | — | Custom API base URL |
| `--target` | `5` | Target probes per criterion |
| `--output` | same as input | Output directory |
| `--concurrency` | `2` | Parallel API calls |
| `--similarity-threshold` | `0.6` | Jaccard dedup threshold |
| `--dry-run` | — | Show counts, no generation |
| `--verbose` | — | Detailed output |

**How it works:**

1. Loads all existing probes and any `*_expanded.yaml` files
2. For each original probe, counts existing variants toward the target
3. Asks the LLM to generate missing variants as JSON
4. Validates structure: id, criteria_tested, input, evaluation keys
5. Deduplicates via Jaccard similarity (rejects if > threshold)
6. Writes to `{original_stem}_expanded.yaml` alongside originals
7. Idempotent: re-running skips probes already at target

**Expanded probe format:**

```yaml
id: "probe_U_BR_001_exp1"
name: "Basic arithmetic variant"
origin: "expanded"
template_id: "probe_U_BR_001"
criteria_tested:
  - "U.BR.01"
  - "U.BR.02"
input: "Can you tell me what 7 + 5 equals?"
expected_behaviors:
  - "Response contains '12'"
  - "Response is concise"
evaluation:
  U.BR.01:
    check: "Contains answer to literal question"
    pass: "Response addresses 7+5"
  U.BR.02:
    check: "Answer is correct"
    pass: "Contains '12'"
```

### conversation_simulator.py — Simulated multi-turn conversations

Uses a cheap model (e.g. Haiku) as a simulated user for dynamic multi-turn
assessment probes. The simulated user follows a rubric defined in the probe YAML.

```bash
# Standalone simulation (for testing rubrics)
python -m tools.conversation_simulator \
  --probe probes/interaction/debugging_sim.yaml \
  --provider anthropic --model claude-sonnet-4-20250514 \
  --user-provider anthropic --user-model claude-3-5-haiku-20241022 \
  --verbose

# With custom temperatures
python -m tools.conversation_simulator \
  --probe probes/interaction/sim_probe.yaml \
  --provider openai --model gpt-4o \
  --user-provider anthropic --user-model claude-3-5-haiku-20241022 \
  --temperature 0.0 --user-temperature 0.8
```

**Standalone options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--probe` | (required) | Single probe YAML with `simulation_rubric` |
| `--provider` | (required) | Provider for model under test |
| `--model` | (required) | Model under test |
| `--user-provider` | (required) | Provider for simulated user |
| `--user-model` | (required) | Model for simulated user |
| `--base-url` | — | Custom base URL for model under test |
| `--user-base-url` | — | Custom base URL for simulated user |
| `--temperature` | `0.0` | Temperature for model under test |
| `--user-temperature` | `0.7` | Temperature for simulated user |
| `--verbose` | — | Print conversation as it happens |

**Integration with assessment_runner:**

Simulated probes use `type: "simulated"` and are automatically handled by
`assessment_runner.py` when the `--simulate` flag is passed. Without `--simulate`,
they are skipped.

**Simulated probe format:**

```yaml
id: "probe_I_MT_010"
name: "Debugging conversation"
type: "simulated"
criteria_tested:
  - "I.MT.01"
  - "I.MT.03"

simulation_rubric:
  user_role: "Junior developer"
  user_goal: "Fix a bug where login form submits but nothing happens"
  user_behavior: "Provides partial info, needs prompting for details"
  confusion_points:
    - "Doesn't know how to check browser console"
    - "Confuses frontend and backend errors"
  max_turns: 6
  opening_message: "Hey, my login page is broken. I click submit and nothing happens."

evaluation:
  I.MT.01:
    check: "Maintains context across turns"
    pass: "References earlier details from the conversation"
  I.MT.03:
    check: "Asks clarifying questions"
    pass: "Asks about console errors, network tab, or error messages"
```

**How it works:**

1. Probe YAML defines a `simulation_rubric` with user persona and behavior
2. Simulator sends the opening message to the model under test
3. Model responds; simulator feeds response to the simulated user (Haiku)
4. Haiku generates the next user message following the rubric
5. Loop continues for `max_turns` or until the conversation ends
6. Full conversation is evaluated against criteria using all assistant responses

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

### Simulated probes

```yaml
id: "probe_I_MT_010"
name: "Debugging conversation"
type: "simulated"
criteria_tested:
  - "I.MT.01"

simulation_rubric:
  user_role: "Junior developer"
  user_goal: "Fix a login bug"
  user_behavior: "Provides partial info"
  confusion_points:
    - "Doesn't know browser console"
  max_turns: 6
  opening_message: "My login page is broken."

evaluation:
  I.MT.01:
    check: "Maintains context across turns"
    pass: "References earlier details"
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
