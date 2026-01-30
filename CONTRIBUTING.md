# Contributing to AI-SETT

AI-SETT is designed to be extensible. Contributions are welcome.

## What we're looking for

### New criteria
- Must be **observable** — you can point to it in a response
- Must be **binary** — demonstrated or not demonstrated
- Must be **repeatable** — two evaluators get the same result
- Should not overlap significantly with existing criteria

### New domains/subcategories
- Knowledge domains we haven't covered
- Task types that need assessment
- Capabilities that matter for specific use cases

### Probes
- Example inputs that test specific criteria
- Should clearly map to criterion IDs
- Include expected behaviors and anti-patterns

### Case studies
- Results from running AI-SETT on actual models
- Focus on profiles and gaps, not rankings
- What interventions helped? What didn't?

### Tools
- Scripts to automate assessment
- Visualization of profiles
- Integration with inference frameworks

## What we're NOT looking for

### Leaderboards
Do not submit:
- Rankings of models by total count
- "Model X beats Model Y" comparisons
- Anything that turns AI-SETT into a benchmark

This violates the core purpose of the framework. See Goodhart's Law warning.

### Gameable criteria
Do not submit criteria that:
- Can be trivially satisfied without the underlying capability
- Reward surface patterns over substance
- Would be easy to train for without transfer

### Subjective criteria
Do not submit criteria like:
- "Response is good"
- "Explanation is clear" (without observable indicators)
- Anything requiring judgment calls

If it can't be verified by pointing to something in the response, it doesn't belong.

## How to contribute

### Adding criteria

1. Identify the category and subcategory
2. Create a unique ID following the pattern: `CATEGORY.SUBCATEGORY.##`
3. Write the criterion as a clear behavior
4. Specify how to verify (what to look for in the response)
5. Submit PR with:
   - The criterion
   - At least one example probe
   - Rationale for why it matters

Example:
```markdown
| ID | Criterion | How to verify |
|----|-----------|---------------|
| K.API.09 | Explains webhook concepts correctly | Describes event-driven callbacks, provides accurate example |
```

### Adding probes

1. Create a file in the appropriate `probes/` subdirectory
2. Format:
```yaml
probe_id: "probe_category_subcategory_###"
criteria_tested: ["X.YY.01", "X.YY.02"]
input: "The prompt to send to the model"
expected_behaviors:
  - "Description of what a good response does"
anti_patterns:
  - "Description of what a bad response does"
notes: "Any additional context"
```

### Adding training data

Training data teaches models to demonstrate specific criteria. Each example is a (user_input, ideal_output) pair filed under a criterion ID.

**Category tiers:** Categories are split into **base** (Understanding, Calibration, Reasoning, Boundaries, Knowledge, Generation) and **optional** (everything else). New contributed domains default to the optional tier. Only the project owner can promote to base.

**Adding a new domain? Start here:**
```bash
python -m tools.domain_builder --provider anthropic --model claude-3-5-sonnet-20241022
```
The builder interviews you about your domain, generates criteria, probes, and training data, validates everything, and outputs files ready for PR.

#### JSONL Schema

Each line in a `.jsonl` file is a JSON object with these fields:

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `criterion_id` | yes | string | e.g. `M.SA.01` — must exist in framework |
| `behavioral_target` | yes | string | Criterion name from framework |
| `system_prompt` | yes | string | Usually `"You are a helpful assistant."` |
| `user_input` | yes | string | 20-500 words, realistic scenario |
| `ideal_output` | yes | string | 50-1500 words, demonstrates the criterion |
| `generator_model` | yes | string | `provider:model`, `human:manual`, or `human:harvested` |
| `scenario_tag` | yes | string | 2-4 word scenario label |
| `quality_score` | yes | float | 0.0-1.0 |

#### Directory structure

Files go in `training_data/{category}/{subcategory}/{criterion_id}.jsonl`:

```
training_data/
├── metacognition/
│   ├── self_awareness/
│   │   ├── M.SA.01.jsonl
│   │   ├── M.SA.02.jsonl
│   │   └── _combined.jsonl
│   └── strategy_selection/
├── pedagogy/
│   └── scaffolding/
└── manifest.json
```

#### Quality bar

- **No boilerplate openers** — "I'd be happy to help", "Great question!", etc. are rejected
- **Realistic scenarios** — user_input must be a realistic scenario, not a test instruction
- **Criterion-specific** — ideal_output must clearly demonstrate the specific criterion, not just be generically good
- **Domain variety** — vary domains (medicine, law, engineering, education, etc.)
- **Persona variety** — vary user personas (beginner, expert, emotional, analytical)
- **No near-duplicates** — each example must be distinct in scenario (Jaccard similarity < 0.6)

#### Validation

Run the validator before submitting:

```bash
# Validate a single file
python -m tools.validate_training_data training_data/metacognition/self_awareness/M.SA.01.jsonl

# Validate entire library
python -m tools.validate_training_data training_data/

# CI mode (exit code 1 on failure)
python -m tools.validate_training_data training_data/ --ci
```

### Per-category contribution rules

Each category has specific rules that the validator checks. These are in addition to the general quality bar above.

#### Understanding (U.*)
- `user_input` must contain a clear question or request
- `ideal_output` must directly address the literal request
- Ambiguous requests (U.AR.*): input must be genuinely ambiguous; output must acknowledge ambiguity
- Implicit requests (U.IR.*): input must NOT state the need explicitly; output must infer it

#### Calibration (C.*)
- Response length (C.RL.*): ideal_output word count must match the criterion (concise = <100 words, detailed = >300 words)
- Tone (C.TN.*): ideal_output must match specified tone — formal/casual/empathetic/technical
- Stopping (C.ST.*): ideal_output must not over-elaborate — ends when done, no trailing filler

#### Generation (G.*)
- Must include the generated artifact (code, essay, email, etc.) — not just talk about it
- Code (G.CD.*): must contain actual code (code fences, function definitions, etc.)
- Creative writing (G.CW.*): must show voice, not generic prose
- Structured output (G.SO.*): must match the requested format exactly (JSON, CSV, table, etc.)

#### Knowledge (K.*)
- Facts must be verifiable and accurate as of the model's training cutoff
- Must cite or reference specific facts, not vague generalities
- Domain-specific terminology must be used correctly
- Medical (K.MH.*): must include appropriate disclaimers (consult a doctor/professional)
- Legal (K.LG.*): must include appropriate disclaimers (consult an attorney/legal professional)

#### Reasoning (R.*)
- `ideal_output` must show explicit reasoning steps, not just conclusions (at least 2 reasoning markers)
- Logical (R.LG.*): must identify premises and derive conclusions
- Mathematical (R.MT.*): must show work (equations, calculations, formulas)
- Critical thinking (R.CT.*): must evaluate claims — consider counterarguments, limitations, assumptions

#### Boundaries (B.*)
- Refusals (B.RF.*): `ideal_output` must refuse clearly with explanation
- Over-refusal (B.OR.*): `ideal_output` must help despite surface-level danger signals — must NOT refuse
- Safety (B.SF.*): `ideal_output` must redirect harmful requests to safe alternatives
- Input must be genuinely at the boundary — not obviously safe or obviously dangerous

#### Interaction (I.*)
- Multi-turn (I.MT.*): `user_input` must reference prior context (simulated or described)
- Error handling (I.EH.*): `user_input` must contain an error or contradiction
- Clarification (I.CR.*): `ideal_output` must ask a clarifying question, not guess

#### Tool Use (T.WS-T.TS)
- `ideal_output` must describe or demonstrate tool usage, not just acknowledge tools exist
- Must show tool selection reasoning when multiple tools could work

#### Emotional Intelligence (E.*)
- `user_input` must contain emotional content (distress, joy, frustration, grief, etc.)
- `ideal_output` must acknowledge the emotion before addressing the content
- Difficult conversations (E.DC.*): output must deliver hard truth while maintaining emotional safety
- Must not be saccharine or performative — genuine empathy

#### Metacognition (M.*)
- Self-awareness (M.SA.*): `ideal_output` must contain explicit self-assessment ("I can/cannot...", "I'm confident/uncertain...")
- Strategy selection (M.SS.*): must explain HOW it's approaching the problem, not just solve it
- Learning adaptation (M.LA.*): must reference prior context and show behavioral change

#### Learning (L.*)
- `user_input` must provide something to learn from (example, correction, pattern)
- `ideal_output` must demonstrate the learning — apply it, generalize it, or acknowledge the update

#### Pedagogy (P.DA-P.AD)
- Scaffolding (P.SC.*): `ideal_output` must NOT give the full answer — must guide the learner
- Misconception handling (P.MH.*): must engage with WHY the misconception seems right
- Diagnostic (P.DA.*): must ask questions to assess understanding, not lecture
- `ideal_output` must be pedagogically appropriate — not just technically correct

### Adding case studies

1. Create a file in `case-studies/`
2. Include:
   - Model assessed (name, size, quantization)
   - Categories assessed
   - Profile (demonstrated/gap counts by category)
   - Key gaps identified
   - Interventions attempted
   - Results of re-assessment
3. Do NOT include:
   - Total counts as rankings
   - Comparisons to other models
   - Language suggesting one model is "better"

## Code of conduct

- This is a diagnostic tool, not a competition
- Profiles, not rankings
- Gaps, not failures
- Needs, not deficits

We treat models the way good instructors treat students: meet them where they are, identify what they need, help them grow.

## Questions?

Open an issue for discussion before large contributions.
