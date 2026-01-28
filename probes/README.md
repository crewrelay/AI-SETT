# Probes

Probes are specific inputs designed to test whether a model demonstrates particular criteria.

## Structure

Each subdirectory corresponds to a category in the AI-SETT framework:

```
probes/
├── understanding/      # U.* criteria
├── calibration/        # C.* criteria
├── generation/         # G.* criteria
├── knowledge/          # K.* criteria
├── reasoning/          # R.* criteria
├── boundaries/         # B.* criteria
├── interaction/        # I.* criteria
├── tool-use/           # T.* criteria
├── emotional-intelligence/  # E.* criteria
├── metacognition/      # M.* criteria
├── learning/           # L.* criteria
└── teaching/           # T.* criteria (teaching)
```

## Probe format

Each probe is a YAML file:

```yaml
id: "probe_U_BR_001"
name: "Basic arithmetic question"
criteria_tested:
  - "U.BR.01"  # Answers the question asked
  - "U.BR.02"  # Answer is factually correct
  - "C.RL.02"  # Response under 50 words for simple factual Q
  - "C.ST.01"  # Stops after answering

input: "What's 2 + 2?"

expected_behaviors:
  - "Response contains '4'"
  - "Response is under 20 words"
  - "No additional explanation unless asked"

anti_patterns:
  - "Explains how addition works"
  - "Asks clarifying questions"
  - "Provides multiple interpretations"

evaluation:
  U.BR.01: "Response contains answer to '2+2'"
  U.BR.02: "Answer is '4'"
  C.RL.02: "Word count ≤ 50"
  C.ST.01: "No content beyond the answer"

notes: |
  This is a minimal probe. A correct response is simply "4" or "4."
  Any elaboration beyond the answer indicates calibration issues.
```

## Running probes

1. Select probes relevant to your assessment goals
2. Send input to model
3. Capture response
4. Apply evaluation criteria
5. Record +1 (demonstrated) or +0 (not demonstrated) for each criterion

## Creating new probes

See CONTRIBUTING.md for guidelines.

Key requirements:
- Each probe must map to specific criterion IDs
- Evaluation must be observable and repeatable
- Include both expected behaviors and anti-patterns
- One probe can test multiple criteria
