# Case studies

Real-world applications of AI-SETT assessment.

## Purpose

Case studies document:
- How AI-SETT was applied to specific models
- What profiles emerged
- What gaps were identified
- What interventions were tried
- What changed after intervention

## What belongs here

✅ **Profiles** — "Model X demonstrated 45/65 generation criteria, with gaps in structured output and summarization"

✅ **Gap analysis** — "Primary gaps were in calibration (response length) and boundaries (over-refusal)"

✅ **Intervention reports** — "After training on 200 calibration examples, response length criteria improved from 2/6 to 5/6"

✅ **Methodology notes** — "We assessed only knowledge domains relevant to our use case (programming, APIs, security)"

## What does NOT belong here

❌ **Rankings** — "Model X scored 423, beating Model Y's 398"

❌ **Comparisons** — "Model X is better than Model Y"

❌ **Total scores as primary metric** — "Our model achieved 547/600"

This violates Goodhart's Law and corrupts the framework.

## Format

Each case study should include:

```markdown
# Case Study: [Model Name]

## Context
- Model: [name, size, quantization]
- Use case: [what it will be used for]
- Assessment date: [date]

## Scope
- Categories assessed: [list]
- Criteria assessed: [count]
- Probes used: [count]

## Profile

### Demonstrated behaviors by category
| Category | Demonstrated | Total assessed |
|----------|--------------|----------------|
| Understanding | X | Y |
| Calibration | X | Y |
| ... | ... | ... |

### Key gaps identified
1. [Gap 1 - specific criteria]
2. [Gap 2 - specific criteria]
3. [Gap 3 - specific criteria]

## Intervention

### Approach
[What was done to address gaps]

### Training data
[If applicable - type, quantity, focus]

### Scaffolds used
[If applicable - skills, few-shot, etc.]

## Re-assessment

### Changes observed
| Category | Before | After | Change |
|----------|--------|-------|--------|
| ... | ... | ... | ... |

### Gaps closed
[List]

### Gaps remaining
[List]

## Lessons learned
[What worked, what didn't, what to try next]
```

## Submitting case studies

See CONTRIBUTING.md for guidelines.
