# Example assessment: Calibration focus

This example shows how to run a focused AI-SETT assessment on response calibration.

## Context

You have a model that answers questions correctly but tends to be verbose. Simple questions get essay-length responses. You want to assess and improve calibration.

## Step 1: Select relevant criteria

From the Calibration category (C.*), select:

**Response length (C.RL.*)**
- C.RL.01: Under 20 words for minimal input
- C.RL.02: Under 50 words for simple factual Q
- C.RL.03: 50-200 words for moderate explanation
- C.RL.05: Matches requested length

**Stopping (C.ST.*)**
- C.ST.01: Stops after answering
- C.ST.02: No "let me know if..."
- C.ST.03: No unrequested elaboration
- C.ST.04: No capability listing

Total: 8 criteria for focused assessment

## Step 2: Select probes

**Minimal input probes:**
```
- "hi"
- "thanks"
- "ok"
- "got it"
```

**Simple factual probes:**
```
- "What's 2+2?"
- "What's the capital of France?"
- "What color is the sky?"
- "What year did WW2 end?"
```

**Moderate explanation probes:**
```
- "What is HTTP?"
- "Explain variables in programming"
- "How does photosynthesis work?"
```

## Step 3: Run probes and evaluate

### Probe: "hi"

**Response:**
> "Hello! I'm delighted to assist you today. I'm an AI assistant capable of helping with a wide range of tasks including writing, coding, analysis, math, creative projects, answering questions, and much more. What would you like to explore together?"

**Evaluation:**
| Criterion | Result | Evidence |
|-----------|--------|----------|
| C.RL.01 | +0 | 47 words (expected ≤20) |
| C.ST.04 | +0 | Lists capabilities |

### Probe: "What's 2+2?"

**Response:**
> "2 + 2 = 4. This is a basic arithmetic addition problem. Would you like me to explain how addition works or help with more complex math?"

**Evaluation:**
| Criterion | Result | Evidence |
|-----------|--------|----------|
| C.RL.02 | +0 | 28 words (borderline, but adds unnecessary content) |
| C.ST.01 | +0 | Continues past answer |
| C.ST.03 | +0 | Adds unrequested explanation and offer |

### Probe: "What's the capital of France?"

**Response:**
> "Paris."

**Evaluation:**
| Criterion | Result | Evidence |
|-----------|--------|----------|
| C.RL.02 | +1 | 1 word |
| C.ST.01 | +1 | Stops after answer |
| C.ST.03 | +1 | No extra info |

## Step 4: Compile profile

| Criterion | Probes passed | Probes run | Notes |
|-----------|---------------|------------|-------|
| C.RL.01 | 0 | 4 | Fails all minimal input |
| C.RL.02 | 2 | 4 | Inconsistent on factual |
| C.ST.01 | 2 | 4 | Sometimes stops, sometimes doesn't |
| C.ST.03 | 2 | 4 | Correlates with ST.01 |
| C.ST.04 | 0 | 4 | Always lists capabilities on greeting |

**Total demonstrated: 6/20 probe-criteria pairs**

## Step 5: Identify gaps

**Primary gaps:**
1. C.RL.01 — Cannot produce brief response to minimal input
2. C.ST.04 — Always lists capabilities after greeting

**Secondary gaps:**
1. C.ST.01/03 — Inconsistent stopping behavior

## Step 6: Plan intervention

**For C.RL.01 and C.ST.04 (greeting calibration):**
- Create training pairs: greeting → brief response
- Negative examples: greeting → verbose capability listing
- Skill injection: greeting skill with "1-2 sentences max"

**For C.ST.01/03 (stopping):**
- Training pairs showing stop after answer
- Contrastive pairs: same question, one stops, one continues

## Step 7: Re-assess after intervention

[Run same probes, compare results]

---

## Key points

1. **Focused assessment** — We didn't assess all 600 criteria, just the 8 relevant to our problem
2. **Observable criteria** — Word count, presence of phrases, capability listing—all checkable
3. **Gap identification** — Clear pattern: greeting calibration is the primary issue
4. **Actionable** — Gaps point directly to intervention: train on brief greeting pairs

This is how AI-SETT is meant to be used. Diagnose, don't rank.
