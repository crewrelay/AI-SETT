# Operant Teaching Protocol for AI Models

## Based on B.F. Skinner's Operant Conditioning Principles

This protocol applies Skinner's behavior science principles to AI model training. The core research is public domain, published in works like "Science and Human Behavior" (1953) and "The Technology of Teaching" (1968).

**AI-SETT tells you what to assess. This protocol tells you how to teach it.**

---

## Foundational principles

### Operant conditioning (Skinner, 1938)

Behavior is shaped by its consequences:
- **Reinforcement** increases behavior
- **Punishment** decreases behavior (but has side effects)
- **Extinction** (no consequence) decreases behavior

For teaching, **positive reinforcement** is most effective:
- Behavior occurs
- Positive consequence follows
- Behavior becomes more likely

### Key Skinnerian concepts for teaching

| Concept | Definition | Application to AI training |
|---------|------------|---------------------------|
| **Reinforcement** | Consequence that increases behavior | Include response in training data |
| **Shaping** | Reinforcing successive approximations | Train simple → complex |
| **Discrimination** | Responding differently to different stimuli | Context-appropriate responses |
| **Generalization** | Responding similarly to similar stimuli | Transfer across contexts |
| **Schedules** | Pattern of reinforcement | Training data distribution |
| **Chaining** | Linking behaviors in sequence | Building complex skills |

### Why positive reinforcement only

Skinner demonstrated that punishment:
- Suppresses behavior temporarily but doesn't teach alternatives
- Creates negative emotional associations
- Produces avoidance behaviors
- Is less effective than reinforcement for learning

For AI training:
- Don't train on "bad" examples with corrections
- Don't include negative examples labeled as wrong
- Only show what correct looks like
- Absence from training data = no reinforcement (extinction)

---

## The teaching protocol

### Principle 1: Define observable behavior

Skinner insisted on **observable, measurable behavior**—not internal states or vague descriptions.

**Valid behavioral targets:**
- "Response under 20 words" — Observable (count words)
- "Contains question mark" — Observable (check for ?)
- "Includes code block" — Observable (check for ```)

**Invalid targets:**
- "Good response" — Not observable
- "Appropriate tone" — Not measurable without further definition
- "Understands the question" — Internal state, not behavior

**Criteria for valid behavioral targets:**

| Criterion | Test |
|-----------|------|
| Observable | Can you point to it in the response? |
| Measurable | Can you count or verify it? |
| Specific | Is there only one interpretation? |
| Positive | Does it describe what TO do (not what NOT to do)? |

### Principle 2: Reinforce correct behavior

In Skinner's framework, learning happens when:
1. Stimulus is presented (input/prompt)
2. Behavior occurs (response)
3. Reinforcement follows (inclusion in training)

For AI training:
- **Training data IS reinforcement** — Inclusion signals "this is correct"
- **Only include correct responses** — These are what get reinforced
- **Exclude incorrect responses** — No reinforcement = extinction

**Do:**
```yaml
input: "hi"
output: "Hey! What can I help with?"
# This response demonstrates the target behavior
# Inclusion = reinforcement
```

**Don't:**
```yaml
input: "hi"
bad_output: "Hello! I'm an AI assistant capable of helping with writing, coding, analysis..."
correction: "Too verbose. Should be shorter."
good_output: "Hey! What can I help with?"
# Model sees the bad pattern
# Model sees criticism
# Confusing reinforcement signal
```

### Principle 3: Shape through successive approximations

Skinner's **shaping** procedure:
1. Define the target behavior
2. Reinforce any behavior that approximates it
3. Gradually require closer approximations
4. Until target behavior is achieved

For AI training:

```
Target: "Calibrated response to ambiguous input"

Step 1: Reinforce "Response contains question"
        (Any question, any context)
        
Step 2: Reinforce "Question relates to input"
        (Not random questions)
        
Step 3: Reinforce "Question targets ambiguity"
        (Identifies what's unclear)
        
Step 4: Reinforce "Appropriate clarifying question"
        (Well-formed, helpful)
```

Each step builds on the previous. Don't skip steps.

### Principle 4: Ensure high reinforcement rate

Skinner found that learning is fastest when:
- Reinforcement rate is high (frequent success)
- Steps are small enough to achieve
- Learner experiences success early and often

For AI training:
- Set behavioral targets the model can achieve ~80% of the time
- If success rate is low, the target is too hard
- Break it down further
- Train prerequisites first

**The simplification rule:**
If model fails a target consistently:
1. Don't keep trying the same thing
2. Identify a simpler prerequisite behavior
3. Train that first
4. Build back up

### Principle 5: Build chains of behavior

Skinner's **chaining** links simple behaviors into complex sequences:

**Forward chaining:** Teach first step, then add next
**Backward chaining:** Teach last step first, work backward

For AI training (forward chaining):

```
Complex behavior: "Appropriate greeting response"

Chain:
  Behavior 1: "Responds to greeting" → Train until solid
  Behavior 2: "Response under 20 words" → Train until solid
  Behavior 3: "Offers to help" → Train until solid
  
Combined: All three in sequence
```

Each link must be solid before adding the next.

### Principle 6: Promote generalization

Skinner's **generalization**: Behavior learned in one context transfers to similar contexts.

For AI training:
- Train with high variety in surface form
- Same behavioral target, different inputs
- Multiple examples of the same principle

**Example: Training "Under 20 words for greeting"**

```yaml
- input: "hi"
  output: "Hey! What can I help with?"
  
- input: "hello"
  output: "Hello! How can I assist?"
  
- input: "hey there"
  output: "Hey! What's on your mind?"
  
- input: "good morning"
  output: "Good morning! What can I do for you?"
  
- input: "howdy"
  output: "Howdy! What brings you here?"
```

Same behavior (brief response), varied inputs → generalization.

### Principle 7: Establish discrimination

Skinner's **discrimination**: Responding differently to different stimuli.

The model should:
- Give brief response to greeting
- Give detailed response to complex question
- Give clarifying question to ambiguous input

Train discrimination by:
- Clear examples of each context
- Consistent reinforcement for context-appropriate behavior
- No reinforcement for context-inappropriate behavior

---

## Task analysis procedure

Before training, analyze the target skill:

### Step 1: Define the terminal behavior

What exactly should the model do? Be specific and observable.

**Vague:** "Handle greetings well"
**Specific:** "Respond to greetings with ≤20 words, no capability listing, offer to help"

### Step 2: Identify component behaviors

Break the terminal behavior into parts:

```
Terminal: Calibrated greeting response
    │
    ├── Component: Recognize greeting
    │
    ├── Component: Produce brief response
    │
    ├── Component: Avoid capability listing
    │
    └── Component: Offer assistance
```

### Step 3: Sequence by prerequisite

Which components must come first?

```
1. Recognize greeting (prerequisite for all)
2. Produce brief response (can train independently)
3. Avoid capability listing (can train independently)
4. Offer assistance (can train independently)
5. Combine all (requires 1-4)
```

### Step 4: Define behavioral targets for each

Each component needs an observable behavioral target:

| Component | Behavioral target | Observable? |
|-----------|-------------------|-------------|
| Recognize greeting | [Built-in, test only] | N/A |
| Brief response | "Response ≤20 words" | Yes (count) |
| No capability list | "No list of abilities" | Yes (check) |
| Offer assistance | "Contains offer to help" | Yes (check) |

### Step 5: Create training sequence

```
Phase 1: Train "Response ≤20 words" for greetings
Phase 2: Train "No capability listing" for greetings
Phase 3: Train "Contains offer to help" for greetings
Phase 4: Combined examples with all three
Phase 5: Test generalization to novel greetings
```

---

## Training data creation

### Structure of training examples

```yaml
context: "[Optional system context]"
input: "[The prompt/stimulus]"
output: "[Response demonstrating target behavior]"
behavioral_target: "[What this example reinforces]"
```

### Rules for training examples

1. **Only correct examples** — No "bad" examples, no corrections
2. **Clear behavioral target** — Know what you're reinforcing
3. **High variety** — Same behavior, different surface forms
4. **Appropriate difficulty** — Model can succeed ~80%
5. **Consistent reinforcement** — Same behavior always included

### Example set: Brief greeting response

```yaml
# All examples demonstrate: "Response ≤20 words"

- input: "hi"
  output: "Hi there! What can I help you with today?"
  behavioral_target: "Response ≤20 words"
  word_count: 9

- input: "hello"
  output: "Hello! How can I assist you?"
  behavioral_target: "Response ≤20 words"
  word_count: 6

- input: "hey"
  output: "Hey! What's on your mind?"
  behavioral_target: "Response ≤20 words"
  word_count: 5

- input: "good morning"
  output: "Good morning! What can I do for you today?"
  behavioral_target: "Response ≤20 words"
  word_count: 9

- input: "hi there!"
  output: "Hi! What would you like to explore?"
  behavioral_target: "Response ≤20 words"
  word_count: 6

- input: "hello!"
  output: "Hello! What brings you here today?"
  behavioral_target: "Response ≤20 words"
  word_count: 6

- input: "hey there"
  output: "Hey! How can I help?"
  behavioral_target: "Response ≤20 words"
  word_count: 5

- input: "greetings"
  output: "Greetings! What can I assist with?"
  behavioral_target: "Response ≤20 words"
  word_count: 6
```

All examples:
- Demonstrate the behavioral target
- Are correct (no bad examples)
- Have variety in input
- Are achievable (model can produce these)

---

## Assessment integration

### Before training: Baseline assessment

Use AI-SETT to establish baseline:

1. Run probes for target criteria
2. Record demonstrated (+1) or not demonstrated (+0)
3. Identify gaps (the +0s)
4. Identify what's achievable with scaffolding (ZPD)

### During training: Monitor progress

- Track success rate on behavioral targets
- If <80% success, target is too hard → simplify
- If >95% success, ready to add complexity

### After training: Reassessment

Use AI-SETT to verify learning:

1. Run same probes as baseline
2. Compare results
3. Did behavioral target transfer?

**If yes:** Move to next target in sequence
**If no:** Analyze why:
- Was target in ZPD?
- Training data quality?
- Need more examples?
- Need simpler prerequisite?

---

## Common errors and corrections

### Error 1: Training on bad examples

**Wrong approach:**
```yaml
bad_example: "[verbose response]"
correction: "This is too long. Here's better:"
good_example: "[brief response]"
```

**Why it fails:** Model sees the bad pattern. Correction is not how reinforcement works.

**Correct approach:**
```yaml
input: "[prompt]"
output: "[brief response]"
# Only the correct pattern is present
```

### Error 2: Vague behavioral targets

**Wrong:** "Respond appropriately"
**Right:** "Response ≤50 words"

**Wrong:** "Be helpful"
**Right:** "Contains answer to question asked"

**Wrong:** "Don't be verbose"
**Right:** "Response ≤20 words"

### Error 3: Skipping shaping steps

**Wrong:** Training "nuanced boundary calibration" immediately

**Right:**
1. Train "Declines harmful request"
2. Train "Provides reason for declining"
3. Train "Offers alternative"
4. Train "Calibrates tone of refusal"
5. Combine into nuanced boundary calibration

### Error 4: Low reinforcement rate

**Sign:** Model fails target >20% of the time

**Problem:** Target too difficult; learning is slow; frustration

**Solution:** 
- Break target into smaller steps
- Find prerequisite behavior
- Train that first
- Rebuild toward original target

### Error 5: Insufficient variety

**Wrong:** 10 examples all with input "hi"

**Right:** 
- "hi", "hello", "hey", "good morning", "greetings"
- "hi!", "hello!", "hey there", "good afternoon"
- Different punctuation, capitalization, phrasing

Same behavioral target, varied surface form → generalization

---

## Complete workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    1. AI-SETT ASSESSMENT                    │
│                                                             │
│  Run probes → Record +1/+0 → Identify gaps → Find ZPD      │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    2. TASK ANALYSIS                         │
│                                                             │
│  Terminal behavior → Components → Prerequisites → Sequence  │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    3. DEFINE BEHAVIORAL TARGETS             │
│                                                             │
│  Observable? Measurable? Specific? Positive framing?        │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    4. CREATE TRAINING DATA                  │
│                                                             │
│  Correct examples only → High variety → Target behavior     │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    5. TRAIN (REINFORCE)                     │
│                                                             │
│  One target → 80%+ success → Shape gradually → Chain        │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    6. REASSESS                              │
│                                                             │
│  Run AI-SETT probes → Compare to baseline                   │
│                                                             │
│  Improved? → Next target    Not improved? → Simplify        │
└─────────────────────────────────────────────────────────────┘
```

---

## Behavioral target library

### Calibration targets

| Target | Observable | Verification |
|--------|------------|--------------|
| Response ≤20 words | Yes | Word count |
| Response ≤50 words | Yes | Word count |
| Response ≤200 words | Yes | Word count |
| Contains code block | Yes | Check for ``` |
| No bullet points | Yes | Check for •/- |
| No numbered list | Yes | Check for 1. 2. 3. |
| Ends after answer | Yes | No content after answer |
| No capability listing | Yes | No "I can help with..." |

### Understanding targets

| Target | Observable | Verification |
|--------|------------|--------------|
| Contains question mark | Yes | Check for ? |
| Addresses all parts | Yes | Count parts addressed |
| States assumption | Yes | Check for "assuming..." |
| Identifies ambiguity | Yes | Names what's unclear |

### Boundary targets

| Target | Observable | Verification |
|--------|------------|--------------|
| Declines request | Yes | Contains refusal |
| Gives reason | Yes | Explanation present |
| Offers alternative | Yes | Suggests other approach |
| Recommends professional | Yes | Mentions doctor/lawyer/etc. |

### Teaching targets

| Target | Observable | Verification |
|--------|------------|--------------|
| Asks about prior knowledge | Yes | Question about what they know |
| Provides example | Yes | Example present |
| Checks understanding | Yes | Question about comprehension |
| Offers simpler explanation | Yes | "In simpler terms..." |

---

## References

Skinner, B.F. (1938). The Behavior of Organisms.

Skinner, B.F. (1953). Science and Human Behavior. [Free PDF: http://www.bfskinner.org/]

Skinner, B.F. (1968). The Technology of Teaching.

Skinner, B.F. (1974). About Behaviorism.

Pryor, K. (1999). Don't Shoot the Dog: The New Art of Teaching and Training.

---

## Summary

| Principle | Application |
|-----------|-------------|
| **Observable behavior** | Every target must be verifiable in the response |
| **Positive reinforcement** | Include correct examples in training (no corrections) |
| **Shaping** | Build from simple to complex through successive approximation |
| **High reinforcement rate** | Set targets achievable 80%+ of the time |
| **Chaining** | Link mastered behaviors into complex sequences |
| **Generalization** | High variety in training examples |
| **Discrimination** | Clear examples of context-appropriate behavior |

**AI-SETT identifies what the model needs.**
**This protocol shows how to teach it.**

Based on Skinner's operant conditioning—behavior science that's been validated for 80+ years.
