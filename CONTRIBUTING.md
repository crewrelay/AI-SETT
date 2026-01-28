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
