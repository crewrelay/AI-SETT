# Goodhart's Law and AI-SETT

> "When a measure becomes a target, it ceases to be a good measure."
> -- Charles Goodhart (paraphrased by Marilyn Strathern)

## Why this warning exists

AI-SETT contains 600 observable criteria. The moment someone publishes "Model X scored 547/600," this framework dies. The count becomes a target, labs optimize for the count, and the diagnostic value disappears.

This has happened to every benchmark. It will happen to AI-SETT if we let it.

## How benchmarks die

### The lifecycle
1. Researchers create a benchmark to measure capability
2. Labs use it to identify gaps (diagnostic phase -- this is good)
3. Labs publish scores as marketing ("State of the art on X!")
4. Other labs optimize specifically for the benchmark
5. Benchmark scores go up, but real capability doesn't
6. Benchmark becomes meaningless
7. New benchmark needed

### The four types of Goodhart corruption

**Regressional:** The metric correlates with the goal, but optimizing the metric doesn't optimize the goal. Training to pass AI-SETT criteria without building underlying capability.

**Extremal:** At extreme values, the relationship between metric and goal breaks down. A model that scores 600/600 on AI-SETT is not necessarily the best model -- it may have learned to satisfy surface patterns.

**Causal:** The metric and goal share a common cause but aren't causally linked. High AI-SETT counts correlate with capable models, but artificially inflating counts doesn't create capability.

**Adversarial:** Once agents know the metric, they game it. Publishing AI-SETT criteria publicly (which we do) means models can be specifically trained to pass them.

## How AI-SETT resists Goodhart (partially)

1. **No ceiling** -- There's no "100%" to optimize toward. The count just grows as criteria are added.
2. **Observable behaviors** -- Harder to fake than multiple choice. You have to actually demonstrate the behavior.
3. **Profile over score** -- The framework emphasizes the shape of capabilities, not a single number.
4. **Extensible** -- New criteria can always be added, moving the target.

## How AI-SETT could still fail

1. Labs publish total counts as marketing
2. Training data includes AI-SETT probes verbatim
3. Models learn to pattern-match criteria without understanding
4. The community treats profiles as rankings
5. "AI-SETT certified" becomes a meaningless badge

## How to use without corrupting

- Share profiles, not scores
- Use to identify gaps, then train the underlying capability
- Embrace the +0 list (gaps matter more than count)
- Pair with qualitative/narrative assessment
- Evolve criteria over time
- Never compare models by total count
- Focus on "what does this model need?" not "how does it rank?"

## The meta-warning

This document itself is subject to Goodhart's Law. The moment "we follow Goodhart's Law guidelines" becomes a badge of legitimacy, it loses meaning. The point isn't compliance with these rules. The point is genuine diagnostic use.

*If you're reading this to check a box, you've already missed the point.*
