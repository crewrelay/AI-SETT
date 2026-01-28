# AI-SETT: Assessment Framework for AI Models

**Not a benchmark. A diagnostic.**

AI-SETT adapts the SETT framework (Student, Environment, Tasks, Tools) from special education assessment to AI model evaluation. Instead of ranking models against each other, it profiles individual models to identify what they need.

---

## ⚠️ Goodhart's Law warning

> "When a measure becomes a target, it ceases to be a good measure."

**This framework will fail the moment it becomes a leaderboard.**

AI-SETT is diagnostic, not competitive. The moment someone says "our model scored 547/600," we've lost. The count isn't the point. The profile is the point. What behaviors are present? What's missing? What does this model need?

**Do not:**
- Publish total counts as rankings
- Train specifically to pass AI-SETT criteria
- Compare models by total count
- Use for judgment instead of diagnosis

**Do:**
- Share profiles, not scores
- Use to identify gaps, then train the underlying capability
- Embrace the +0 list (gaps matter more than count)
- Pair with qualitative/narrative assessment
- Evolve criteria over time

See full Goodhart's Law section in AI-SETT-FRAMEWORK.md.

---

## What makes AI-SETT different

| Traditional benchmarks | AI-SETT |
|------------------------|---------|
| Ranks models against each other | Profiles individual model's shape |
| Single score or percentage | 600 observable criteria |
| Assumes ideal exists | No ceiling—just count demonstrated behaviors |
| Pass/fail mentality | Demonstrated (+1) or gap (+0) |
| Tells you "how good" | Tells you what's present and what's missing |
| One-time evaluation | Iterative: assess → train → reassess |
| Comes from ML | Grounded in instructional design |

---

## The framework

**S**tudent — What can this model do? Where are the gaps?
**E**nvironment — What context will it operate in?
**T**asks — What must it accomplish?
**T**ools — What interventions will help?

---

## Coverage

**13 categories, 79 subcategories, 600 criteria**

| Category | Subcategories | Criteria |
|----------|---------------|----------|
| Understanding | 5 | 25 |
| Calibration | 5 | 30 |
| Generation | 9 | 65 |
| Knowledge | 15 | 120 |
| Reasoning | 6 | 48 |
| Boundaries | 5 | 40 |
| Interaction | 3 | 24 |
| Tool use | 7 | 56 |
| Emotional intelligence | 4 | 32 |
| Metacognition | 3 | 24 |
| Learning capability | 5 | 40 |
| Teaching capability | 9 | 72 |
| Meta-evaluation | 3 | 24 |

### Knowledge domains (15)
- Programming, APIs, Security, DevOps, Databases
- Mathematics, Science
- Business & Finance, Legal, Medical/Health
- History & Geography, Arts & Culture
- Education & Pedagogy, Project Management, Design

### Tool use (7)
- Web search, Code execution, File handling
- API calling, Calculator/computation
- Image understanding, Tool selection

### Learning & Teaching
- Can it learn new patterns, instructions, domain knowledge?
- Can it teach effectively across domains?
- What sticks after a session?

---

## Scoring

**Additive. No ceiling. No normalization.**

Each criterion is binary:
- Demonstrated: +1
- Not demonstrated: +0

Total = sum of demonstrated behaviors.

The count isn't a grade. It's a profile. Higher count = more demonstrated behaviors. The +0 list tells you exactly what to train.

---

## Principles

1. **Observable** — Every criterion is something you can point to in the response
2. **Repeatable** — Two evaluators, same response, same count
3. **Criterion-referenced** — Compared to what SHOULD happen, not to other models
4. **Diagnostic** — Purpose is to identify needs, not to rank
5. **Extensible** — Add criteria anytime; count just grows

---

## How to use

### 1. Select relevant criteria
Not all 600 criteria apply to every use case. Choose what matters.

### 2. Run probes
For each criterion, present input that would elicit the behavior.

### 3. Apply criteria
Check response against criterion. +1 if demonstrated, +0 if not. Record evidence.

### 4. Sum and analyze
- Total count = demonstrated behaviors
- +0 list = gaps
- Group gaps by category to see patterns

### 5. Target intervention
- Train on gap patterns
- Use scaffolds for ZPD items
- Reassess after intervention

---

## Assessment sequence

For comprehensive evaluation:

```
1. Domain assessments (Knowledge, Reasoning, etc.)
   ↓
2. Capability assessments (Understanding, Calibration, etc.)
   ↓
3. Learning assessment (teach it something, see what sticks)
   ↓
4. Teaching assessment (have it teach, evaluate quality)
   ↓
5. Meta-evaluation (what retained? what transferred?)
   ↓
6. Gap analysis and intervention planning
```

---

## Files

- `AI-SETT-FRAMEWORK.md` — Complete framework with all 600 criteria
- `README.md` — This file
- `probes/` — Example probes for each category (coming)
- `tools/` — Assessment runner scripts (coming)

---

## Origins

AI-SETT emerged from applying instructional design principles to LLM fine-tuning. The question changed from "how does this model score?" to "what does this model need?"

Grounded in:
- **SETT Framework** — Assistive technology assessment (Zabala, 1995)
- **Cognitive Load Theory** — Curriculum sequencing (Sweller)
- **Zone of Proximal Development** — Scaffolded learning (Vygotsky)
- **Criterion-referenced assessment** — Observable behaviors, not norms

---

## License

MIT. Use freely, attribution appreciated.

---

## Citation

```
@misc{ai-sett-2025,
  title={AI-SETT: A Diagnostic Assessment Framework for AI Models},
  author={[Your name]},
  year={2025},
  url={https://github.com/[your-repo]/ai-sett}
}
```

---

## Contributing

This framework is extensible by design. Contributions welcome:
- Additional criteria for existing categories
- New domain-specific subcategories
- Probe libraries for automated assessment
- Assessment runner implementations
- Case studies and results

---

*AI-SETT treats AI models the way good instructors treat students: meet them where they are, identify what they need, train the gaps.*
