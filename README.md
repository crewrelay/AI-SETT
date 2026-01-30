# AI-SETT: Assessment Framework for AI Models

**Author**: CrewRelay
**Repository**: [https://github.com/crewrelay/AI-SETT](https://github.com/crewrelay/AI-SETT)

**Not a benchmark. A diagnostic.**

AI-SETT adapts the SETT framework (Student, Environment, Tasks, Tools) from special education assessment to AI model evaluation. Instead of ranking models against each other, it profiles individual models to identify what they need.

---

## Goodhart's Law Warning

> "When a measure becomes a target, it ceases to be a good measure."

**This framework will fail the moment it becomes a leaderboard.**

AI-SETT is diagnostic, not competitive. The moment someone says "our model scored 547/577," we've lost. The count isn't the point. The profile is the point. What behaviors are present? What's missing? What does this model need?

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

## What Makes AI-SETT Different

| Traditional benchmarks | AI-SETT |
|------------------------|---------|
| Ranks models against each other | Profiles individual model's shape |
| Single score or percentage | 577 observable criteria across 12 categories |
| Assumes ideal exists | No ceiling — just count demonstrated behaviors |
| Pass/fail mentality | Demonstrated (+1) or gap (+0) |
| Tells you "how good" | Tells you what's present and what's missing |
| One-time evaluation | Iterative: assess → train → reassess |
| Comes from ML | Grounded in instructional design |

---

## The Framework

**S**tudent — What can this model do? Where are the gaps?
**E**nvironment — What context will it operate in?
**T**asks — What must it accomplish?
**T**ools — What interventions will help?

---

## Coverage

**12 categories, 577 criteria, 953 probes**

Categories are organized into two tiers:

### Base Tier (always assessed)

| Category | Criteria |
|----------|----------|
| Understanding | 25 |
| Calibration | 30 |
| Reasoning | 48 |
| Boundaries | 40 |
| Knowledge | 120 |
| Generation | 65 |

### Optional Tier (assessed with `--tier all`)

| Category | Criteria |
|----------|----------|
| Interaction | 24 |
| Tool Use | 56 |
| Emotional Intelligence | 32 |
| Metacognition | 24 |
| Learning | 40 |
| Pedagogy | 72 |

### Knowledge Domains (15)
- Programming, APIs, Security, DevOps, Databases
- Mathematics, Science
- Business & Finance, Legal, Medical/Health
- History & Geography, Arts & Culture
- Education & Pedagogy, Project Management, Design

### Tool Use (7)
- Web search, Code execution, File handling
- API calling, Calculator/computation
- Image understanding, Tool selection

### Learning & Pedagogy
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

## How to Use

### 1. Select relevant criteria
Not all 577 criteria apply to every use case. Choose what matters.

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

## Automated Assessment

Run the assessment runner against any OpenAI-compatible API:

```bash
export OPENAI_API_KEY=not-needed

python3 -m tools.assessment_runner \
  --probes probes \
  --provider openai \
  --model your-model-name \
  --base-url http://localhost:11434/v1 \
  --temperature 0.0 \
  --concurrency 2 \
  --tier all \
  --output results/assessment.json \
  --verbose
```

Use `--tier base` for base categories only, or `--tier all` for all 12 categories.

---

## Training Data

AI-SETT includes tools for generating and managing training data:

- **3,415 synthetic training examples** across all 12 categories
- **2,711 balanced examples** (224 per category target)
- Conversation harvester for extracting training signal from real conversations
- Tag-and-sample pipeline for balanced dataset creation

See `training_data/` for the full dataset and tools.

---

## Assessment Sequence

For comprehensive evaluation:

```
1. Domain assessments (Knowledge, Reasoning, etc.)
   ↓
2. Capability assessments (Understanding, Calibration, etc.)
   ↓
3. Learning assessment (teach it something, see what sticks)
   ↓
4. Pedagogy assessment (have it teach, evaluate quality)
   ↓
5. Gap analysis and intervention planning
   ↓
6. Targeted training on identified gaps
   ↓
7. Reassessment
```

---

## Files

- `AI-SETT-FRAMEWORK.md` — Complete framework with all 577 criteria
- `README.md` — This file
- `probes/` — 953 probes across all 12 categories
- `tools/` — Assessment runner, validator, domain builder
- `training_data/` — 3,415 synthetic training examples
- `Modelfile.aisett` — Ollama Modelfile for NEMOclaude AI-SETT model
- `run_aisett_eval.sh` — Full assessment runner script

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
  author={CrewRelay},
  year={2025},
  url={https://github.com/crewrelay/AI-SETT}
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

*AI-SETT treats AI models the way good instructors treat students: meet them where they are, identify what they need, train the gaps.*
