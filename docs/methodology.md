# Methodology: Instructional Design Foundations

AI-SETT is grounded in established instructional design frameworks, not ML benchmarking conventions. This document explains the theoretical foundations.

## SETT Framework (Zabala, 1995)

The SETT framework was developed by Joy Zabala for assistive technology assessment in special education. It considers four elements:

- **Student** -- What can the student do? What are the gaps?
- **Environment** -- What context will the student operate in?
- **Tasks** -- What must the student accomplish?
- **Tools** -- What interventions (assistive technologies) will help?

### Application to AI models

AI-SETT adapts SETT by treating AI models as learners:

| SETT Element | Traditional | AI-SETT |
|--------------|-------------|---------|
| Student | Student with disabilities | AI model with gaps |
| Environment | Classroom, home, community | Production context, API, chat |
| Tasks | Academic, functional, social | Generation, reasoning, tool use |
| Tools | Assistive technology | Fine-tuning, prompting, scaffolds |

The key insight: instead of asking "how good is this model?" we ask "what does this model need?" -- the same shift that transformed special education from deficit-based to needs-based assessment.

## Cognitive Load Theory (Sweller, 1988)

Cognitive Load Theory (CLT) describes how working memory limits affect learning. Three types of load:

- **Intrinsic** -- Complexity inherent to the material
- **Extraneous** -- Complexity from poor instruction
- **Germane** -- Effort spent building schemas

### Application to AI assessment

CLT informs how we sequence AI-SETT assessments:

1. **Start with lower intrinsic load** -- Basic understanding before complex reasoning
2. **Reduce extraneous load** -- Clear, unambiguous probes
3. **Build on prior assessment** -- Knowledge domains before cross-domain transfer
4. **Scaffold complexity** -- Simple probes before multi-step ones

The assessment sequence in AI-SETT (Understanding -> Calibration -> Generation -> Knowledge -> Reasoning -> etc.) follows CLT principles: establish foundational capabilities before assessing complex ones.

## Zone of Proximal Development (Vygotsky, 1978)

ZPD describes the space between what a learner can do independently and what they can do with assistance. Three zones:

- **Can do independently** -- Demonstrated capabilities
- **ZPD** -- Can do with scaffolding (prompting, few-shot, etc.)
- **Cannot do yet** -- Even with scaffolding

### Application to AI models

AI-SETT's +1/+0 scoring maps directly to ZPD:

| AI-SETT Result | ZPD Zone | Implication |
|----------------|----------|-------------|
| +1 (demonstrated) | Independent | Model has this capability |
| +0 with scaffolding = +1 | ZPD | Model can do this with help |
| +0 even with scaffolding | Cannot yet | Needs training, not prompting |

This distinction matters for intervention planning:
- **Independent behaviors** -- No intervention needed
- **ZPD behaviors** -- Use scaffolds (system prompts, few-shot, skills)
- **Cannot yet** -- Requires training data or architectural changes

## Criterion-Referenced Assessment

Unlike norm-referenced assessment (comparing to other students/models), criterion-referenced assessment compares performance to defined standards.

### Key principles applied in AI-SETT

1. **Observable** -- Every criterion is something you can point to in the response
2. **Repeatable** -- Two evaluators, same response, same result
3. **Criterion-referenced** -- Compared to what SHOULD happen, not to other models
4. **Diagnostic** -- Purpose is to identify needs, not to rank

This means:
- A model isn't "bad" for scoring low -- it has identified needs
- A model isn't "good" for scoring high -- it has demonstrated behaviors
- Two models with the same count may have completely different profiles
- The +0 list is more valuable than the +1 count

## Why not ML benchmarking?

Traditional ML benchmarks (MMLU, HumanEval, GSM8K, etc.) come from a different tradition:

| ML Benchmarks | AI-SETT |
|---------------|---------|
| Ranks models | Profiles models |
| Single score | 600 observable criteria |
| Norm-referenced | Criterion-referenced |
| Pass/fail | Demonstrated/not yet |
| Competitive | Diagnostic |
| Static | Extensible |
| From ML research | From instructional design |

Both have value. ML benchmarks are useful for broad comparisons. AI-SETT is useful for identifying specific needs and planning interventions.

## Further reading

- Zabala, J. (1995). The SETT Framework: Critical areas to address in educational assistive technology.
- Sweller, J. (1988). Cognitive load during problem solving: Effects on learning. Cognitive Science, 12(2), 257-285.
- Vygotsky, L. S. (1978). Mind in Society: The Development of Higher Psychological Processes.
- Popham, W. J. (1978). Criterion-Referenced Measurement. Prentice-Hall.
