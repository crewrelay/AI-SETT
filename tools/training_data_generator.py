#!/usr/bin/env python3
"""AI-SETT Training Data Generator

Generates fine-tuning data to close assessment gaps. Multiple LLMs contribute
training examples for diversity and quality cross-validation via Jaccard-based
consensus scoring.

Usage:
    # Generate from assessment gaps (multi-model)
    python -m tools.training_data_generator \
        --input results/gpt4o.json \
        --framework docs/AI-SETT-FRAMEWORK.md \
        --generator anthropic:claude-3-5-haiku-20241022 \
        --generator openai:gpt-4o-mini \
        --format openai_jsonl \
        --examples-per-criterion 5 \
        --output training_data/gaps.jsonl

    # Manual criterion targeting
    python -m tools.training_data_generator \
        --criteria U.BR.01,U.BR.02,C.RL.01 \
        --framework docs/AI-SETT-FRAMEWORK.md \
        --generator anthropic:claude-3-5-haiku-20241022 \
        --format raw_jsonl

    # ZPD-focused (highest-ROI gaps)
    python -m tools.training_data_generator \
        --input results/gpt4o.json --zpd-only \
        --framework docs/AI-SETT-FRAMEWORK.md \
        --generator openai:gpt-4o-mini

    # Dry run
    python -m tools.training_data_generator \
        --input results/gpt4o.json \
        --framework docs/AI-SETT-FRAMEWORK.md \
        --generator openai:gpt-4o-mini --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tools.providers import get_provider
from tools.providers.base import CompletionRequest, Message


# ---------------------------------------------------------------------------
# Category mapping (criterion ID prefix -> (category, subcategory))
# ---------------------------------------------------------------------------

CATEGORY_MAP = {
    "U.BR": ("understanding", "basic_requests"),
    "U.CR": ("understanding", "complex_requests"),
    "U.AR": ("understanding", "ambiguous_requests"),
    "U.IR": ("understanding", "implicit_requests"),
    "U.CX": ("understanding", "contextual_understanding"),
    "C.RL": ("calibration", "response_length"),
    "C.RD": ("calibration", "response_depth"),
    "C.FM": ("calibration", "format"),
    "C.TN": ("calibration", "tone"),
    "C.ST": ("calibration", "stopping"),
    "G.TW": ("generation", "technical_writing"),
    "G.CW": ("generation", "creative_writing"),
    "G.PW": ("generation", "persuasive_writing"),
    "G.AW": ("generation", "academic_writing"),
    "G.BW": ("generation", "business_writing"),
    "G.CD": ("generation", "code_generation"),
    "G.SO": ("generation", "structured_output"),
    "G.SM": ("generation", "summarization"),
    "G.TR": ("generation", "translation"),
    "K.PG": ("knowledge", "programming_fundamentals"),
    "K.AP": ("knowledge", "apis"),
    "K.SC": ("knowledge", "security"),
    "K.DO": ("knowledge", "devops"),
    "K.DT": ("knowledge", "databases"),
    "K.MT": ("knowledge", "mathematics"),
    "K.SG": ("knowledge", "science"),
    "K.BF": ("knowledge", "business_finance"),
    "K.LG": ("knowledge", "legal_concepts"),
    "K.MH": ("knowledge", "medical_health"),
    "K.HG": ("knowledge", "history_geography"),
    "K.AC": ("knowledge", "arts_culture"),
    "K.ED": ("knowledge", "education_pedagogy"),
    "K.PM": ("knowledge", "project_management"),
    "K.DS": ("knowledge", "design"),
    "R.LG": ("reasoning", "logical_reasoning"),
    "R.MT": ("reasoning", "mathematical_reasoning"),
    "R.CS": ("reasoning", "causal_reasoning"),
    "R.AN": ("reasoning", "analogical_reasoning"),
    "R.CT": ("reasoning", "critical_thinking"),
    "R.PS": ("reasoning", "problem_solving"),
    "B.RF": ("boundaries", "appropriate_refusals"),
    "B.OR": ("boundaries", "avoids_over_refusal"),
    "B.UN": ("boundaries", "uncertainty_limits"),
    "B.PB": ("boundaries", "professional_boundaries"),
    "B.SF": ("boundaries", "safety"),
    "I.MT": ("interaction", "multi_turn_coherence"),
    "I.EH": ("interaction", "error_handling"),
    "I.CR": ("interaction", "clarification_repair"),
    "T.WS": ("tool_use", "web_search"),
    "T.CE": ("tool_use", "code_execution"),
    "T.FH": ("tool_use", "file_handling"),
    "T.AC": ("tool_use", "api_calling"),
    "T.CL": ("tool_use", "calculator_computation"),
    "T.IU": ("tool_use", "image_understanding"),
    "T.TS": ("tool_use", "tool_selection"),
    "E.ER": ("emotional_intelligence", "emotional_recognition"),
    "E.EP": ("emotional_intelligence", "empathetic_response"),
    "E.DC": ("emotional_intelligence", "difficult_conversations"),
    "E.SA": ("emotional_intelligence", "social_awareness"),
    "M.SA": ("metacognition", "self_awareness"),
    "M.LA": ("metacognition", "learning_adaptation"),
    "M.SS": ("metacognition", "strategy_selection"),
    "L.IC": ("learning", "in_context_learning"),
    "L.IF": ("learning", "instruction_following"),
    "L.DM": ("learning", "domain_specific_learning"),
    "L.ER": ("learning", "error_based_learning"),
    "L.TR": ("learning", "learning_transfer"),
    "T.DA": ("teaching", "diagnostic_assessment"),
    "T.EQ": ("teaching", "explanation_quality"),
    "T.SC": ("teaching", "scaffolding"),
    "T.FB": ("teaching", "feedback"),
    "T.AD": ("teaching", "adaptation"),
    "T.MH": ("teaching", "misconception_handling"),
    "T.CH": ("teaching", "checking_understanding"),
    "T.DT": ("teaching", "domain_teaching"),
}


def criterion_category(cid: str) -> tuple[str, str]:
    """Map a criterion ID to (category_slug, subcategory_slug)."""
    prefix = cid[:4]  # e.g. "U.BR"
    return CATEGORY_MAP.get(prefix, ("uncategorized", "uncategorized"))


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GeneratorSpec:
    """Parsed generator specification."""
    provider: str
    model: str
    base_url: Optional[str] = None


@dataclass
class TrainingExample:
    """A single generated training example."""
    criterion_id: str
    behavioral_target: str
    system_prompt: str
    user_input: str
    ideal_output: str
    generator_model: str
    scenario_tag: str
    quality_score: float = 0.0
    validation_passed: Optional[bool] = None


# ---------------------------------------------------------------------------
# Generator spec parsing
# ---------------------------------------------------------------------------

ENV_KEY_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
}


def parse_generator_spec(spec: str) -> GeneratorSpec:
    """Parse 'provider:model' or 'provider:model:base_url' into a GeneratorSpec."""
    if ":" not in spec:
        raise ValueError(f"Generator spec must be 'provider:model', got '{spec}'")

    parts = spec.split(":", maxsplit=2)
    provider = parts[0]
    model = parts[1]
    base_url = parts[2] if len(parts) > 2 else None

    return GeneratorSpec(provider=provider, model=model, base_url=base_url)


def init_provider(spec: GeneratorSpec):
    """Initialize a provider from a GeneratorSpec."""
    env_var = ENV_KEY_MAP.get(spec.provider, f"{spec.provider.upper()}_API_KEY")
    api_key = os.environ.get(env_var, "")

    # Local models may not need an API key
    if not api_key and spec.base_url:
        api_key = "not-needed"

    if not api_key:
        print(f"Error: Set {env_var} environment variable for {spec.provider}", file=sys.stderr)
        sys.exit(1)

    return get_provider(spec.provider, api_key=api_key, base_url=spec.base_url)


# ---------------------------------------------------------------------------
# Framework parsing (reuses probe_filler pattern)
# ---------------------------------------------------------------------------

def parse_framework(framework_path: str) -> dict[str, dict]:
    """Parse AI-SETT-FRAMEWORK.md and extract criterion definitions.

    Returns: {criterion_id: {"criterion": str, "verify": str, "section": str}}
    """
    path = Path(framework_path)
    if not path.is_file():
        print(f"Error: Framework file not found: {framework_path}", file=sys.stderr)
        sys.exit(1)

    criteria: dict[str, dict] = {}
    current_section = ""

    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("##"):
                current_section = line.lstrip("#").strip()
            m = re.match(
                r'^\|\s*([A-Z]\.[A-Z]{2}\.\d{2})\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|',
                line,
            )
            if m:
                cid = m.group(1)
                criteria[cid] = {
                    "criterion": m.group(2).strip(),
                    "verify": m.group(3).strip(),
                    "section": current_section,
                }

    return criteria


# ---------------------------------------------------------------------------
# Gap extraction from assessment results
# ---------------------------------------------------------------------------

def extract_gap_criteria(result: dict, zpd_only: bool = False) -> list[str]:
    """Extract criterion IDs where demonstrated == false from an assessment result.

    If zpd_only, only include criteria from ZPD subcategories (gap_ratio 0.1-0.6).
    """
    profile = result.get("profile", {})
    gap_ids: list[str] = []

    for cat_name, cat in profile.items():
        for sub_name, sub in cat.get("subcategories", {}).items():
            total = sub.get("total", 0)
            demonstrated = sub.get("demonstrated", 0)
            gap_count = total - demonstrated

            if total == 0 or gap_count == 0:
                continue

            gap_ratio = gap_count / total

            if zpd_only:
                # ZPD: partial mastery, at least 1 demonstrated, ratio 0.1-0.6
                if demonstrated == 0 or gap_ratio < 0.1 or gap_ratio > 0.6:
                    continue

            for cid, demo in sub.get("criteria", {}).items():
                if not demo:
                    gap_ids.append(cid)

    return gap_ids


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a training data generator for AI language model fine-tuning.

You generate high-quality training examples that teach a model to demonstrate specific behavioral criteria. Each example consists of a system prompt, user input, and an ideal assistant output.

Rules:
- Each example should be a realistic, natural interaction (not a test instruction).
- The ideal output must clearly demonstrate the target criterion.
- Use concrete scenarios: real-ish names, numbers, programming languages, topics.
- Vary the scenarios across examples — different domains, difficulty levels, contexts.
- The system prompt should be brief and natural (e.g., "You are a helpful assistant." or task-specific).
- Keep user inputs concise (1-3 sentences typically).
- Ideal outputs should be the length appropriate to the task — not padded, not truncated.
- Include a short scenario_tag (2-4 words) describing each example's scenario.

Return a JSON array of objects, each with these fields:
- "system_prompt": string
- "user_input": string
- "ideal_output": string
- "scenario_tag": string (2-4 word scenario description)

Return ONLY the JSON array. No markdown fences, no explanation."""


def build_generation_prompt(
    criterion_id: str,
    criterion_info: dict,
    num_examples: int,
) -> str:
    """Build a user prompt requesting training examples for a criterion."""
    return (
        f"Generate {num_examples} diverse training examples for this criterion:\n\n"
        f"Criterion ID: {criterion_id}\n"
        f"Section: {criterion_info.get('section', 'Unknown')}\n"
        f"Behavioral target: {criterion_info.get('criterion', 'Unknown')}\n"
        f"How to verify: {criterion_info.get('verify', 'Unknown')}\n\n"
        f"Each example should clearly teach a model to demonstrate this behavior. "
        f"Vary the scenarios — different topics, contexts, and difficulty levels."
    )


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def extract_json_array(text: str) -> list[dict]:
    """Extract a JSON array from text, handling markdown fences and wrapper text."""
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

    # Try direct parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Search for [...] pattern (for models that wrap in explanatory text)
    bracket_match = re.search(r'\[[\s\S]*\]', text)
    if bracket_match:
        try:
            parsed = json.loads(bracket_match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    return []


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_examples(
    criterion_id: str,
    criterion_info: dict,
    spec: GeneratorSpec,
    provider,
    num_examples: int,
    temperature: float,
    verbose: bool = False,
) -> list[TrainingExample]:
    """Generate training examples for one criterion using one generator."""
    user_prompt = build_generation_prompt(criterion_id, criterion_info, num_examples)

    request = CompletionRequest(
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ],
        model=spec.model,
        temperature=temperature,
        max_tokens=4096,
    )

    try:
        response = provider.complete(request)
    except Exception as e:
        print(f"  Error [{spec.provider}:{spec.model}] for {criterion_id}: {e}", file=sys.stderr)
        return []

    raw_examples = extract_json_array(response.content)
    if not raw_examples:
        print(f"  No valid JSON from [{spec.provider}:{spec.model}] for {criterion_id}", file=sys.stderr)
        if verbose:
            print(f"    Raw response: {response.content[:200]}...", file=sys.stderr)
        return []

    examples = []
    for raw in raw_examples:
        if not isinstance(raw, dict):
            continue
        user_input = raw.get("user_input", "").strip()
        ideal_output = raw.get("ideal_output", "").strip()
        if not user_input or not ideal_output:
            continue

        examples.append(TrainingExample(
            criterion_id=criterion_id,
            behavioral_target=criterion_info.get("criterion", ""),
            system_prompt=raw.get("system_prompt", "You are a helpful assistant.").strip(),
            user_input=user_input,
            ideal_output=ideal_output,
            generator_model=f"{spec.provider}:{spec.model}",
            scenario_tag=raw.get("scenario_tag", "").strip(),
        ))

    return examples


# ---------------------------------------------------------------------------
# Deduplication & quality scoring
# ---------------------------------------------------------------------------

def jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two strings (word-level)."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def dedup_examples(
    examples: list[TrainingExample],
    threshold: float,
) -> list[TrainingExample]:
    """Remove near-duplicate examples within the same criterion group.

    Uses Jaccard similarity on user_input + ideal_output.
    O(n^2) per criterion but n is small (~15 max).
    """
    if not examples:
        return []

    kept: list[TrainingExample] = []
    for ex in examples:
        ex_text = ex.user_input + " " + ex.ideal_output
        is_dup = False
        for existing in kept:
            existing_text = existing.user_input + " " + existing.ideal_output
            if jaccard_similarity(ex_text, existing_text) > threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(ex)

    return kept


def score_cross_model_agreement(
    examples: list[TrainingExample],
    threshold: float,
    num_generators: int,
) -> list[TrainingExample]:
    """Score examples by cross-model agreement.

    When multiple generators produce similar examples for the same criterion,
    those get higher quality scores:
    - 1.0: All generators produced similar example (consensus)
    - 0.5-0.9: Some generators agree
    - 0.0: Unique to one generator (still kept, lower confidence)
    """
    if num_generators <= 1:
        for ex in examples:
            ex.quality_score = 0.5  # No cross-validation possible
        return examples

    # Group by criterion
    by_criterion: dict[str, list[TrainingExample]] = {}
    for ex in examples:
        by_criterion.setdefault(ex.criterion_id, []).append(ex)

    for cid, group in by_criterion.items():
        for ex in group:
            # Count how many different generators produced a similar example
            similar_generators: set[str] = {ex.generator_model}
            ex_text = ex.user_input + " " + ex.ideal_output
            for other in group:
                if other is ex:
                    continue
                if other.generator_model == ex.generator_model:
                    continue
                other_text = other.user_input + " " + other.ideal_output
                if jaccard_similarity(ex_text, other_text) > threshold:
                    similar_generators.add(other.generator_model)

            # Score based on fraction of generators that agree
            ex.quality_score = round(len(similar_generators) / num_generators, 2)

    return examples


# ---------------------------------------------------------------------------
# Validation (optional)
# ---------------------------------------------------------------------------

def validate_examples(examples: list[TrainingExample]) -> list[TrainingExample]:
    """Run rule-based validation on generated ideal_output using rule_evaluator.

    Only validates deterministic criteria (contains, word count, regex).
    Heuristic criteria are skipped.
    """
    try:
        from tools.evaluators.rule_evaluator import evaluate_criterion
    except ImportError:
        print("Warning: rule_evaluator not available, skipping validation", file=sys.stderr)
        return examples

    # We need probe-style criterion dicts with 'check' and 'pass' fields.
    # Since we don't have those for generated examples, we do a lightweight
    # check: verify the ideal_output is non-empty and reasonably sized.
    for ex in examples:
        # Basic sanity checks
        if len(ex.ideal_output.split()) < 2:
            ex.validation_passed = False
        elif len(ex.ideal_output.split()) > 2000:
            ex.validation_passed = False
        elif ex.user_input.strip().lower() == ex.ideal_output.strip().lower():
            ex.validation_passed = False  # Parrot response
        else:
            ex.validation_passed = True

    return examples


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_openai_jsonl(ex: TrainingExample) -> dict:
    """Format as OpenAI fine-tuning JSONL."""
    return {
        "messages": [
            {"role": "system", "content": ex.system_prompt},
            {"role": "user", "content": ex.user_input},
            {"role": "assistant", "content": ex.ideal_output},
        ]
    }


def format_anthropic_jsonl(ex: TrainingExample) -> dict:
    """Format as Anthropic fine-tuning JSONL."""
    return {
        "system": ex.system_prompt,
        "messages": [
            {"role": "user", "content": ex.user_input},
            {"role": "assistant", "content": ex.ideal_output},
        ]
    }


def format_huggingface_jsonl(ex: TrainingExample) -> dict:
    """Format as HuggingFace instruction-tuning JSONL."""
    return {
        "instruction": ex.user_input,
        "input": "",
        "output": ex.ideal_output,
        "system": ex.system_prompt,
        "metadata": {
            "criterion_id": ex.criterion_id,
            "generator_model": ex.generator_model,
            "quality_score": ex.quality_score,
        }
    }


def format_raw_jsonl(ex: TrainingExample) -> dict:
    """Format as raw JSONL with all metadata."""
    result = {
        "criterion_id": ex.criterion_id,
        "behavioral_target": ex.behavioral_target,
        "system_prompt": ex.system_prompt,
        "user_input": ex.user_input,
        "ideal_output": ex.ideal_output,
        "generator_model": ex.generator_model,
        "scenario_tag": ex.scenario_tag,
        "quality_score": ex.quality_score,
    }
    if ex.validation_passed is not None:
        result["validation_passed"] = ex.validation_passed
    return result


FORMATTERS = {
    "openai_jsonl": format_openai_jsonl,
    "anthropic_jsonl": format_anthropic_jsonl,
    "huggingface_jsonl": format_huggingface_jsonl,
    "raw_jsonl": format_raw_jsonl,
}


# ---------------------------------------------------------------------------
# Split output (granular directory structure)
# ---------------------------------------------------------------------------

def write_split_output(
    examples: list[TrainingExample],
    output_dir: str,
    output_format: str,
    formatter,
    append: bool = False,
) -> dict:
    """Write examples split by category/subcategory/criterion.

    Directory structure:
        output_dir/
        ├── understanding/
        │   ├── basic_requests/
        │   │   ├── U.BR.01.jsonl
        │   │   ├── U.BR.02.jsonl
        │   │   └── _combined.jsonl
        │   └── _combined.jsonl
        ├── calibration/
        │   └── ...
        └── manifest.json

    Returns manifest data for write_manifest().
    """
    base = Path(output_dir)
    mode = "a" if append else "w"

    # Group examples by category -> subcategory -> criterion
    tree: dict[str, dict[str, dict[str, list[TrainingExample]]]] = {}
    for ex in examples:
        cat, subcat = criterion_category(ex.criterion_id)
        tree.setdefault(cat, {}).setdefault(subcat, {}).setdefault(ex.criterion_id, []).append(ex)

    manifest_data: dict[str, dict] = {}
    files_written = 0

    for cat, subcats in sorted(tree.items()):
        cat_dir = base / cat
        cat_examples: list[TrainingExample] = []

        for subcat, criteria in sorted(subcats.items()):
            subcat_dir = cat_dir / subcat
            subcat_dir.mkdir(parents=True, exist_ok=True)
            subcat_examples: list[TrainingExample] = []

            for cid, cid_examples in sorted(criteria.items()):
                # Per-criterion file
                cid_path = subcat_dir / f"{cid}.jsonl"
                _write_jsonl(cid_path, cid_examples, formatter, mode)
                files_written += 1

                # Track for manifest
                existing_count = _count_lines(cid_path) if append else len(cid_examples)
                manifest_data[cid] = {
                    "category": cat,
                    "subcategory": subcat,
                    "file": str(cid_path.relative_to(base)),
                    "example_count": existing_count,
                    "generators": sorted(set(ex.generator_model for ex in cid_examples)),
                    "avg_quality": round(
                        sum(ex.quality_score for ex in cid_examples) / len(cid_examples), 2
                    ),
                }

                subcat_examples.extend(cid_examples)

            # Per-subcategory combined file
            subcat_combined = subcat_dir / "_combined.jsonl"
            _write_jsonl(subcat_combined, subcat_examples, formatter, mode)
            files_written += 1
            cat_examples.extend(subcat_examples)

        # Per-category combined file
        cat_dir.mkdir(parents=True, exist_ok=True)
        cat_combined = cat_dir / "_combined.jsonl"
        _write_jsonl(cat_combined, cat_examples, formatter, mode)
        files_written += 1

    return manifest_data


def _write_jsonl(path: Path, examples: list[TrainingExample], formatter, mode: str):
    """Write examples to a JSONL file."""
    with open(path, mode) as f:
        for ex in examples:
            line = json.dumps(formatter(ex), ensure_ascii=False)
            f.write(line + "\n")


def _count_lines(path: Path) -> int:
    """Count lines in a file (for manifest after append)."""
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def write_manifest(
    manifest_data: dict[str, dict],
    output_dir: str,
    output_format: str,
    generator_specs: list,
):
    """Write manifest.json summarizing the training data library."""
    base = Path(output_dir)

    # Aggregate stats
    categories: dict[str, dict] = {}
    for cid, info in manifest_data.items():
        cat = info["category"]
        subcat = info["subcategory"]
        if cat not in categories:
            categories[cat] = {"subcategories": {}, "total_examples": 0, "criteria_count": 0}
        categories[cat]["total_examples"] += info["example_count"]
        categories[cat]["criteria_count"] += 1
        if subcat not in categories[cat]["subcategories"]:
            categories[cat]["subcategories"][subcat] = {"total_examples": 0, "criteria_count": 0}
        categories[cat]["subcategories"][subcat]["total_examples"] += info["example_count"]
        categories[cat]["subcategories"][subcat]["criteria_count"] += 1

    manifest = {
        "version": "1.0.0",
        "format": output_format,
        "total_examples": sum(info["example_count"] for info in manifest_data.values()),
        "total_criteria": len(manifest_data),
        "total_categories": len(categories),
        "generators_used": sorted(set(
            g for info in manifest_data.values() for g in info["generators"]
        )),
        "categories": categories,
        "criteria": manifest_data,
    }

    manifest_path = base / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def generate_training_data(
    criteria_ids: list[str],
    criteria_db: dict[str, dict],
    generator_specs: list[GeneratorSpec],
    output_format: str,
    output_path: str,
    examples_per_criterion: int = 3,
    concurrency: int = 2,
    temperature: float = 0.7,
    dedup_threshold: float = 0.7,
    do_validate: bool = False,
    split_by_category: bool = False,
    append: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
):
    """Main generation logic."""
    # Resolve criteria against framework
    resolved = []
    missing = []
    for cid in criteria_ids:
        if cid in criteria_db:
            resolved.append(cid)
        else:
            missing.append(cid)

    if missing:
        print(f"Warning: {len(missing)} criteria not found in framework: {', '.join(missing[:10])}", file=sys.stderr)

    if not resolved:
        print("Error: No valid criteria to generate for", file=sys.stderr)
        sys.exit(1)

    total_calls = len(resolved) * len(generator_specs)
    total_examples = len(resolved) * len(generator_specs) * examples_per_criterion

    print(f"Criteria: {len(resolved)}")
    print(f"Generators: {len(generator_specs)}")
    print(f"Examples per criterion per generator: {examples_per_criterion}")
    print(f"API calls planned: {total_calls}")
    print(f"Max examples: {total_examples}")

    if dry_run:
        print("\n--- Dry run: no API calls ---")
        if split_by_category:
            # Group by category/subcategory for preview
            grouped: dict[str, dict[str, list[str]]] = {}
            for cid in resolved:
                cat, subcat = criterion_category(cid)
                grouped.setdefault(cat, {}).setdefault(subcat, []).append(cid)
            for cat in sorted(grouped):
                print(f"\n  {cat}/")
                for subcat in sorted(grouped[cat]):
                    cids = grouped[cat][subcat]
                    print(f"    {subcat}/")
                    for cid in cids:
                        info = criteria_db[cid]
                        print(f"      {cid}.jsonl  — {info['criterion']}")
            if append:
                print(f"\nMode: append (adding to existing library)")
            else:
                print(f"\nMode: overwrite")
        else:
            for cid in resolved:
                info = criteria_db[cid]
                print(f"  {cid}: {info['criterion']}")
                print(f"    Section: {info['section']}")
                print(f"    Verify: {info['verify']}")
        print(f"\nGenerators:")
        for spec in generator_specs:
            url_info = f" @ {spec.base_url}" if spec.base_url else ""
            print(f"  {spec.provider}:{spec.model}{url_info}")
        print(f"\nOutput: {output_path} ({output_format})")
        return

    # Initialize providers
    providers: dict[str, tuple[GeneratorSpec, object]] = {}
    for spec in generator_specs:
        key = f"{spec.provider}:{spec.model}"
        provider = init_provider(spec)
        providers[key] = (spec, provider)
        if verbose:
            print(f"Initialized: {key}")

    # Generate examples — one API call per (criterion, generator)
    all_examples: list[TrainingExample] = []
    completed = 0

    def _generate_task(cid: str, spec: GeneratorSpec, provider) -> list[TrainingExample]:
        return generate_examples(
            criterion_id=cid,
            criterion_info=criteria_db[cid],
            spec=spec,
            provider=provider,
            num_examples=examples_per_criterion,
            temperature=temperature,
            verbose=verbose,
        )

    tasks = []
    for cid in resolved:
        for key, (spec, provider) in providers.items():
            tasks.append((cid, spec, provider))

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_generate_task, cid, spec, prov): (cid, spec)
                for cid, spec, prov in tasks
            }
            for future in as_completed(futures):
                cid, spec = futures[future]
                try:
                    examples = future.result()
                    all_examples.extend(examples)
                    completed += 1
                    print(f"  [{completed}/{total_calls}] {spec.provider}:{spec.model} x {cid}: {len(examples)} examples")
                except Exception as e:
                    completed += 1
                    print(f"  [{completed}/{total_calls}] {spec.provider}:{spec.model} x {cid}: ERROR {e}", file=sys.stderr)
    else:
        for cid, spec, prov in tasks:
            examples = _generate_task(cid, spec, prov)
            all_examples.extend(examples)
            completed += 1
            print(f"  [{completed}/{total_calls}] {spec.provider}:{spec.model} x {cid}: {len(examples)} examples")

    print(f"\nGenerated {len(all_examples)} raw examples")

    # Deduplication (per criterion)
    by_criterion: dict[str, list[TrainingExample]] = {}
    for ex in all_examples:
        by_criterion.setdefault(ex.criterion_id, []).append(ex)

    deduped: list[TrainingExample] = []
    for cid, group in by_criterion.items():
        kept = dedup_examples(group, dedup_threshold)
        deduped.extend(kept)

    removed = len(all_examples) - len(deduped)
    if removed > 0:
        print(f"Deduplication removed {removed} near-duplicates (threshold: {dedup_threshold})")
    all_examples = deduped

    # Cross-model quality scoring
    all_examples = score_cross_model_agreement(
        all_examples, dedup_threshold, len(generator_specs)
    )

    # Optional validation
    if do_validate:
        all_examples = validate_examples(all_examples)
        failed = sum(1 for ex in all_examples if ex.validation_passed is False)
        print(f"Validation: {len(all_examples) - failed} passed, {failed} failed")

    # Format and write output
    formatter = FORMATTERS[output_format]

    if split_by_category:
        # Granular directory structure
        output_dir = output_path  # Treated as directory
        manifest_data = write_split_output(
            all_examples, output_dir, output_format, formatter, append=append
        )
        manifest_path = write_manifest(manifest_data, output_dir, output_format, generator_specs)
        print(f"\nWrote {len(all_examples)} examples across {len(manifest_data)} criteria")
        print(f"Library: {output_dir}/")
        print(f"Manifest: {manifest_path}")

        # Print category summary
        cats: dict[str, int] = {}
        for info in manifest_data.values():
            cats[info["category"]] = cats.get(info["category"], 0) + info["example_count"]
        for cat in sorted(cats):
            print(f"  {cat}: {cats[cat]} examples")
    else:
        # Single flat file
        output_dir_path = Path(output_path).parent
        output_dir_path.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"
        with open(output_path, mode) as f:
            for ex in all_examples:
                line = json.dumps(formatter(ex), ensure_ascii=False)
                f.write(line + "\n")

        print(f"\nWrote {len(all_examples)} examples to {output_path} ({output_format})")

    # Summary by criterion
    if verbose:
        print("\n--- Per-criterion summary ---")
        for cid in sorted(by_criterion):
            criterion_examples = [ex for ex in all_examples if ex.criterion_id == cid]
            avg_score = sum(ex.quality_score for ex in criterion_examples) / len(criterion_examples) if criterion_examples else 0
            print(f"  {cid}: {len(criterion_examples)} examples, avg quality: {avg_score:.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI-SETT Training Data Generator — generate fine-tuning data from assessment gaps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from assessment gaps (multi-model)
  %(prog)s --input results/gpt4o.json \\
    --framework docs/AI-SETT-FRAMEWORK.md \\
    --generator anthropic:claude-3-5-haiku-20241022 \\
    --generator openai:gpt-4o-mini \\
    --format openai_jsonl --output training_data/gaps.jsonl

  # Manual criterion targeting
  %(prog)s --criteria U.BR.01,U.BR.02,C.RL.01 \\
    --framework docs/AI-SETT-FRAMEWORK.md \\
    --generator anthropic:claude-3-5-haiku-20241022

  # ZPD-focused (highest-ROI gaps)
  %(prog)s --input results/gpt4o.json --zpd-only \\
    --framework docs/AI-SETT-FRAMEWORK.md \\
    --generator openai:gpt-4o-mini

  # Dry run
  %(prog)s --input results/gpt4o.json \\
    --framework docs/AI-SETT-FRAMEWORK.md \\
    --generator openai:gpt-4o-mini --dry-run

  # Build a reusable training data library (granular)
  %(prog)s --input results/gpt4o.json \\
    --framework docs/AI-SETT-FRAMEWORK.md \\
    --generator anthropic:claude-3-5-haiku-20241022 \\
    --generator openai:gpt-4o-mini \\
    --split-by-category --output training_data/

  # Add more data to existing library (append)
  %(prog)s --input results/llama4.json \\
    --framework docs/AI-SETT-FRAMEWORK.md \\
    --generator openai:gpt-4o-mini \\
    --split-by-category --append --output training_data/

Generator spec format:
  provider:model                    — use default API endpoint
  provider:model:base_url           — custom endpoint (Ollama, vLLM, Groq)

  Examples:
    anthropic:claude-3-5-haiku-20241022
    openai:gpt-4o-mini
    openai:llama3:http://localhost:11434/v1
    openai:mixtral:http://localhost:8000/v1
    openai:llama3:https://api.groq.com/openai/v1
        """,
    )

    # Input sources (mutually exclusive-ish: --input or --criteria)
    parser.add_argument("--input", help="Assessment result JSON (gap source)")
    parser.add_argument("--criteria", help="Comma-separated criterion IDs (alternative to --input)")
    parser.add_argument("--framework", required=True,
                        help="Path to AI-SETT-FRAMEWORK.md (for criterion definitions)")

    # Generators
    parser.add_argument("--generator", action="append", dest="generators",
                        help="Generator spec as 'provider:model' or 'provider:model:base_url' (repeatable)")

    # Output
    parser.add_argument("--format", default="raw_jsonl", choices=list(FORMATTERS),
                        help="Output format (default: raw_jsonl)")
    parser.add_argument("--output", default="training_data.jsonl",
                        help="Output file path (or directory when using --split-by-category)")

    # Generation parameters
    parser.add_argument("--examples-per-criterion", type=int, default=3,
                        help="Examples each generator produces per criterion (default: 3)")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="Parallel API calls (default: 2)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature (default: 0.7)")
    parser.add_argument("--dedup-threshold", type=float, default=0.7,
                        help="Jaccard similarity threshold for dedup (default: 0.7)")

    # Filters
    parser.add_argument("--zpd-only", action="store_true",
                        help="Only target ZPD candidates (gap ratio 0.1-0.6)")

    # Output structure
    parser.add_argument("--split-by-category", action="store_true",
                        help="Write to category/subcategory/criterion.jsonl directory structure")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing files instead of overwriting (builds library over time)")

    # Optional steps
    parser.add_argument("--validate", action="store_true",
                        help="Run rule-based validation on generated outputs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan, no API calls")
    parser.add_argument("--verbose", action="store_true", help="Detailed output")

    args = parser.parse_args()

    # Validate input source
    if not args.input and not args.criteria:
        parser.error("Either --input (assessment result) or --criteria (comma-separated IDs) is required")

    if not args.dry_run and not args.generators:
        parser.error("--generator is required (unless using --dry-run)")

    # Parse framework
    criteria_db = parse_framework(args.framework)
    print(f"Parsed {len(criteria_db)} criteria from framework")

    # Determine target criteria
    if args.input:
        with open(args.input) as f:
            result = json.load(f)
        criteria_ids = extract_gap_criteria(result, zpd_only=args.zpd_only)
        source = "ZPD gaps" if args.zpd_only else "all gaps"
        print(f"Extracted {len(criteria_ids)} {source} from {args.input}")
    else:
        criteria_ids = [c.strip() for c in args.criteria.split(",") if c.strip()]
        print(f"Manual criteria: {len(criteria_ids)}")

    if not criteria_ids:
        print("No criteria to generate for. Exiting.")
        return

    # Parse generator specs
    generator_specs = []
    if args.generators:
        for spec_str in args.generators:
            generator_specs.append(parse_generator_spec(spec_str))

    # For dry run, allow empty generators
    if args.dry_run and not generator_specs:
        generator_specs = [GeneratorSpec(provider="(none)", model="(dry-run)")]

    # Default output for split mode
    output = args.output
    if args.split_by_category and output == "training_data.jsonl":
        output = "training_data"  # Use as directory name

    generate_training_data(
        criteria_ids=criteria_ids,
        criteria_db=criteria_db,
        generator_specs=generator_specs,
        output_format=args.format,
        output_path=output,
        examples_per_criterion=args.examples_per_criterion,
        concurrency=args.concurrency,
        temperature=args.temperature,
        dedup_threshold=args.dedup_threshold,
        do_validate=args.validate,
        split_by_category=args.split_by_category,
        append=args.append,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
