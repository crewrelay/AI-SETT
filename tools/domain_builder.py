#!/usr/bin/env python3
"""AI-SETT Domain Builder

Interactive CLI that walks a contributor through creating a new domain —
from idea to criteria to probes to training data. Uses an LLM to interview
the contributor and generate everything.

Usage:
    # Interactive mode — AI asks questions, generates everything
    python -m tools.domain_builder \
        --provider anthropic --model claude-3-5-sonnet-20241022

    # Start with a topic
    python -m tools.domain_builder \
        --provider anthropic --model claude-3-5-sonnet-20241022 \
        --topic "cybersecurity incident response"

    # Generate into existing category
    python -m tools.domain_builder \
        --provider anthropic --model claude-3-5-sonnet-20241022 \
        --category knowledge --topic "veterinary medicine"

    # Dry run — generate criteria only, no files written
    python -m tools.domain_builder \
        --provider anthropic --model claude-3-5-sonnet-20241022 \
        --topic "test domain" --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from tools.providers import get_provider
from tools.providers.base import CompletionRequest, Message
from tools.training_data_generator import (
    CATEGORY_MAP,
    GeneratorSpec,
    TrainingExample,
    format_raw_jsonl,
    generate_examples,
    init_provider,
    parse_framework,
    parse_generator_spec,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES = [
    "understanding", "calibration", "generation", "knowledge",
    "reasoning", "boundaries", "interaction", "tool_use",
    "emotional_intelligence", "metacognition", "learning", "teaching",
]

CATEGORY_PREFIXES = {
    "understanding": "U", "calibration": "C", "generation": "G",
    "knowledge": "K", "reasoning": "R", "boundaries": "B",
    "interaction": "I", "tool_use": "T", "emotional_intelligence": "E",
    "metacognition": "M", "learning": "L", "teaching": "T",
}

ENV_KEY_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
}


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def ask_llm(provider, model: str, system: str, user: str, temperature: float = 0.7) -> str:
    """Send a prompt to the LLM and return the response text."""
    request = CompletionRequest(
        messages=[
            Message(role="system", content=system),
            Message(role="user", content=user),
        ],
        model=model,
        temperature=temperature,
        max_tokens=4096,
    )
    response = provider.complete(request)
    return response.content.strip()


def extract_json_from_response(text: str) -> list | dict:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array or object
        for pattern in [r'\[[\s\S]*\]', r'\{[\s\S]*\}']:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue
    return []


# ---------------------------------------------------------------------------
# Interactive interview
# ---------------------------------------------------------------------------

def interview_contributor(topic: Optional[str] = None, category: Optional[str] = None) -> dict:
    """Ask the contributor scoping questions. Returns interview answers."""
    answers = {}

    if not topic:
        topic = input("\nWhat domain would you like to add?\n> ").strip()
    answers["topic"] = topic
    print(f"\nDomain: {topic}")

    if not category:
        print(f"\nWhich category does this belong to?")
        for i, cat in enumerate(CATEGORIES, 1):
            print(f"  ({i}) {cat}")
        choice = input("> ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(CATEGORIES):
                category = CATEGORIES[idx]
            else:
                category = "knowledge"
        except ValueError:
            # Accept category name directly
            if choice.lower() in CATEGORIES:
                category = choice.lower()
            else:
                category = "knowledge"
    answers["category"] = category
    print(f"Category: {category}")

    print(f"\nI'll ask a few questions to scope this properly.\n")

    # Expertise level
    print("1. What level of expertise should the model demonstrate?")
    print("   (a) General awareness")
    print("   (b) Practitioner knowledge")
    print("   (c) Expert knowledge")
    level = input("> ").strip().lower()
    level_map = {"a": "general_awareness", "b": "practitioner", "c": "expert"}
    answers["expertise_level"] = level_map.get(level, "practitioner")

    # Sub-areas
    print("\n2. Which sub-areas matter most? (comma-separated)")
    sub_areas = input("> ").strip()
    answers["sub_areas"] = [s.strip() for s in sub_areas.split(",") if s.strip()]

    # Boundaries
    print("\n3. What should the model NOT do?")
    print("   (e.g., diagnose specific conditions, prescribe treatments)")
    boundaries = input("> ").strip()
    answers["boundaries"] = boundaries

    # Scenarios
    print("\n4. Any specific scenarios you care about?")
    scenarios = input("> ").strip()
    answers["scenarios"] = scenarios

    return answers


# ---------------------------------------------------------------------------
# Criteria generation
# ---------------------------------------------------------------------------

CRITERIA_SYSTEM_PROMPT = """You are an expert in AI evaluation framework design. You create behavioral criteria for assessing language model capabilities.

Rules for criteria:
- Each criterion must be OBSERVABLE — you can point to it in a response
- Each criterion must be BINARY — demonstrated or not demonstrated
- Each criterion must be REPEATABLE — two evaluators get the same result
- Criteria must not overlap significantly with each other
- Each criterion needs a clear "How to verify" instruction

Return a JSON array of objects with:
- "id": criterion ID in format {PREFIX}.{SUBCAT}.{NN} (e.g., K.VM.01)
- "criterion": the behavioral target (clear, specific)
- "verify": how to verify it (what to look for in the response)"""


def generate_criteria(
    provider,
    model: str,
    answers: dict,
    framework: dict,
    prefix: str,
    subcategory_code: str,
) -> list[dict]:
    """Generate criteria for the new domain using LLM."""
    # Build existing criteria context
    existing = []
    for cid, info in framework.items():
        if cid.startswith(prefix + "."):
            existing.append(f"  {cid}: {info['criterion']} — {info['verify']}")

    existing_text = "\n".join(existing[:20]) if existing else "  (none yet)"

    user_prompt = (
        f"Generate 6-10 behavioral criteria for this new domain:\n\n"
        f"Domain: {answers['topic']}\n"
        f"Category: {answers['category']}\n"
        f"Expertise level: {answers.get('expertise_level', 'practitioner')}\n"
        f"Key sub-areas: {', '.join(answers.get('sub_areas', []))}\n"
        f"Boundaries (what NOT to do): {answers.get('boundaries', 'none specified')}\n"
        f"Important scenarios: {answers.get('scenarios', 'none specified')}\n\n"
        f"Use the ID format: {prefix}.{subcategory_code}.NN (01-10)\n\n"
        f"Existing criteria in this category (don't overlap):\n{existing_text}\n\n"
        f"Return ONLY a JSON array."
    )

    response = ask_llm(provider, model, CRITERIA_SYSTEM_PROMPT, user_prompt, temperature=0.7)
    criteria = extract_json_from_response(response)

    if not isinstance(criteria, list):
        return []

    # Validate and clean
    valid = []
    for c in criteria:
        if isinstance(c, dict) and "id" in c and "criterion" in c and "verify" in c:
            valid.append({
                "id": c["id"].strip(),
                "criterion": c["criterion"].strip(),
                "verify": c["verify"].strip(),
            })

    return valid


# ---------------------------------------------------------------------------
# Probe generation
# ---------------------------------------------------------------------------

PROBE_SYSTEM_PROMPT = """You are a probe designer for AI language model evaluation. You create YAML-formatted probes that test specific behavioral criteria.

Each probe must have:
- A realistic user input (not a test instruction)
- Expected behaviors (what a good response does)
- Anti-patterns (what a bad response does)
- Evaluation criteria with pass conditions

Return a JSON array of probe objects with:
- "id": probe ID (e.g., "probe_K_VM_001")
- "name": descriptive name
- "criteria_tested": list of criterion IDs
- "input": the user prompt (realistic scenario)
- "expected_behaviors": list of expected behaviors
- "anti_patterns": list of anti-patterns
- "evaluation": dict of criterion_id -> {"check": str, "pass": str}"""


def generate_probes(
    provider,
    model: str,
    criteria: list[dict],
    answers: dict,
    probes_per_criterion: int = 5,
) -> list[dict]:
    """Generate probes for the new criteria."""
    all_probes = []

    for criterion in criteria:
        user_prompt = (
            f"Generate {probes_per_criterion} diverse probes for this criterion:\n\n"
            f"ID: {criterion['id']}\n"
            f"Criterion: {criterion['criterion']}\n"
            f"How to verify: {criterion['verify']}\n"
            f"Domain: {answers['topic']}\n\n"
            f"Each probe should test a different scenario. Use realistic user inputs, "
            f"not test instructions. Vary the difficulty and context.\n\n"
            f"Return ONLY a JSON array."
        )

        response = ask_llm(provider, model, PROBE_SYSTEM_PROMPT, user_prompt, temperature=0.8)
        probes = extract_json_from_response(response)

        if isinstance(probes, list):
            for p in probes:
                if isinstance(p, dict) and "input" in p:
                    # Ensure criteria_tested references the current criterion
                    if "criteria_tested" not in p:
                        p["criteria_tested"] = [criterion["id"]]
                    all_probes.append(p)

    return all_probes


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

def generate_training_examples(
    provider,
    model: str,
    spec: GeneratorSpec,
    criteria: list[dict],
    examples_per_criterion: int = 5,
) -> list[TrainingExample]:
    """Generate training examples for the new criteria."""
    all_examples = []

    for criterion in criteria:
        criterion_info = {
            "criterion": criterion["criterion"],
            "verify": criterion["verify"],
            "section": "New Domain",
        }
        examples = generate_examples(
            criterion_id=criterion["id"],
            criterion_info=criterion_info,
            spec=spec,
            provider=provider,
            num_examples=examples_per_criterion,
            temperature=0.7,
        )
        all_examples.extend(examples)

    return all_examples


# ---------------------------------------------------------------------------
# File writing
# ---------------------------------------------------------------------------

def write_probes_yaml(probes: list[dict], output_path: Path):
    """Write probes to a YAML file."""
    import yaml  # Optional dependency, fallback to manual YAML

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for i, probe in enumerate(probes):
            if i > 0:
                f.write("---\n")

            f.write(f'id: "{probe.get("id", "")}"\n')
            f.write(f'name: "{probe.get("name", "")}"\n')

            criteria = probe.get("criteria_tested", [])
            f.write("criteria_tested:\n")
            for c in criteria:
                f.write(f'  - "{c}"\n')

            f.write(f'\ninput: "{_escape_yaml_string(probe.get("input", ""))}"\n')

            behaviors = probe.get("expected_behaviors", [])
            if behaviors:
                f.write("\nexpected_behaviors:\n")
                for b in behaviors:
                    f.write(f'  - "{_escape_yaml_string(b)}"\n')

            anti = probe.get("anti_patterns", [])
            if anti:
                f.write("\nanti_patterns:\n")
                for a in anti:
                    f.write(f'  - "{_escape_yaml_string(a)}"\n')

            evaluation = probe.get("evaluation", {})
            if evaluation:
                f.write("\nevaluation:\n")
                for cid, spec in evaluation.items():
                    f.write(f"  {cid}:\n")
                    if isinstance(spec, dict):
                        f.write(f'    check: "{_escape_yaml_string(spec.get("check", ""))}"\n')
                        f.write(f'    pass: "{_escape_yaml_string(spec.get("pass", ""))}"\n')

            f.write("\n")


def _escape_yaml_string(s: str) -> str:
    """Escape quotes and newlines for YAML string output."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def write_training_data(examples: list[TrainingExample], output_dir: Path):
    """Write training examples to per-criterion JSONL files."""
    by_criterion: dict[str, list[TrainingExample]] = {}
    for ex in examples:
        by_criterion.setdefault(ex.criterion_id, []).append(ex)

    files_written = 0
    for cid, group in sorted(by_criterion.items()):
        # Determine path from CATEGORY_MAP
        prefix = cid[:4]
        mapping = CATEGORY_MAP.get(prefix)
        if mapping:
            cat, subcat = mapping
        else:
            cat = "uncategorized"
            subcat = "uncategorized"

        cid_dir = output_dir / cat / subcat
        cid_dir.mkdir(parents=True, exist_ok=True)
        cid_path = cid_dir / f"{cid}.jsonl"

        with open(cid_path, "w") as f:
            for ex in group:
                line = json.dumps(format_raw_jsonl(ex), ensure_ascii=False)
                f.write(line + "\n")

        files_written += 1

    return files_written


# ---------------------------------------------------------------------------
# Presentation and approval
# ---------------------------------------------------------------------------

def present_criteria(criteria: list[dict]) -> list[dict]:
    """Present generated criteria and get user approval."""
    print(f"\n{'=' * 60}")
    print(f"Proposed criteria: {len(criteria)}")
    print(f"{'=' * 60}\n")

    print(f"| {'ID':<12} | {'Criterion':<40} | {'How to verify':<40} |")
    print(f"|{'-' * 14}|{'-' * 42}|{'-' * 42}|")
    for c in criteria:
        crit = c["criterion"][:40]
        verify = c["verify"][:40]
        print(f"| {c['id']:<12} | {crit:<40} | {verify:<40} |")

    print()
    choice = input("Accept these criteria? (y/n/edit) ").strip().lower()

    if choice == "y":
        return criteria
    elif choice == "n":
        print("Criteria rejected. Exiting.")
        sys.exit(0)
    elif choice == "edit":
        # Allow removing individual criteria
        print("\nEnter criterion IDs to REMOVE (comma-separated), or press Enter to keep all:")
        remove = input("> ").strip()
        if remove:
            remove_ids = {r.strip() for r in remove.split(",")}
            criteria = [c for c in criteria if c["id"] not in remove_ids]
            print(f"Kept {len(criteria)} criteria")
        return criteria
    else:
        return criteria


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run_domain_builder(
    provider,
    model: str,
    spec: GeneratorSpec,
    topic: Optional[str] = None,
    category: Optional[str] = None,
    framework_path: str = "docs/AI-SETT-FRAMEWORK.md",
    probes_dir: str = "probes",
    training_dir: str = "training_data",
    probes_per_criterion: int = 5,
    examples_per_criterion: int = 5,
    dry_run: bool = False,
):
    """Main domain builder flow."""
    print("=== AI-SETT Domain Builder ===")

    # Step 1: Interview
    answers = interview_contributor(topic=topic, category=category)
    category = answers["category"]

    # Parse framework for context
    framework = {}
    fw_path = Path(framework_path)
    if fw_path.exists():
        framework = parse_framework(framework_path)
        print(f"\nLoaded {len(framework)} existing criteria from framework")

    # Determine ID prefix and subcategory code
    prefix = CATEGORY_PREFIXES.get(category, "X")
    # Generate subcategory code from topic
    topic_words = answers["topic"].lower().split()
    subcat_code = "".join(w[0].upper() for w in topic_words[:2]) if len(topic_words) >= 2 else "XX"
    subcat_slug = "_".join(answers["topic"].lower().split())
    subcat_slug = re.sub(r'[^a-z0-9_]', '', subcat_slug)

    print(f"\nGenerating criteria for {prefix}.{subcat_code}...")

    # Step 2: Generate criteria
    criteria = generate_criteria(provider, model, answers, framework, prefix, subcat_code)

    if not criteria:
        print("Failed to generate criteria. Exiting.")
        sys.exit(1)

    # Step 3: Present for approval
    criteria = present_criteria(criteria)

    if not criteria:
        print("No criteria accepted. Exiting.")
        sys.exit(1)

    if dry_run:
        print(f"\n--- Dry run: {len(criteria)} criteria generated, no files written ---")
        return

    # Step 4: Generate probes
    print(f"\nGenerating {probes_per_criterion} probes per criterion ({len(criteria) * probes_per_criterion} total)...")
    probes = generate_probes(provider, model, criteria, answers, probes_per_criterion)
    probe_path = Path(probes_dir) / category / f"{subcat_slug}.yaml"
    write_probes_yaml(probes, probe_path)
    print(f"\u2713 {len(probes)} probes written to {probe_path}")

    # Step 5: Generate training data (optional)
    print(f"\nGenerate training data too? (y/n)")
    gen_training = input("> ").strip().lower()

    if gen_training == "y":
        count_input = input(f"How many examples per criterion? [{examples_per_criterion}] ").strip()
        if count_input:
            try:
                examples_per_criterion = int(count_input)
            except ValueError:
                pass

        total = len(criteria) * examples_per_criterion
        print(f"\nGenerating {total} training examples...")
        examples = generate_training_examples(
            provider, model, spec, criteria, examples_per_criterion
        )

        training_path = Path(training_dir)
        # Register new criterion IDs in CATEGORY_MAP for path resolution
        for c in criteria:
            cid_prefix = c["id"][:4]
            if cid_prefix not in CATEGORY_MAP:
                CATEGORY_MAP[cid_prefix] = (category, subcat_slug)

        files = write_training_data(examples, training_path)
        print(f"\u2713 {len(examples)} examples written to {training_path}/")

        # Run validation
        print(f"\nRunning validation...")
        try:
            from tools.validate_training_data import validate_directory
            passed, failed, issues = validate_directory(
                str(training_path / category / subcat_slug),
                framework,
            )
            if failed == 0:
                print(f"\u2713 All {passed} files pass validation")
            else:
                print(f"\u2717 {failed} files failed validation ({issues} issues)")
        except Exception as e:
            print(f"Validation skipped: {e}")

    # Step 6: Summary
    print(f"\n{'=' * 60}")
    print(f"=== Summary ===")
    print(f"{'=' * 60}")
    print(f"New domain: {category.replace('_', ' ').title()} > {answers['topic']} ({prefix}.{subcat_code})")
    print(f"  {len(criteria)} criteria defined")
    print(f"  {len(probes)} probes generated")
    if gen_training == "y":
        print(f"  {len(examples)} training examples generated")
    print(f"\nNext steps:")
    print(f"  1. Review generated files")
    print(f"  2. Edit criteria/probes/examples as needed")
    print(f"  3. Add criteria to docs/AI-SETT-FRAMEWORK.md")
    print(f"  4. Submit PR")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI-SETT Domain Builder — interactive domain creation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  %(prog)s --provider anthropic --model claude-3-5-sonnet-20241022

  # Start with a topic
  %(prog)s --provider anthropic --model claude-3-5-sonnet-20241022 \\
    --topic "cybersecurity incident response"

  # Generate into existing category
  %(prog)s --provider anthropic --model claude-3-5-sonnet-20241022 \\
    --category knowledge --topic "veterinary medicine"

  # Dry run
  %(prog)s --provider anthropic --model claude-3-5-sonnet-20241022 \\
    --topic "test domain" --dry-run
        """,
    )

    parser.add_argument("--provider", required=True,
                        help="LLM provider (anthropic, openai, etc.)")
    parser.add_argument("--model", required=True,
                        help="Model identifier")
    parser.add_argument("--base-url", help="Custom API base URL")
    parser.add_argument("--topic", help="Domain topic (skip first interview question)")
    parser.add_argument("--category", help="Target category (skip category selection)")
    parser.add_argument("--framework", default="docs/AI-SETT-FRAMEWORK.md",
                        help="Path to AI-SETT-FRAMEWORK.md")
    parser.add_argument("--probes-dir", default="probes",
                        help="Output directory for probes (default: probes/)")
    parser.add_argument("--training-dir", default="training_data",
                        help="Output directory for training data (default: training_data/)")
    parser.add_argument("--probes-per-criterion", type=int, default=5,
                        help="Probes per criterion (default: 5)")
    parser.add_argument("--examples-per-criterion", type=int, default=5,
                        help="Training examples per criterion (default: 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate criteria only, don't write files")

    args = parser.parse_args()

    # Initialize provider
    spec_str = f"{args.provider}:{args.model}"
    if args.base_url:
        spec_str += f":{args.base_url}"
    spec = parse_generator_spec(spec_str)
    provider = init_provider(spec)

    run_domain_builder(
        provider=provider,
        model=args.model,
        spec=spec,
        topic=args.topic,
        category=args.category,
        framework_path=args.framework,
        probes_dir=args.probes_dir,
        training_dir=args.training_dir,
        probes_per_criterion=args.probes_per_criterion,
        examples_per_criterion=args.examples_per_criterion,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
