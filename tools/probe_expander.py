#!/usr/bin/env python3
"""AI-SETT Probe Expander

Generates variant probes from existing hand-crafted templates using an LLM.
Each variant tests the same criteria but with different scenarios/wording.

Usage:
    python -m tools.probe_expander \
        --probes probes/understanding/ \
        --provider anthropic --model claude-3-5-haiku-20241022 \
        --target 5

    python -m tools.probe_expander --probes probes/ --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None

from tools.providers import get_provider
from tools.providers.base import CompletionRequest, Message


# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------

def _load_yaml_file(path: Path) -> list[dict]:
    """Load one or more probes from a YAML file (supports multi-document)."""
    results = []
    with open(path) as f:
        for doc in yaml.safe_load_all(f):
            if doc is None:
                continue
            if isinstance(doc, list):
                results.extend(doc)
            else:
                results.append(doc)
    return results


def load_all_probes(probe_path: str) -> tuple[list[dict], dict[str, Path]]:
    """Load probes from path and return (probes, {probe_id: source_file}).

    Only loads probes with non-empty inputs.
    """
    if yaml is None:
        print("Error: PyYAML required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    path = Path(probe_path)
    probes = []
    source_map: dict[str, Path] = {}

    files: list[Path] = []
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("**/*.yaml"))
    else:
        print(f"Error: '{probe_path}' is not a file or directory", file=sys.stderr)
        sys.exit(1)

    for yaml_file in files:
        for p in _load_yaml_file(yaml_file):
            pid = p.get("id", "")
            if not pid:
                continue
            # Skip probes with empty inputs
            if p.get("type") == "multi_turn":
                turns = p.get("turns", [])
                if not turns or not all(t.get("content") for t in turns):
                    continue
            elif p.get("type") == "simulated":
                # Simulated probes don't need input field
                pass
            else:
                if not p.get("input"):
                    continue
            probes.append(p)
            source_map[pid] = yaml_file

    return probes, source_map


def load_existing_expanded(probe_path: str) -> dict[str, list[dict]]:
    """Load existing *_expanded.yaml files and group by template_id."""
    path = Path(probe_path)
    expanded: dict[str, list[dict]] = {}

    if path.is_file():
        search_dir = path.parent
    elif path.is_dir():
        search_dir = path
    else:
        return expanded

    for yaml_file in sorted(search_dir.glob("**/*_expanded.yaml")):
        for p in _load_yaml_file(yaml_file):
            template_id = p.get("template_id", "")
            if template_id:
                expanded.setdefault(template_id, []).append(p)

    return expanded


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Simple whitespace + lowercase tokenization."""
    return set(re.findall(r'\w+', text.lower()))


def jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two strings."""
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def is_duplicate(new_input: str, existing_inputs: list[str], threshold: float) -> bool:
    """Check if new_input is too similar to any existing input."""
    for existing in existing_inputs:
        if jaccard_similarity(new_input, existing) > threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a probe variant generator for AI-SETT, a diagnostic profiling framework for language models.

Given an original probe (a test case), generate {count} variant probes that test the SAME criteria but with DIFFERENT scenarios, wording, or contexts.

Rules:
1. Each variant must test the exact same criteria_tested as the original
2. Use different scenarios, domains, or phrasings — don't just rephrase
3. Vary difficulty slightly (some easier, some harder)
4. Keep evaluation structure matching the original criteria
5. The "pass" conditions should be adapted to the new scenario
6. For multi-turn probes, vary the conversation flow while testing the same skills

Return a JSON array of probe objects. Each object must have:
- "id": "{base_id}_exp{{N}}" where N starts at {start_n}
- "name": a descriptive name for the variant
- "origin": "expanded"
- "template_id": "{template_id}"
- "criteria_tested": exactly {criteria_tested}
- "input": the new probe input (non-empty string)
- "expected_behaviors": list of expected behaviors
- "evaluation": object with keys matching criteria_tested, each having "check" and "pass" fields

Return ONLY the JSON array, no markdown fences or explanation."""


def build_generation_prompt(probe: dict, count: int, start_n: int) -> tuple[str, str]:
    """Build system and user prompts for variant generation."""
    probe_id = probe.get("id", "unknown")
    criteria = json.dumps(probe.get("criteria_tested", []))

    system = SYSTEM_PROMPT.format(
        count=count,
        base_id=probe_id,
        start_n=start_n,
        template_id=probe_id,
        criteria_tested=criteria,
    )

    user = f"Original probe:\n```json\n{json.dumps(probe, indent=2, default=str)}\n```\n\nGenerate {count} variants."
    return system, user


def generate_variants(
    probe: dict,
    provider,
    model: str,
    count: int,
    start_n: int,
    existing_inputs: list[str],
    similarity_threshold: float,
) -> list[dict]:
    """Generate variant probes using an LLM. Returns validated, deduplicated variants."""
    if count <= 0:
        return []

    system_prompt, user_prompt = build_generation_prompt(probe, count, start_n)
    probe_id = probe.get("id", "unknown")
    criteria = set(probe.get("criteria_tested", []))

    request = CompletionRequest(
        messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ],
        model=model,
        temperature=0.7,
        max_tokens=4096,
    )

    try:
        response = provider.complete(request)
    except Exception as e:
        print(f"  Error generating variants for {probe_id}: {e}", file=sys.stderr)
        return []

    # Parse JSON from response
    content = response.content.strip()
    # Strip markdown fences if present
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*', '', content)
        content = re.sub(r'\s*```$', '', content)

    try:
        variants = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  Failed to parse JSON for {probe_id}: {e}", file=sys.stderr)
        return []

    if not isinstance(variants, list):
        print(f"  Expected JSON array for {probe_id}, got {type(variants).__name__}", file=sys.stderr)
        return []

    # Validate and deduplicate
    valid = []
    for v in variants:
        # Validation checks
        if not isinstance(v, dict):
            continue
        if not v.get("id"):
            continue
        if not v.get("input"):
            continue
        v_criteria = set(v.get("criteria_tested", []))
        if v_criteria != criteria:
            continue
        eval_keys = set(v.get("evaluation", {}).keys())
        if not eval_keys.issubset(criteria):
            continue

        # Ensure required fields
        v.setdefault("origin", "expanded")
        v.setdefault("template_id", probe_id)

        # Dedup check
        if is_duplicate(v["input"], existing_inputs, similarity_threshold):
            continue

        existing_inputs.append(v["input"])
        valid.append(v)

    return valid


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------

def write_expanded_yaml(probes: list[dict], output_path: Path):
    """Write expanded probes as multi-document YAML."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for i, p in enumerate(probes):
            if i > 0:
                f.write("---\n")
            yaml.dump(p, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def expand_probes(
    probe_path: str,
    provider_name: str,
    model: str,
    target: int = 5,
    output_dir: Optional[str] = None,
    concurrency: int = 2,
    similarity_threshold: float = 0.6,
    base_url: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = False,
):
    """Main expansion logic."""
    # Load existing probes
    probes, source_map = load_all_probes(probe_path)
    existing_expanded = load_existing_expanded(probe_path)

    print(f"Loaded {len(probes)} original probes")

    # Skip expanded probes (don't expand expansions)
    original_probes = [p for p in probes if p.get("origin") != "expanded"]
    print(f"Original (non-expanded) probes: {len(original_probes)}")

    # Group by criteria to count coverage
    criteria_probes: dict[str, list[dict]] = {}
    for p in probes:
        for cid in p.get("criteria_tested", []):
            criteria_probes.setdefault(cid, []).append(p)

    # Also count existing expanded
    for template_id, exp_probes in existing_expanded.items():
        for ep in exp_probes:
            for cid in ep.get("criteria_tested", []):
                criteria_probes.setdefault(cid, []).append(ep)

    # Determine which probes need expansion
    needs_expansion: list[tuple[dict, int]] = []  # (probe, needed_count)
    for p in original_probes:
        pid = p.get("id", "")
        existing_count = len(existing_expanded.get(pid, []))
        # Count all probes for this probe's criteria
        criteria = p.get("criteria_tested", [])
        if not criteria:
            continue

        # How many expanded variants does THIS probe already have?
        total_for_probe = 1 + existing_count  # 1 = original
        needed = target - total_for_probe
        if needed > 0:
            needs_expansion.append((p, needed))

    total_needed = sum(n for _, n in needs_expansion)
    print(f"Probes needing expansion: {len(needs_expansion)}")
    print(f"Total variants to generate: {total_needed}")
    print(f"Target: {target} probes per criterion")

    if dry_run:
        print("\n--- Dry run: no generation performed ---")
        for p, needed in needs_expansion:
            pid = p.get("id", "")
            existing = len(existing_expanded.get(pid, []))
            print(f"  {pid}: has {1 + existing}, needs {needed} more")
        return

    if total_needed == 0:
        print("All probes already at target. Nothing to generate.")
        return

    # Initialize provider
    env_var = f"{provider_name.upper()}_API_KEY"
    api_key = os.environ.get(env_var, "")
    if not api_key:
        print(f"Error: Set {env_var} environment variable", file=sys.stderr)
        sys.exit(1)

    provider = get_provider(provider_name, api_key=api_key, base_url=base_url)
    print(f"Provider: {provider_name} | Model: {model}")

    # Generate variants
    results: dict[str, list[dict]] = {}  # source_file_stem -> new probes

    def _expand_one(probe: dict, needed: int) -> tuple[str, list[dict]]:
        pid = probe.get("id", "")
        existing = existing_expanded.get(pid, [])
        start_n = len(existing) + 1

        # Collect all existing inputs for dedup
        all_inputs = [probe.get("input", "")]
        all_inputs.extend(ep.get("input", "") for ep in existing)

        new_variants = generate_variants(
            probe, provider, model, needed, start_n,
            all_inputs, similarity_threshold,
        )

        if verbose:
            print(f"  {pid}: generated {len(new_variants)}/{needed} variants")

        return pid, new_variants

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_expand_one, p, n): p
                for p, n in needs_expansion
            }
            for i, future in enumerate(as_completed(futures), 1):
                pid, new_variants = future.result()
                if new_variants:
                    results[pid] = new_variants
                print(f"  [{i}/{len(needs_expansion)}] {pid}: {len(new_variants)} variants")
    else:
        for i, (probe, needed) in enumerate(needs_expansion, 1):
            pid, new_variants = _expand_one(probe, needed)
            if new_variants:
                results[pid] = new_variants
            print(f"  [{i}/{len(needs_expansion)}] {pid}: {len(new_variants)} variants")

    # Write output files
    # Group new variants by source file
    file_variants: dict[Path, list[dict]] = {}
    for pid, variants in results.items():
        source_file = source_map.get(pid)
        if source_file:
            file_variants.setdefault(source_file, []).extend(variants)

    total_written = 0
    for source_file, variants in file_variants.items():
        stem = source_file.stem
        if output_dir:
            out_dir = Path(output_dir)
        else:
            out_dir = source_file.parent

        out_path = out_dir / f"{stem}_expanded.yaml"

        # Merge with any existing expanded probes in the file
        existing_in_file = []
        if out_path.exists():
            existing_in_file = _load_yaml_file(out_path)

        all_probes = existing_in_file + variants
        write_expanded_yaml(all_probes, out_path)
        total_written += len(variants)
        print(f"  Wrote {len(variants)} new variants to {out_path}")

    print(f"\nExpansion complete. Generated {total_written} new probe variants.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI-SETT Probe Expander — generate LLM-powered probe variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Expand all probes to 5 per criterion (default)
  %(prog)s --probes probes/ --provider anthropic --model claude-3-5-haiku-20241022 --target 5

  # Expand specific directory, 10 per criterion
  %(prog)s --probes probes/understanding/ --provider openai --model gpt-4o-mini \\
    --target 10 --output probes/understanding/

  # Dry run — show what would be generated
  %(prog)s --probes probes/ --dry-run
        """,
    )
    parser.add_argument("--probes", required=True, help="Path to probe YAML files or directory")
    parser.add_argument("--provider", help="LLM provider for generation")
    parser.add_argument("--model", help="Model for generation")
    parser.add_argument("--base-url", help="Custom API base URL")
    parser.add_argument("--target", type=int, default=5, help="Target probes per criterion (default: 5)")
    parser.add_argument("--output", help="Output directory (default: same as input)")
    parser.add_argument("--concurrency", type=int, default=2, help="Parallel API calls (default: 2)")
    parser.add_argument("--similarity-threshold", type=float, default=0.6,
                        help="Jaccard dedup threshold (default: 0.6)")
    parser.add_argument("--dry-run", action="store_true", help="Show counts, no generation")
    parser.add_argument("--verbose", action="store_true", help="Detailed output")

    args = parser.parse_args()

    if not args.dry_run:
        if not args.provider:
            parser.error("--provider is required (unless using --dry-run)")
        if not args.model:
            parser.error("--model is required (unless using --dry-run)")

    expand_probes(
        probe_path=args.probes,
        provider_name=args.provider or "",
        model=args.model or "",
        target=args.target,
        output_dir=args.output,
        concurrency=args.concurrency,
        similarity_threshold=args.similarity_threshold,
        base_url=args.base_url,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
