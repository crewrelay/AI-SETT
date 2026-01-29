#!/usr/bin/env python3
"""AI-SETT Probe Filler

Fills empty probe template stubs with LLM-generated content.
Reads criterion definitions from the framework document to generate
meaningful test inputs, expected behaviors, anti-patterns, and
concrete evaluation pass conditions.

Usage:
    python -m tools.probe_filler \
        --probes probes/ \
        --framework docs/AI-SETT-FRAMEWORK.md \
        --provider anthropic --model claude-3-5-haiku-20241022

    python -m tools.probe_filler --probes probes/ --framework docs/AI-SETT-FRAMEWORK.md --dry-run
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
# Framework parsing
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
            # Track section headers
            if line.startswith("##"):
                current_section = line.lstrip("#").strip()
            # Parse criterion table rows: | ID | Criterion | How to verify |
            m = re.match(
                r'^\|\s*([A-Z]\.[A-Z]{2}\.\d{2})\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|',
                line,
            )
            if m:
                cid = m.group(1)
                criterion_text = m.group(2).strip()
                verify_text = m.group(3).strip()
                criteria[cid] = {
                    "criterion": criterion_text,
                    "verify": verify_text,
                    "section": current_section,
                }

    return criteria


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


def find_empty_probes(probe_path: str) -> list[tuple[Path, list[dict]]]:
    """Find all probe files containing empty stubs.

    Returns list of (file_path, [empty_probes]) tuples.
    """
    if yaml is None:
        print("Error: PyYAML required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    path = Path(probe_path)
    result: list[tuple[Path, list[dict]]] = []

    files: list[Path] = []
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("**/*.yaml"))
    else:
        print(f"Error: '{probe_path}' is not a file or directory", file=sys.stderr)
        sys.exit(1)

    for yaml_file in files:
        # Skip expanded files
        if yaml_file.stem.endswith("_expanded"):
            continue

        probes = _load_yaml_file(yaml_file)
        empty = []
        for p in probes:
            if not isinstance(p, dict):
                continue
            if not p.get("id"):
                continue
            # A probe is "empty" if it has no input or empty input
            probe_type = p.get("type", "single_turn")
            if probe_type == "multi_turn":
                turns = p.get("turns", [])
                if not turns or not all(t.get("content") for t in turns):
                    empty.append(p)
            elif probe_type == "simulated":
                continue  # Simulated probes handled separately
            else:
                if not p.get("input") or p["input"].strip() == "":
                    empty.append(p)

        if empty:
            result.append((yaml_file, empty))

    return result


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a probe content writer for AI-SETT, a diagnostic profiling framework for language models.

You receive empty probe templates that have an ID, name, criteria, and evaluation structure, but lack actual test content. Your job is to write concrete, executable probe content.

For each probe, you must generate:
1. **input**: A realistic user prompt that tests the specified criteria. Be specific, not generic. Use concrete scenarios (real-ish names, numbers, topics).
2. **expected_behaviors**: 2-3 specific, observable behaviors the response should exhibit.
3. **anti_patterns**: 1-2 specific failure modes to watch for.
4. **evaluation.*.pass**: A concrete, verifiable pass condition for each criterion. Use patterns the rule evaluator supports:
   - "Contains 'X'" — string inclusion check
   - "does not contain 'X'" — absence check
   - "word count ≤ N" or "under N words" — length check
   - "Contains question" — question mark detection
   - "regex: pattern" — regex match
   - For complex checks, describe the observable behavior clearly

Rules:
- Each probe tests specific criteria. Read the criterion description and "how to verify" carefully.
- The input should be a natural user message, not a test instruction.
- Vary difficulty: some straightforward, some requiring nuance.
- Be concrete: use real programming languages, real-ish scenarios, specific numbers.
- For Knowledge domain probes, test actual knowledge (not meta-questions about knowledge).
- For Reasoning probes, include problems that require actual reasoning.
- For Boundaries probes, test edge cases and refusal scenarios.
- For Calibration probes, design inputs where the length/format/tone can be measured.
- Keep inputs concise — most should be 1-3 sentences.

Return a JSON array of filled probe objects. Each must have:
- "id": same as input
- "input": the new test prompt (non-empty)
- "expected_behaviors": list of 2-3 strings
- "anti_patterns": list of 1-2 strings
- "evaluation": object with same criterion keys, each having "check" and "pass"

Return ONLY the JSON array, no markdown fences or explanation."""


def _build_batch_prompt(
    probes: list[dict],
    criteria_db: dict[str, dict],
) -> str:
    """Build a user prompt for a batch of probes to fill."""
    parts = []
    for p in probes:
        pid = p.get("id", "?")
        name = p.get("name", "?")
        criteria_tested = p.get("criteria_tested", [])

        # Look up criterion definitions
        criterion_info = []
        for cid in criteria_tested:
            info = criteria_db.get(cid, {})
            if info:
                criterion_info.append(
                    f"  - {cid}: \"{info['criterion']}\" — verify: \"{info['verify']}\""
                )
            else:
                criterion_info.append(f"  - {cid}: (no definition found)")

        eval_section = p.get("evaluation", {})
        eval_info = []
        for cid, spec in eval_section.items():
            check = spec.get("check", "?")
            eval_info.append(f"  - {cid}: check=\"{check}\"")

        parts.append(
            f"Probe: {pid}\n"
            f"Name: {name}\n"
            f"Criteria:\n" + "\n".join(criterion_info) + "\n"
            f"Evaluation structure:\n" + "\n".join(eval_info)
        )

    return "Fill these empty probes:\n\n" + "\n\n---\n\n".join(parts)


def fill_batch(
    probes: list[dict],
    criteria_db: dict[str, dict],
    provider,
    model: str,
) -> list[dict]:
    """Fill a batch of empty probes using an LLM."""
    user_prompt = _build_batch_prompt(probes, criteria_db)
    probe_ids = {p.get("id") for p in probes}

    request = CompletionRequest(
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ],
        model=model,
        temperature=0.7,
        max_tokens=4096 * 2,  # Larger for batches
    )

    try:
        response = provider.complete(request)
    except Exception as e:
        print(f"  Error generating content: {e}", file=sys.stderr)
        return []

    # Parse JSON from response
    content = response.content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*', '', content)
        content = re.sub(r'\s*```$', '', content)

    try:
        filled = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  Failed to parse JSON: {e}", file=sys.stderr)
        return []

    if not isinstance(filled, list):
        print(f"  Expected JSON array, got {type(filled).__name__}", file=sys.stderr)
        return []

    # Validate
    valid = []
    for f in filled:
        if not isinstance(f, dict):
            continue
        fid = f.get("id", "")
        if fid not in probe_ids:
            continue
        if not f.get("input") or not f["input"].strip():
            continue
        if not f.get("evaluation"):
            continue
        valid.append(f)

    return valid


# ---------------------------------------------------------------------------
# File rewriting
# ---------------------------------------------------------------------------

def _rewrite_yaml_file(
    file_path: Path,
    filled_map: dict[str, dict],
) -> int:
    """Rewrite a YAML file, merging filled content into empty probes.

    Returns count of probes updated.
    """
    probes = _load_yaml_file(file_path)
    updated = 0

    for p in probes:
        pid = p.get("id", "")
        if pid in filled_map:
            filled = filled_map[pid]
            p["input"] = filled["input"]
            if "expected_behaviors" in filled:
                p["expected_behaviors"] = filled["expected_behaviors"]
            if "anti_patterns" in filled:
                p["anti_patterns"] = filled["anti_patterns"]
            if "evaluation" in filled:
                # Merge evaluation: keep structure, update pass conditions
                for cid, spec in filled["evaluation"].items():
                    if cid in p.get("evaluation", {}):
                        if "check" in spec:
                            p["evaluation"][cid]["check"] = spec["check"]
                        if "pass" in spec:
                            p["evaluation"][cid]["pass"] = spec["pass"]
                    else:
                        p.setdefault("evaluation", {})[cid] = spec
            updated += 1

    # Write back as multi-document YAML
    with open(file_path, "w") as f:
        for i, p in enumerate(probes):
            if i > 0:
                f.write("---\n")
            yaml.dump(p, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return updated


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

BATCH_SIZE = 10  # Probes per LLM call


def fill_probes(
    probe_path: str,
    framework_path: str,
    provider_name: str,
    model: str,
    base_url: Optional[str] = None,
    concurrency: int = 2,
    batch_size: int = BATCH_SIZE,
    dry_run: bool = False,
    verbose: bool = False,
    category: Optional[str] = None,
):
    """Main filler logic."""
    # Parse framework
    criteria_db = parse_framework(framework_path)
    print(f"Parsed {len(criteria_db)} criteria from framework")

    # Find empty probes
    empty_files = find_empty_probes(probe_path)
    total_empty = sum(len(probes) for _, probes in empty_files)
    print(f"Found {total_empty} empty probes across {len(empty_files)} files")

    if total_empty == 0:
        print("All probes have content. Nothing to fill.")
        return

    # Optional category filter
    if category:
        filtered = []
        for fp, probes in empty_files:
            matching = [
                p for p in probes
                if any(criteria_db.get(cid, {}).get("section", "").lower().startswith(category.lower())
                       for cid in p.get("criteria_tested", []))
            ]
            if matching:
                filtered.append((fp, matching))
        empty_files = filtered
        total_empty = sum(len(probes) for _, probes in empty_files)
        print(f"After category filter '{category}': {total_empty} probes")

    if dry_run:
        print("\n--- Dry run: no generation performed ---")
        for fp, probes in empty_files:
            rel = fp.relative_to(Path(probe_path)) if Path(probe_path).is_dir() else fp.name
            print(f"  {rel}: {len(probes)} empty probes")
            if verbose:
                for p in probes:
                    pid = p.get("id", "?")
                    criteria = p.get("criteria_tested", [])
                    print(f"    {pid}: {', '.join(criteria)}")
        print(f"\nTotal to fill: {total_empty}")
        return

    # Initialize provider
    env_var = f"{provider_name.upper()}_API_KEY"
    api_key = os.environ.get(env_var, "")
    if not api_key:
        print(f"Error: Set {env_var} environment variable", file=sys.stderr)
        sys.exit(1)

    provider = get_provider(provider_name, api_key=api_key, base_url=base_url)
    print(f"Provider: {provider_name} | Model: {model}")

    # Flatten all empty probes into batches
    all_empty: list[tuple[Path, dict]] = []
    for fp, probes in empty_files:
        for p in probes:
            all_empty.append((fp, p))

    # Create batches
    batches: list[list[tuple[Path, dict]]] = []
    for i in range(0, len(all_empty), batch_size):
        batches.append(all_empty[i:i + batch_size])

    print(f"Processing {len(batches)} batches of up to {batch_size} probes each")

    # Process batches
    all_filled: dict[str, dict] = {}  # probe_id -> filled content
    processed = 0

    def _process_batch(batch: list[tuple[Path, dict]]) -> list[dict]:
        probes_only = [p for _, p in batch]
        return fill_batch(probes_only, criteria_db, provider, model)

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_process_batch, batch): idx
                for idx, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                idx = futures[future]
                filled = future.result()
                for f in filled:
                    all_filled[f["id"]] = f
                processed += 1
                print(f"  Batch [{processed}/{len(batches)}]: {len(filled)} probes filled")
    else:
        for idx, batch in enumerate(batches):
            filled = _process_batch(batch)
            for f in filled:
                all_filled[f["id"]] = f
            processed += 1
            print(f"  Batch [{processed}/{len(batches)}]: {len(filled)} probes filled")

    print(f"\nGenerated content for {len(all_filled)} probes")

    # Write back to files
    total_written = 0
    files_to_update: dict[Path, dict[str, dict]] = {}
    for fp, probes in empty_files:
        for p in probes:
            pid = p.get("id", "")
            if pid in all_filled:
                files_to_update.setdefault(fp, {})[pid] = all_filled[pid]

    for fp, filled_map in files_to_update.items():
        count = _rewrite_yaml_file(fp, filled_map)
        total_written += count
        if verbose:
            rel = fp.name
            print(f"  Updated {count} probes in {rel}")

    print(f"Wrote {total_written} filled probes across {len(files_to_update)} files")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI-SETT Probe Filler — fill empty probe templates with LLM-generated content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fill all empty probes
  %(prog)s --probes probes/ --framework docs/AI-SETT-FRAMEWORK.md \\
    --provider anthropic --model claude-3-5-haiku-20241022

  # Fill only a specific category
  %(prog)s --probes probes/generation/ --framework docs/AI-SETT-FRAMEWORK.md \\
    --provider openai --model gpt-4o-mini --category generation

  # Dry run — show empty probes without generating
  %(prog)s --probes probes/ --framework docs/AI-SETT-FRAMEWORK.md --dry-run

  # Higher concurrency and custom batch size
  %(prog)s --probes probes/ --framework docs/AI-SETT-FRAMEWORK.md \\
    --provider anthropic --model claude-3-5-haiku-20241022 \\
    --concurrency 4 --batch-size 15
        """,
    )
    parser.add_argument("--probes", required=True, help="Path to probe YAML files or directory")
    parser.add_argument("--framework", required=True,
                        help="Path to AI-SETT-FRAMEWORK.md (for criterion definitions)")
    parser.add_argument("--provider", help="LLM provider for generation")
    parser.add_argument("--model", help="Model for generation")
    parser.add_argument("--base-url", help="Custom API base URL")
    parser.add_argument("--concurrency", type=int, default=2, help="Parallel API calls (default: 2)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Probes per LLM call (default: {BATCH_SIZE})")
    parser.add_argument("--category", help="Only fill probes for this category (e.g. 'generation')")
    parser.add_argument("--dry-run", action="store_true", help="Show empty probes, no generation")
    parser.add_argument("--verbose", action="store_true", help="Detailed output")

    args = parser.parse_args()

    if not args.dry_run:
        if not args.provider:
            parser.error("--provider is required (unless using --dry-run)")
        if not args.model:
            parser.error("--model is required (unless using --dry-run)")

    fill_probes(
        probe_path=args.probes,
        framework_path=args.framework,
        provider_name=args.provider or "",
        model=args.model or "",
        base_url=args.base_url,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        verbose=args.verbose,
        category=args.category,
    )


if __name__ == "__main__":
    main()
