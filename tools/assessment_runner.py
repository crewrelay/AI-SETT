#!/usr/bin/env python3
"""AI-SETT Assessment Runner

Loads probes from YAML files, sends them to a model provider,
evaluates responses against criteria, and builds a diagnostic profile.

Usage:
    python -m tools.assessment_runner \
        --probes probes/understanding/ \
        --provider openai --model gpt-4o \
        --output results/gpt4o.json

    python -m tools.assessment_runner --dry-run --probes probes/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None

from tools import __version__
from tools.evaluators import evaluate
from tools.providers import get_provider, list_providers
from tools.providers.base import CompletionRequest, CompletionResponse, Message

# ---------------------------------------------------------------------------
# Category mapping: criterion ID prefix -> (category_name, subcategory_name)
# ---------------------------------------------------------------------------
CATEGORY_MAP = {
    "U.BR": ("Understanding", "Basic requests"),
    "U.CR": ("Understanding", "Complex requests"),
    "U.AR": ("Understanding", "Ambiguous requests"),
    "U.IR": ("Understanding", "Implicit requests"),
    "U.CX": ("Understanding", "Contextual understanding"),
    "C.RL": ("Calibration", "Response length"),
    "C.RD": ("Calibration", "Response depth"),
    "C.FM": ("Calibration", "Format"),
    "C.TN": ("Calibration", "Tone"),
    "C.ST": ("Calibration", "Stopping"),
    "G.TW": ("Generation", "Technical writing"),
    "G.CW": ("Generation", "Creative writing"),
    "G.PW": ("Generation", "Persuasive writing"),
    "G.AW": ("Generation", "Academic writing"),
    "G.BW": ("Generation", "Business writing"),
    "G.CD": ("Generation", "Code generation"),
    "G.SO": ("Generation", "Structured output"),
    "G.SM": ("Generation", "Summarization"),
    "G.TR": ("Generation", "Translation"),
    "K.PG": ("Knowledge", "Programming fundamentals"),
    "K.AP": ("Knowledge", "APIs"),
    "K.SC": ("Knowledge", "Security"),
    "K.DO": ("Knowledge", "DevOps"),
    "K.DT": ("Knowledge", "Databases"),
    "K.MT": ("Knowledge", "Mathematics"),
    "K.SG": ("Knowledge", "Science"),
    "K.BF": ("Knowledge", "Business & Finance"),
    "K.LG": ("Knowledge", "Legal concepts"),
    "K.MH": ("Knowledge", "Medical/Health"),
    "K.HG": ("Knowledge", "History & Geography"),
    "K.AC": ("Knowledge", "Arts & Culture"),
    "K.ED": ("Knowledge", "Education & Pedagogy"),
    "K.PM": ("Knowledge", "Project Management"),
    "K.DS": ("Knowledge", "Design"),
    "R.LG": ("Reasoning", "Logical reasoning"),
    "R.MT": ("Reasoning", "Mathematical reasoning"),
    "R.CS": ("Reasoning", "Causal reasoning"),
    "R.AN": ("Reasoning", "Analogical reasoning"),
    "R.CT": ("Reasoning", "Critical thinking"),
    "R.PS": ("Reasoning", "Problem solving"),
    "B.RF": ("Boundaries", "Appropriate refusals"),
    "B.OR": ("Boundaries", "Avoids over-refusal"),
    "B.UN": ("Boundaries", "Uncertainty & limits"),
    "B.PB": ("Boundaries", "Professional boundaries"),
    "B.SF": ("Boundaries", "Safety"),
    "I.MT": ("Interaction", "Multi-turn coherence"),
    "I.EH": ("Interaction", "Error handling"),
    "I.CR": ("Interaction", "Clarification & repair"),
    "T.WS": ("Tool use", "Web search"),
    "T.CE": ("Tool use", "Code execution"),
    "T.FH": ("Tool use", "File handling"),
    "T.AC": ("Tool use", "API calling"),
    "T.CL": ("Tool use", "Calculator/computation"),
    "T.IU": ("Tool use", "Image understanding"),
    "T.TS": ("Tool use", "Tool selection"),
    "E.ER": ("Emotional intelligence", "Emotional recognition"),
    "E.EP": ("Emotional intelligence", "Empathetic response"),
    "E.DC": ("Emotional intelligence", "Difficult conversations"),
    "E.SA": ("Emotional intelligence", "Social awareness"),
    "M.SA": ("Metacognition", "Self-awareness"),
    "M.LA": ("Metacognition", "Learning & adaptation"),
    "M.SS": ("Metacognition", "Strategy selection"),
    "L.IC": ("Learning", "In-context learning"),
    "L.IF": ("Learning", "Instruction following"),
    "L.DM": ("Learning", "Domain-specific learning"),
    "L.ER": ("Learning", "Error-based learning"),
    "L.TR": ("Learning", "Learning transfer"),
    "T.DA": ("Teaching", "Diagnostic assessment"),
    "T.EQ": ("Teaching", "Explanation quality"),
    "T.SC": ("Teaching", "Scaffolding"),
    "T.FB": ("Teaching", "Feedback"),
    "T.AD": ("Teaching", "Adaptation"),
}


def _criterion_prefix(criterion_id: str) -> str:
    """Extract the category prefix from a criterion ID (e.g. 'U.BR' from 'U.BR.01')."""
    parts = criterion_id.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return criterion_id


# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------

def load_probes(probe_path: str) -> list[dict]:
    """Load probe definitions from YAML files."""
    if yaml is None:
        print("Error: PyYAML required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    path = Path(probe_path)
    probes = []

    if path.is_file():
        probes.extend(_load_yaml_file(path))
    elif path.is_dir():
        for yaml_file in sorted(path.glob("**/*.yaml")):
            probes.extend(_load_yaml_file(yaml_file))
    else:
        print(f"Error: '{probe_path}' is not a file or directory", file=sys.stderr)
        sys.exit(1)

    return probes


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


# ---------------------------------------------------------------------------
# Running probes
# ---------------------------------------------------------------------------

def run_single_turn(probe: dict, provider, model: str, temperature: float) -> CompletionResponse:
    """Run a single-turn probe."""
    messages = [Message(role="user", content=probe["input"])]

    # If the probe has a system message, prepend it
    if "system" in probe:
        messages.insert(0, Message(role="system", content=probe["system"]))

    request = CompletionRequest(
        messages=messages,
        model=model,
        temperature=temperature,
    )
    return provider.complete(request)


def run_multi_turn(probe: dict, provider, model: str, temperature: float) -> CompletionResponse:
    """Run a multi-turn probe. Returns the final response.

    Multi-turn probes have a 'turns' list of {role, content} dicts.
    We send each user turn, capture assistant response, and feed it back.
    """
    messages: list[Message] = []

    if "system" in probe:
        messages.append(Message(role="system", content=probe["system"]))

    last_response: Optional[CompletionResponse] = None
    total_latency = 0.0

    for turn in probe["turns"]:
        if turn["role"] == "user":
            messages.append(Message(role="user", content=turn["content"]))
            request = CompletionRequest(
                messages=messages,
                model=model,
                temperature=temperature,
            )
            last_response = provider.complete(request)
            total_latency += last_response.latency_ms
            messages.append(Message(role="assistant", content=last_response.content))
        elif turn["role"] == "assistant":
            # Scripted assistant turn (for setting up context)
            messages.append(Message(role="assistant", content=turn["content"]))

    if last_response is None:
        raise ValueError(f"Multi-turn probe '{probe.get('id', '?')}' produced no response")

    # Override latency to total
    last_response.latency_ms = total_latency
    return last_response


def run_probe(probe: dict, provider, model: str, temperature: float) -> dict:
    """Run a probe and evaluate the response.

    Returns a probe_result dict matching the result schema.
    """
    probe_id = probe.get("id", "unknown")
    probe_type = probe.get("type", "single_turn")

    try:
        if probe_type == "multi_turn":
            response = run_multi_turn(probe, provider, model, temperature)
        else:
            response = run_single_turn(probe, provider, model, temperature)
    except Exception as e:
        return {
            "probe_id": probe_id,
            "input": probe.get("input", probe.get("turns", "")),
            "response": "",
            "criteria_results": {},
            "evidence": {},
            "latency_ms": 0,
            "error": str(e),
        }

    # Evaluate each criterion
    criteria_results = {}
    evidence = {}

    evaluation = probe.get("evaluation", {})
    for criterion_id, spec in evaluation.items():
        passed, ev = evaluate(response.content, spec)
        criteria_results[criterion_id] = passed
        evidence[criterion_id] = ev

    return {
        "probe_id": probe_id,
        "input": probe.get("input", [{"role": t["role"], "content": t["content"]} for t in probe.get("turns", [])]),
        "response": response.content,
        "criteria_results": criteria_results,
        "evidence": evidence,
        "latency_ms": response.latency_ms,
    }


# ---------------------------------------------------------------------------
# Profile building
# ---------------------------------------------------------------------------

def build_profile(probe_results: list[dict], aggregation: str = "majority") -> dict:
    """Aggregate probe results into a diagnostic profile.

    When multiple probes test the same criterion, use aggregation mode:
    - "any": criterion demonstrated if ANY probe passes
    - "all": criterion demonstrated only if ALL probes pass
    - "majority": criterion demonstrated if >50% of probes pass (default)
    """
    # Collect all results per criterion
    criterion_votes: dict[str, list[bool]] = {}
    for result in probe_results:
        for cid, passed in result.get("criteria_results", {}).items():
            criterion_votes.setdefault(cid, []).append(passed)

    # Aggregate
    criterion_final: dict[str, bool] = {}
    for cid, votes in criterion_votes.items():
        if aggregation == "any":
            criterion_final[cid] = any(votes)
        elif aggregation == "all":
            criterion_final[cid] = all(votes)
        else:  # majority
            criterion_final[cid] = sum(votes) > len(votes) / 2

    # Build hierarchical profile
    profile: dict = {}
    for cid, demonstrated in criterion_final.items():
        prefix = _criterion_prefix(cid)
        cat_name, subcat_name = CATEGORY_MAP.get(prefix, ("Unknown", "Unknown"))

        if cat_name not in profile:
            profile[cat_name] = {"demonstrated": 0, "total": 0, "subcategories": {}}

        cat = profile[cat_name]
        cat["total"] += 1
        if demonstrated:
            cat["demonstrated"] += 1

        if subcat_name not in cat["subcategories"]:
            cat["subcategories"][subcat_name] = {"demonstrated": 0, "total": 0, "criteria": {}}

        subcat = cat["subcategories"][subcat_name]
        subcat["total"] += 1
        if demonstrated:
            subcat["demonstrated"] += 1
        subcat["criteria"][cid] = demonstrated

    return profile


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI-SETT Assessment Runner — diagnostic profiling for language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run probes against OpenAI
  %(prog)s --probes probes/understanding/ --provider openai --model gpt-4o

  # Use a local OpenAI-compatible endpoint
  %(prog)s --probes probes/ --provider openai --model local-model \\
    --base-url http://localhost:8000/v1

  # Dry run (load and validate probes without API calls)
  %(prog)s --dry-run --probes probes/

  # Custom output and concurrency
  %(prog)s --probes probes/ --provider anthropic --model claude-3-opus-20240229 \\
    --output results/claude.json --concurrency 4
        """,
    )
    parser.add_argument("--probes", help="Path to probe YAML file or directory")
    parser.add_argument("--provider", help="Provider name (openai, anthropic, google, mistral, cohere)")
    parser.add_argument("--model", help="Model identifier")
    parser.add_argument("--base-url", help="Custom base URL for API endpoint")
    parser.add_argument("--output", default="assessment_results.json", help="Output path for results JSON")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature (default: 0.0)")
    parser.add_argument("--concurrency", type=int, default=2, help="Max concurrent API calls (default: 2)")
    parser.add_argument("--aggregation", choices=["any", "all", "majority"], default="majority",
                        help="How to aggregate when multiple probes test the same criterion (default: majority)")
    parser.add_argument("--dry-run", action="store_true", help="Load probes and validate without making API calls")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output during assessment")
    parser.add_argument("--list-providers", action="store_true", help="List available providers and exit")

    args = parser.parse_args()

    if args.list_providers:
        print("Available providers:", ", ".join(list_providers()))
        return

    if not args.probes:
        parser.error("--probes is required (unless using --list-providers)")

    # Load probes
    probes = load_probes(args.probes)
    print(f"Loaded {len(probes)} probes from {args.probes}")

    # Collect unique criteria
    all_criteria = set()
    for p in probes:
        all_criteria.update(p.get("criteria_tested", []))
    print(f"Testing {len(all_criteria)} unique criteria")

    if args.dry_run:
        print("\n--- Dry run: probes loaded successfully ---")
        for p in probes:
            pid = p.get("id", "?")
            name = p.get("name", "?")
            criteria = p.get("criteria_tested", [])
            ptype = p.get("type", "single_turn")
            print(f"  [{ptype}] {pid}: {name} ({len(criteria)} criteria)")
        categories = set()
        for cid in all_criteria:
            prefix = _criterion_prefix(cid)
            if prefix in CATEGORY_MAP:
                categories.add(CATEGORY_MAP[prefix][0])
        print(f"\nCategories covered: {', '.join(sorted(categories)) or 'none'}")
        return

    # Validate required args for live run
    if not args.provider:
        parser.error("--provider is required for live runs")
    if not args.model:
        parser.error("--model is required for live runs")

    # Resolve API key
    env_var = f"{args.provider.upper()}_API_KEY"
    api_key = os.environ.get(env_var, "")
    if not api_key:
        print(f"Error: Set {env_var} environment variable", file=sys.stderr)
        sys.exit(1)

    # Initialize provider
    provider = get_provider(args.provider, api_key=api_key, base_url=args.base_url)
    print(f"Provider: {args.provider} | Model: {args.model} | Temperature: {args.temperature}")

    # Run probes
    start_time = time.time()
    probe_results = []

    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(run_probe, p, provider, args.model, args.temperature): p
                for p in probes
            }
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                probe_results.append(result)
                pid = result["probe_id"]
                if result.get("error"):
                    print(f"  [{i}/{len(probes)}] {pid}: ERROR - {result['error']}")
                elif args.verbose:
                    passed = sum(1 for v in result["criteria_results"].values() if v)
                    total = len(result["criteria_results"])
                    print(f"  [{i}/{len(probes)}] {pid}: {passed}/{total} demonstrated ({result['latency_ms']:.0f}ms)")
                else:
                    print(f"  [{i}/{len(probes)}] {pid}: done")
    else:
        for i, probe in enumerate(probes, 1):
            result = run_probe(probe, provider, args.model, args.temperature)
            probe_results.append(result)
            pid = result["probe_id"]
            if result.get("error"):
                print(f"  [{i}/{len(probes)}] {pid}: ERROR - {result['error']}")
            elif args.verbose:
                passed = sum(1 for v in result["criteria_results"].values() if v)
                total = len(result["criteria_results"])
                print(f"  [{i}/{len(probes)}] {pid}: {passed}/{total} demonstrated ({result['latency_ms']:.0f}ms)")
            else:
                print(f"  [{i}/{len(probes)}] {pid}: done")

    duration = time.time() - start_time

    # Build profile
    profile = build_profile(probe_results, aggregation=args.aggregation)

    # Collect categories
    categories_tested = set()
    for cid in all_criteria:
        prefix = _criterion_prefix(cid)
        if prefix in CATEGORY_MAP:
            categories_tested.add(CATEGORY_MAP[prefix][0])

    # Build result
    result = {
        "version": "1.0.0",
        "model": args.model,
        "provider": args.provider,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "probe_count": len(probes),
            "criteria_tested": len(all_criteria),
            "categories_tested": len(categories_tested),
            "temperature": args.temperature,
            "duration_seconds": round(duration, 2),
            "assessment_runner_version": __version__,
        },
        "probe_results": probe_results,
        "profile": profile,
    }

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nAssessment complete in {duration:.1f}s")
    print(f"Results written to: {output_path}")

    # Print summary
    print("\n--- Profile Summary ---")
    for cat_name in sorted(profile):
        cat = profile[cat_name]
        print(f"  {cat_name}: {cat['demonstrated']}/{cat['total']} demonstrated")
        if args.verbose:
            for sub_name in sorted(cat["subcategories"]):
                sub = cat["subcategories"][sub_name]
                print(f"    {sub_name}: {sub['demonstrated']}/{sub['total']}")

    # Print gaps
    gaps = []
    for cat_name, cat in profile.items():
        for sub_name, sub in cat["subcategories"].items():
            for cid, demonstrated in sub["criteria"].items():
                if not demonstrated:
                    gaps.append(cid)

    if gaps:
        print(f"\nGaps identified: {len(gaps)}")
        if args.verbose:
            for g in sorted(gaps):
                print(f"  - {g}")
    else:
        print("\nNo gaps identified in tested criteria.")


if __name__ == "__main__":
    main()
