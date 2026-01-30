#!/usr/bin/env python3
"""AI-SETT Model Gym — quick behavioral workout routines for language models.

Samples a diverse subset of probes, runs assessment, analyzes gaps, and
suggests targeted training — all in one command. Like a daily workout:
pick exercises, do the reps, see what's weak.

Usage:
    # Quick workout (30 probes, all categories)
    python -m tools.model_gym --provider anthropic --model claude-3-5-sonnet-20241022

    # Focus on weak areas from last run
    python -m tools.model_gym --provider openai --model gpt-4o \
        --focus-gaps results/gpt4o-last.json

    # Schedule nightly workouts (2am default, power is cheaper 12-6am)
    python -m tools.model_gym --provider anthropic --model claude-3-5-sonnet-20241022 \
        --schedule 02:00

    # Dry run — show workout plan
    python -m tools.model_gym --dry-run --probes probes/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tools import __version__
from tools.assessment_runner import (
    CATEGORY_MAP,
    _criterion_prefix,
    build_profile,
    load_probes,
    run_probe,
)
from tools.training_data_generator import TIER_BASE, TIER_OPTIONAL
from tools.gap_analyzer import (
    compare_assessments,
    compute_priorities,
    extract_gaps,
    find_zpd_candidates,
)
from tools.providers import get_provider

# ---------------------------------------------------------------------------
# Probe sampling
# ---------------------------------------------------------------------------

def _group_by_category(probes: list[dict]) -> dict[str, list[dict]]:
    """Group probes by their primary category name."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in probes:
        criteria = p.get("criteria_tested", [])
        if not criteria:
            groups["Unknown"].append(p)
            continue
        # Use first criterion to determine category
        prefix = _criterion_prefix(criteria[0])
        cat_name = CATEGORY_MAP.get(prefix, ("Unknown", "Unknown"))[0]
        groups[cat_name].append(p)
    return dict(groups)


def _load_gap_weights(gap_result_path: str) -> dict[str, float]:
    """Load a previous result and compute per-category gap weights.

    Categories with higher gap ratios get higher weights.
    Categories with 0 gaps get weight 1 (minimum).
    Categories with gaps get weight 1 + 2 * gap_ratio (up to 3x).
    """
    with open(gap_result_path) as f:
        data = json.load(f)

    profile = data.get("profile", {})
    weights: dict[str, float] = {}

    for cat_name, cat in profile.items():
        total = cat.get("total", 0)
        demonstrated = cat.get("demonstrated", 0)
        if total == 0:
            weights[cat_name] = 1.0
            continue
        gap_ratio = (total - demonstrated) / total
        weights[cat_name] = 1.0 + 2.0 * gap_ratio  # 1x to 3x

    return weights


def sample_workout(
    probes: list[dict],
    budget: int,
    categories: Optional[list[str]] = None,
    gap_result_path: Optional[str] = None,
) -> list[dict]:
    """Sample a diverse subset of probes for a workout.

    Args:
        probes: All available probes.
        budget: Max probes to select (0 = all).
        categories: Optional category filter (lowercase names).
        gap_result_path: Previous result JSON for gap-focused sampling.

    Returns:
        Selected probe list.
    """
    if budget <= 0:
        # Full workout
        if categories:
            groups = _group_by_category(probes)
            cat_filter = {c.lower() for c in categories}
            return [
                p for cat, ps in groups.items()
                if cat.lower() in cat_filter
                for p in ps
            ]
        return list(probes)

    groups = _group_by_category(probes)

    # Apply category filter
    if categories:
        cat_filter = {c.lower() for c in categories}
        groups = {k: v for k, v in groups.items() if k.lower() in cat_filter}

    if not groups:
        return []

    # Compute per-category weights
    if gap_result_path:
        weights = _load_gap_weights(gap_result_path)
    else:
        weights = {cat: 1.0 for cat in groups}

    # Allocate budget proportionally by weight
    total_weight = sum(weights.get(cat, 1.0) for cat in groups)
    allocations: dict[str, int] = {}
    remaining = budget

    for cat in groups:
        w = weights.get(cat, 1.0)
        alloc = max(1, round(budget * w / total_weight))
        allocations[cat] = min(alloc, len(groups[cat]))
        remaining -= allocations[cat]

    # Distribute leftover to highest-weight categories
    if remaining > 0:
        sorted_cats = sorted(groups.keys(), key=lambda c: -weights.get(c, 1.0))
        for cat in sorted_cats:
            if remaining <= 0:
                break
            extra = min(remaining, len(groups[cat]) - allocations[cat])
            if extra > 0:
                allocations[cat] += extra
                remaining -= extra

    # If over budget, trim from lowest-weight categories
    while sum(allocations.values()) > budget:
        sorted_cats = sorted(groups.keys(), key=lambda c: weights.get(c, 1.0))
        for cat in sorted_cats:
            if sum(allocations.values()) <= budget:
                break
            if allocations[cat] > 1:
                allocations[cat] -= 1

    # Sample within each category, maximizing criterion diversity
    selected = []
    for cat, count in allocations.items():
        cat_probes = groups[cat]
        if count >= len(cat_probes):
            selected.extend(cat_probes)
            continue

        # Prefer probes testing different criteria
        seen_criteria: set[str] = set()
        diverse: list[dict] = []
        rest: list[dict] = []

        random.shuffle(cat_probes)
        for p in cat_probes:
            criteria = set(p.get("criteria_tested", []))
            if criteria - seen_criteria:
                diverse.append(p)
                seen_criteria.update(criteria)
            else:
                rest.append(p)

        pool = diverse + rest
        selected.extend(pool[:count])

    random.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

BAR_FULL = "\u2588"
BAR_EMPTY = "\u2591"
BAR_WIDTH = 12


def _bar(demonstrated: int, total: int) -> str:
    """Render a Unicode bar chart."""
    if total == 0:
        return BAR_EMPTY * BAR_WIDTH
    ratio = demonstrated / total
    filled = round(ratio * BAR_WIDTH)
    return BAR_FULL * filled + BAR_EMPTY * (BAR_WIDTH - filled)


def _pct(demonstrated: int, total: int) -> str:
    if total == 0:
        return "  --%"
    return f"{100 * demonstrated / total:4.0f}%"


def print_gym_report(
    profile: dict,
    gaps: list[dict],
    zpd: list[dict],
    priorities: list[dict],
    comparison: Optional[dict],
    duration: float,
    model: str,
    probe_count: int,
    total_probes: int,
    sampling: str,
):
    """Print formatted gym report to terminal."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    mins = int(duration // 60)
    secs = int(duration % 60)

    print()
    print("=" * 50)
    print("  MODEL GYM REPORT")
    print("=" * 50)
    print(f"  Model:    {model}")
    print(f"  Date:     {now}")
    print(f"  Probes:   {probe_count} / {total_probes} ({sampling})")
    print(f"  Duration: {mins}m {secs:02d}s")
    print()

    # Category scores
    print("--- Category Scores ---")
    FOCUS_THRESHOLD = 0.7
    for cat_name in sorted(profile):
        cat = profile[cat_name]
        d, t = cat["demonstrated"], cat["total"]
        bar = _bar(d, t)
        pct = _pct(d, t)
        focus = "  <- FOCUS" if t > 0 and d / t < FOCUS_THRESHOLD else ""
        label = f"{cat_name}:".ljust(30)
        score = f"{d}/{t}".rjust(6)
        print(f"  {label} {score}   {bar} {pct}{focus}")

    # Top gaps
    if gaps:
        print(f"\n--- Top Gaps ({len(gaps)} total) ---")
        # Group by category > subcategory
        gap_groups: dict[str, list[str]] = defaultdict(list)
        for g in gaps:
            key = f"{g['category']} > {g['subcategory']}"
            gap_groups[key].append(g["criterion_id"])

        for i, (key, cids) in enumerate(list(gap_groups.items())[:10], 1):
            print(f"  {i}. {key}: {', '.join(cids)}")

    # ZPD candidates
    if zpd:
        print("\n--- ZPD (Most Productive Targets) ---")
        for c in zpd[:5]:
            print(f"  {c['category']} > {c['subcategory']}: "
                  f"{c['demonstrated']}/{c['total']} demonstrated, "
                  f"{c['gap_count']} gaps to close")
            print(f"    Gaps: {', '.join(c['gap_criteria'])}")

    # Training suggestions
    all_gap_ids = [g["criterion_id"] for g in gaps]
    if all_gap_ids:
        print("\n--- Training Suggestions ---")
        # Prioritize ZPD gaps first, then top priorities
        zpd_gaps = []
        for c in zpd:
            zpd_gaps.extend(c["gap_criteria"])
        priority_gaps = []
        for p in priorities:
            if p["gap_count"] > 0:
                priority_gaps.extend(p["gap_criteria"])

        # Deduplicate preserving order
        suggested = []
        seen = set()
        for cid in zpd_gaps + priority_gaps + all_gap_ids:
            if cid not in seen:
                suggested.append(cid)
                seen.add(cid)

        print(f"  Focus training data on: {', '.join(suggested[:10])}")
        if len(suggested) > 10:
            print(f"  ... and {len(suggested) - 10} more")

    # Longitudinal comparison
    if comparison:
        print("\n--- Progress Since Last Run ---")
        nd = comparison.get("newly_demonstrated", [])
        reg = comparison.get("regressions", [])
        print(f"  Newly demonstrated: {', '.join(nd) if nd else 'none'} "
              f"({'+' if nd else ''}{len(nd)})")
        print(f"  Regressions: {', '.join(reg) if reg else 'none'} "
              f"({len(reg)})")
        stable_gaps = comparison.get("stable_gaps", [])
        prev_gaps = len(stable_gaps) + len(nd)
        curr_gaps = len(stable_gaps) + len(reg)
        delta = curr_gaps - prev_gaps
        sign = "+" if delta > 0 else ""
        print(f"  Gap delta: {prev_gaps} -> {curr_gaps} ({sign}{delta})")

    print()


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def install_schedule(
    time_str: str,
    provider: str,
    model: str,
    base_url: Optional[str],
    probes_path: str,
    budget: int,
    output_dir: str,
    focus_gaps: Optional[str],
    categories: Optional[str],
):
    """Install a cron job for scheduled gym runs.

    Default: 2am daily (off-peak power, 12am-6am window).
    """
    # Parse time
    parts = time_str.split(":")
    if len(parts) != 2:
        print(f"Error: --schedule expects HH:MM format, got '{time_str}'", file=sys.stderr)
        sys.exit(1)
    hour, minute = int(parts[0]), int(parts[1])

    # Build command
    script_dir = Path(__file__).resolve().parent.parent
    python = sys.executable
    cmd_parts = [
        python, "-m", "tools.model_gym",
        "--provider", provider,
        "--model", model,
        "--probes", probes_path,
        "--budget", str(budget),
        "--output", output_dir,
        "--concurrency", "8",
    ]
    if base_url:
        cmd_parts.extend(["--base-url", base_url])
    if focus_gaps:
        cmd_parts.extend(["--focus-gaps", focus_gaps])
    if categories:
        cmd_parts.extend(["--categories", categories])

    cmd = " ".join(cmd_parts)
    cron_line = f"{minute} {hour} * * * cd {script_dir} && {cmd} >> {output_dir}/gym.log 2>&1"

    # Check for existing gym cron
    try:
        existing = subprocess.run(
            ["crontab", "-l"], capture_output=True, text=True
        )
        current_crontab = existing.stdout if existing.returncode == 0 else ""
    except FileNotFoundError:
        print("Error: crontab not available on this system", file=sys.stderr)
        sys.exit(1)

    # Remove any existing model_gym lines
    lines = [
        line for line in current_crontab.splitlines()
        if "tools.model_gym" not in line
    ]
    lines.append(cron_line)

    new_crontab = "\n".join(lines) + "\n"

    # Install
    proc = subprocess.run(
        ["crontab", "-"], input=new_crontab, capture_output=True, text=True
    )
    if proc.returncode != 0:
        print(f"Error installing cron: {proc.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"Scheduled gym workout at {hour:02d}:{minute:02d} daily")
    print(f"  Provider: {provider} | Model: {model}")
    print(f"  Budget: {budget} probes | Output: {output_dir}/")
    if focus_gaps:
        print(f"  Gap focus: {focus_gaps}")
    print(f"\nCron line:\n  {cron_line}")
    print(f"\nTo remove: crontab -e  (delete the tools.model_gym line)")
    print(f"To view logs: tail -f {output_dir}/gym.log")


def uninstall_schedule():
    """Remove all model_gym cron jobs."""
    try:
        existing = subprocess.run(
            ["crontab", "-l"], capture_output=True, text=True
        )
        if existing.returncode != 0:
            print("No crontab found.")
            return

        lines = [
            line for line in existing.stdout.splitlines()
            if "tools.model_gym" not in line
        ]
        new_crontab = "\n".join(lines) + "\n" if lines else ""

        subprocess.run(
            ["crontab", "-"], input=new_crontab, capture_output=True, text=True
        )
        print("Removed all model_gym scheduled workouts.")
    except FileNotFoundError:
        print("Error: crontab not available on this system", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI-SETT Model Gym — quick behavioral workout routines for language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick workout (30 probes, balanced)
  %(prog)s --provider anthropic --model claude-3-5-sonnet-20241022

  # Focus on weak areas from last run
  %(prog)s --provider openai --model gpt-4o --focus-gaps results/last.json

  # Target specific categories
  %(prog)s --provider anthropic --model claude-3-5-sonnet-20241022 \\
    --categories metacognition,pedagogy,boundaries

  # Full workout (all probes)
  %(prog)s --provider anthropic --model claude-3-5-sonnet-20241022 --full

  # Schedule nightly at 2am (off-peak power)
  %(prog)s --provider anthropic --model claude-3-5-sonnet-20241022 --schedule 02:00

  # Dry run
  %(prog)s --dry-run
        """,
    )

    parser.add_argument("--provider", help="Provider name (openai, anthropic, google, mistral, cohere)")
    parser.add_argument("--model", help="Model identifier")
    parser.add_argument("--base-url", help="Custom API base URL")
    parser.add_argument("--probes", default="probes/", help="Probe directory (default: probes/)")
    parser.add_argument("--budget", type=int, default=30, help="Max probes per workout (default: 30)")
    parser.add_argument("--tier", choices=["base", "all"], default="base",
                        help="Category tier: 'base' runs base categories only (default), 'all' includes optional")
    parser.add_argument("--categories", help="Comma-separated category filter (e.g. metacognition,pedagogy). Overrides --tier.")
    parser.add_argument("--focus-gaps", help="Previous result JSON — prioritize gap areas")
    parser.add_argument("--compare", help="Previous result JSON — show longitudinal diff")
    parser.add_argument("--full", action="store_true", help="Run all probes (ignore budget)")
    parser.add_argument("--output", default="results/", help="Output directory (default: results/)")
    parser.add_argument("--concurrency", type=int, default=8, help="Parallel API calls (default: 8)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature (default: 0.0)")
    parser.add_argument("--aggregation", choices=["any", "all", "majority"], default="majority",
                        help="Criterion aggregation mode (default: majority)")

    # Simulation
    parser.add_argument("--simulate", action="store_true", help="Enable simulated conversation probes")
    parser.add_argument("--user-provider", help="Provider for simulated user")
    parser.add_argument("--user-model", help="Model for simulated user")
    parser.add_argument("--user-base-url", help="Custom base URL for simulated user")

    # Scheduling
    parser.add_argument("--schedule", metavar="HH:MM",
                        help="Install cron job for scheduled runs (e.g. 02:00 for 2am)")
    parser.add_argument("--unschedule", action="store_true", help="Remove scheduled gym workouts")

    # Modes
    parser.add_argument("--dry-run", action="store_true", help="Show workout plan without running")
    parser.add_argument("--verbose", action="store_true", help="Detailed output")

    args = parser.parse_args()

    # --- Unschedule ---
    if args.unschedule:
        uninstall_schedule()
        return

    # --- Schedule ---
    if args.schedule:
        if not args.provider:
            parser.error("--provider is required for scheduling")
        if not args.model:
            parser.error("--model is required for scheduling")

        install_schedule(
            time_str=args.schedule,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            probes_path=args.probes,
            budget=0 if args.full else args.budget,
            output_dir=args.output,
            focus_gaps=args.focus_gaps,
            categories=args.categories,
        )
        return

    # --- Load probes ---
    all_probes = load_probes(args.probes)
    total_probes = len(all_probes)
    print(f"Loaded {total_probes} probes from {args.probes}")

    # Filter out simulated probes if --simulate not set
    if not args.simulate:
        regular = [p for p in all_probes if p.get("type") != "simulated"]
        sim_count = total_probes - len(regular)
        if sim_count > 0:
            print(f"  ({sim_count} simulated probes skipped — use --simulate to include)")
        all_probes = regular
        total_probes = len(all_probes)

    # --- Resolve tier into category list ---
    # TIER_BASE/TIER_OPTIONAL use slugs (e.g. "emotional_intelligence") but
    # sample_workout matches on CATEGORY_MAP display names (e.g. "Emotional intelligence").
    # Build a slug -> display name mapping.
    _slug_to_display: dict[str, str] = {}
    for _prefix, (_display, _) in CATEGORY_MAP.items():
        _slug = _display.lower().replace(" ", "_")
        _slug_to_display[_slug] = _display

    def _tier_to_display_names(tier_slugs: set[str]) -> list[str]:
        return [_slug_to_display.get(s, s) for s in tier_slugs]

    if args.categories:
        # --categories overrides --tier; merge with base if tier=base
        cat_list = [c.strip() for c in args.categories.split(",")]
        if args.tier == "base":
            cat_list = list(set(cat_list) | set(_tier_to_display_names(TIER_BASE)))
    elif args.tier == "base":
        cat_list = _tier_to_display_names(TIER_BASE)
    else:
        cat_list = None  # all categories

    effective_budget = 0 if args.full else args.budget

    workout = sample_workout(
        probes=all_probes,
        budget=effective_budget,
        categories=cat_list,
        gap_result_path=args.focus_gaps,
    )

    sampling = "full" if args.full else ("gap-focused" if args.focus_gaps else "balanced")

    # Show workout plan
    groups = _group_by_category(workout)
    print(f"\nWorkout plan: {len(workout)} probes ({sampling} sampling)")
    for cat_name in sorted(groups):
        cat_probes = groups[cat_name]
        criteria = set()
        for p in cat_probes:
            criteria.update(p.get("criteria_tested", []))
        print(f"  {cat_name}: {len(cat_probes)} probes, {len(criteria)} criteria")

    if args.dry_run:
        print("\n--- Dry run complete ---")
        if args.verbose:
            for p in workout:
                pid = p.get("id", "?")
                name = p.get("name", "?")
                ptype = p.get("type", "single_turn")
                criteria = p.get("criteria_tested", [])
                print(f"  [{ptype}] {pid}: {name} ({', '.join(criteria)})")
        return

    # --- Validate provider ---
    if not args.provider:
        parser.error("--provider is required for live runs")
    if not args.model:
        parser.error("--model is required for live runs")

    env_var = f"{args.provider.upper()}_API_KEY"
    api_key = os.environ.get(env_var, "")
    if not api_key:
        print(f"Error: Set {env_var} environment variable", file=sys.stderr)
        sys.exit(1)

    provider = get_provider(args.provider, api_key=api_key, base_url=args.base_url)
    print(f"\nProvider: {args.provider} | Model: {args.model} | "
          f"Temperature: {args.temperature} | Concurrency: {args.concurrency}")

    # --- Run workout ---
    from concurrent.futures import ThreadPoolExecutor, as_completed

    start_time = time.time()
    probe_results = []

    # Import simulation support if needed
    if args.simulate:
        from tools.conversation_simulator import (
            run_simulation,
            simulation_result_to_probe_result,
        )
        if not args.user_provider or not args.user_model:
            parser.error("--user-provider and --user-model required with --simulate")

        user_env_var = f"{args.user_provider.upper()}_API_KEY"
        user_api_key = os.environ.get(user_env_var, "")
        if not user_api_key:
            print(f"Error: Set {user_env_var} environment variable", file=sys.stderr)
            sys.exit(1)

        user_provider = get_provider(args.user_provider, api_key=user_api_key,
                                     base_url=args.user_base_url)

    def _run_one(probe):
        if probe.get("type") == "simulated" and args.simulate:
            sim_result = run_simulation(
                probe=probe,
                test_provider=provider,
                test_model=args.model,
                user_provider=user_provider,
                user_model=args.user_model,
                test_temperature=args.temperature,
                user_temperature=0.7,
                verbose=False,
            )
            return simulation_result_to_probe_result(sim_result, probe)
        return run_probe(probe, provider, args.model, args.temperature)

    print(f"\nRunning {len(workout)} probes...")

    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(_run_one, p): p for p in workout}
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                probe_results.append(result)
                pid = result["probe_id"]
                if result.get("error"):
                    print(f"  [{i}/{len(workout)}] {pid}: ERROR - {result['error']}")
                elif args.verbose:
                    passed = sum(1 for v in result.get("criteria_results", {}).values() if v)
                    total = len(result.get("criteria_results", {}))
                    latency = result.get("latency_ms", 0)
                    print(f"  [{i}/{len(workout)}] {pid}: {passed}/{total} ({latency:.0f}ms)")
                else:
                    print(f"  [{i}/{len(workout)}] {pid}: done")
    else:
        for i, probe in enumerate(workout, 1):
            result = _run_one(probe)
            probe_results.append(result)
            pid = result["probe_id"]
            if result.get("error"):
                print(f"  [{i}/{len(workout)}] {pid}: ERROR - {result['error']}")
            else:
                print(f"  [{i}/{len(workout)}] {pid}: done")

    duration = time.time() - start_time

    # --- Build profile ---
    profile = build_profile(probe_results, aggregation=args.aggregation)
    gaps = extract_gaps(profile)
    zpd = find_zpd_candidates(profile)
    priorities = compute_priorities(profile)

    # --- Longitudinal comparison ---
    comparison = None
    if args.compare:
        with open(args.compare) as f:
            previous = json.load(f)
        current_result = {
            "model": args.model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "profile": profile,
        }
        comparison = compare_assessments(current_result, previous)

    # --- Print report ---
    print_gym_report(
        profile=profile,
        gaps=gaps,
        zpd=zpd,
        priorities=priorities,
        comparison=comparison,
        duration=duration,
        model=args.model,
        probe_count=len(workout),
        total_probes=total_probes,
        sampling=sampling,
    )

    # --- Save result ---
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
    # Sanitize model name for filename
    safe_model = args.model.replace("/", "-").replace(":", "-")
    output_path = output_dir / f"{safe_model}-{ts}.json"

    # Collect all tested criteria
    all_criteria = set()
    for p in workout:
        all_criteria.update(p.get("criteria_tested", []))

    categories_tested = set()
    for cid in all_criteria:
        prefix = _criterion_prefix(cid)
        if prefix in CATEGORY_MAP:
            categories_tested.add(CATEGORY_MAP[prefix][0])

    result_data = {
        "version": "1.0.0",
        "model": args.model,
        "provider": args.provider,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "probe_count": len(workout),
            "total_probes_available": total_probes,
            "criteria_tested": len(all_criteria),
            "categories_tested": len(categories_tested),
            "temperature": args.temperature,
            "duration_seconds": round(duration, 2),
            "assessment_runner_version": __version__,
        },
        "gym_metadata": {
            "budget": effective_budget,
            "sampling": sampling,
            "categories_targeted": cat_list,
            "focus_gaps_from": args.focus_gaps,
            "compared_to": args.compare,
        },
        "probe_results": probe_results,
        "profile": profile,
    }

    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"Result saved: {output_path}")

    # Hint for next run
    if gaps:
        print(f"\nNext run tip: --focus-gaps {output_path}")


if __name__ == "__main__":
    main()
