#!/usr/bin/env python3
"""AI-SETT Gap Analyzer

Identifies gap patterns, clusters, intervention priorities, ZPD candidates,
and supports longitudinal comparison between assessment runs.

Usage:
    python -m tools.gap_analyzer --input results/gpt4o.json --priorities
    python -m tools.gap_analyzer --input results/v2.json --compare results/v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_result(path: str) -> dict:
    """Load an assessment result JSON file."""
    with open(path) as f:
        data = json.load(f)
    if data.get("version") != "1.0.0":
        print(f"Warning: expected version 1.0.0, got {data.get('version')}", file=sys.stderr)
    return data


# ---------------------------------------------------------------------------
# Gap extraction
# ---------------------------------------------------------------------------

def extract_gaps(profile: dict) -> list[dict]:
    """Extract all gap criteria from a profile.

    Returns list of {criterion_id, category, subcategory}.
    """
    gaps = []
    for cat_name, cat in profile.items():
        for sub_name, sub in cat.get("subcategories", {}).items():
            for cid, demonstrated in sub.get("criteria", {}).items():
                if not demonstrated:
                    gaps.append({
                        "criterion_id": cid,
                        "category": cat_name,
                        "subcategory": sub_name,
                    })
    return gaps


def extract_demonstrated(profile: dict) -> list[dict]:
    """Extract all demonstrated criteria from a profile."""
    demonstrated = []
    for cat_name, cat in profile.items():
        for sub_name, sub in cat.get("subcategories", {}).items():
            for cid, demo in sub.get("criteria", {}).items():
                if demo:
                    demonstrated.append({
                        "criterion_id": cid,
                        "category": cat_name,
                        "subcategory": sub_name,
                    })
    return demonstrated


# ---------------------------------------------------------------------------
# Gap clustering
# ---------------------------------------------------------------------------

def cluster_gaps(gaps: list[dict]) -> dict[str, dict[str, list[str]]]:
    """Cluster gaps by category -> subcategory -> [criterion_ids].

    Returns nested dict for structured display.
    """
    clusters: dict[str, dict[str, list[str]]] = {}
    for gap in gaps:
        cat = gap["category"]
        sub = gap["subcategory"]
        clusters.setdefault(cat, {}).setdefault(sub, []).append(gap["criterion_id"])
    return clusters


# ---------------------------------------------------------------------------
# Intervention priorities
# ---------------------------------------------------------------------------

def compute_priorities(profile: dict) -> list[dict]:
    """Rank subcategories by intervention priority.

    Priority heuristics:
    1. Subcategories with highest gap ratio (gaps/total) come first
    2. Among equal ratios, larger subcategories (more criteria) rank higher
       because they affect more behaviors
    3. Foundation categories (Understanding, Calibration) get a boost
       because they underpin everything else
    """
    FOUNDATION_BOOST = 0.1
    FOUNDATION_CATEGORIES = {"Understanding", "Calibration", "Boundaries"}

    priorities = []
    for cat_name, cat in profile.items():
        for sub_name, sub in cat.get("subcategories", {}).items():
            total = sub["total"]
            demonstrated = sub["demonstrated"]
            gap_count = total - demonstrated

            if total == 0:
                continue

            gap_ratio = gap_count / total
            # Boost foundation categories
            if cat_name in FOUNDATION_CATEGORIES:
                gap_ratio += FOUNDATION_BOOST

            priorities.append({
                "category": cat_name,
                "subcategory": sub_name,
                "demonstrated": demonstrated,
                "total": total,
                "gap_count": gap_count,
                "gap_ratio": round(gap_ratio, 3),
                "gap_criteria": [
                    cid for cid, demo in sub.get("criteria", {}).items() if not demo
                ],
            })

    # Sort by gap_ratio descending, then by total descending
    priorities.sort(key=lambda p: (-p["gap_ratio"], -p["total"]))
    return priorities


# ---------------------------------------------------------------------------
# ZPD (Zone of Proximal Development) candidates
# ---------------------------------------------------------------------------

def find_zpd_candidates(profile: dict) -> list[dict]:
    """Identify ZPD candidates — subcategories where the model is close to
    demonstrating but has a few gaps. These are the most productive targets
    for intervention.

    A good ZPD candidate has:
    - At least 1 demonstrated criterion (not starting from zero)
    - At least 1 gap (not already complete)
    - Gap ratio between 0.1 and 0.6 (partial mastery)
    """
    candidates = []
    for cat_name, cat in profile.items():
        for sub_name, sub in cat.get("subcategories", {}).items():
            total = sub["total"]
            demonstrated = sub["demonstrated"]
            gap_count = total - demonstrated

            if total == 0 or demonstrated == 0 or gap_count == 0:
                continue

            gap_ratio = gap_count / total
            if 0.1 <= gap_ratio <= 0.6:
                candidates.append({
                    "category": cat_name,
                    "subcategory": sub_name,
                    "demonstrated": demonstrated,
                    "total": total,
                    "gap_count": gap_count,
                    "gap_ratio": round(gap_ratio, 3),
                    "gap_criteria": [
                        cid for cid, demo in sub.get("criteria", {}).items() if not demo
                    ],
                })

    candidates.sort(key=lambda c: c["gap_ratio"])
    return candidates


# ---------------------------------------------------------------------------
# Longitudinal comparison
# ---------------------------------------------------------------------------

def compare_assessments(current: dict, previous: dict) -> dict:
    """Compare two assessment results to track changes.

    Returns a change report showing:
    - Newly demonstrated criteria (gaps closed)
    - Newly gapped criteria (regressions)
    - Unchanged gaps and demonstrated
    - Per-category delta
    """
    curr_profile = current["profile"]
    prev_profile = previous["profile"]

    # Build flat criterion maps
    def _flat_criteria(profile: dict) -> dict[str, bool]:
        flat = {}
        for cat in profile.values():
            for sub in cat.get("subcategories", {}).values():
                for cid, demo in sub.get("criteria", {}).items():
                    flat[cid] = demo
        return flat

    curr_flat = _flat_criteria(curr_profile)
    prev_flat = _flat_criteria(prev_profile)

    all_criteria = set(curr_flat) | set(prev_flat)

    newly_demonstrated = []
    regressions = []
    stable_demonstrated = []
    stable_gaps = []
    new_criteria = []

    for cid in sorted(all_criteria):
        in_curr = cid in curr_flat
        in_prev = cid in prev_flat

        if in_curr and in_prev:
            if curr_flat[cid] and not prev_flat[cid]:
                newly_demonstrated.append(cid)
            elif not curr_flat[cid] and prev_flat[cid]:
                regressions.append(cid)
            elif curr_flat[cid]:
                stable_demonstrated.append(cid)
            else:
                stable_gaps.append(cid)
        elif in_curr and not in_prev:
            new_criteria.append(cid)

    # Per-category summary
    category_deltas = {}
    for cat_name in set(list(curr_profile.keys()) + list(prev_profile.keys())):
        curr_cat = curr_profile.get(cat_name, {"demonstrated": 0, "total": 0})
        prev_cat = prev_profile.get(cat_name, {"demonstrated": 0, "total": 0})
        category_deltas[cat_name] = {
            "demonstrated_delta": curr_cat["demonstrated"] - prev_cat["demonstrated"],
            "total_delta": curr_cat["total"] - prev_cat["total"],
            "current": f"{curr_cat['demonstrated']}/{curr_cat['total']}",
            "previous": f"{prev_cat['demonstrated']}/{prev_cat['total']}",
        }

    return {
        "current_model": current.get("model", "?"),
        "previous_model": previous.get("model", "?"),
        "current_timestamp": current.get("timestamp", "?"),
        "previous_timestamp": previous.get("timestamp", "?"),
        "newly_demonstrated": newly_demonstrated,
        "regressions": regressions,
        "stable_demonstrated": len(stable_demonstrated),
        "stable_gaps": stable_gaps,
        "new_criteria": new_criteria,
        "category_deltas": category_deltas,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI-SETT Gap Analyzer — identify patterns and priorities in assessment gaps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input results/gpt4o.json --priorities
  %(prog)s --input results/gpt4o.json --zpd
  %(prog)s --input results/v2.json --compare results/v1.json
  %(prog)s --input results/gpt4o.json --json
        """,
    )
    parser.add_argument("--input", required=True, help="Path to assessment result JSON")
    parser.add_argument("--compare", help="Path to previous result for longitudinal comparison")
    parser.add_argument("--priorities", action="store_true", help="Show intervention priorities")
    parser.add_argument("--zpd", action="store_true", help="Show ZPD (Zone of Proximal Development) candidates")
    parser.add_argument("--clusters", action="store_true", help="Show gap clusters")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of text")

    args = parser.parse_args()

    result = load_result(args.input)
    profile = result["profile"]

    # Default: show everything if no specific flag
    show_all = not (args.priorities or args.zpd or args.clusters or args.compare)

    output = {}

    # Gap summary
    gaps = extract_gaps(profile)
    demonstrated = extract_demonstrated(profile)
    output["summary"] = {
        "model": result.get("model", "?"),
        "total_criteria_tested": len(gaps) + len(demonstrated),
        "demonstrated": len(demonstrated),
        "gaps": len(gaps),
    }

    if args.clusters or show_all:
        output["clusters"] = cluster_gaps(gaps)

    if args.priorities or show_all:
        output["priorities"] = compute_priorities(profile)

    if args.zpd or show_all:
        output["zpd_candidates"] = find_zpd_candidates(profile)

    if args.compare:
        previous = load_result(args.compare)
        output["comparison"] = compare_assessments(result, previous)

    # Output
    if args.json:
        print(json.dumps(output, indent=2))
        return

    # Text output
    s = output["summary"]
    print(f"Model: {s['model']}")
    print(f"Criteria tested: {s['total_criteria_tested']}")
    print(f"Demonstrated: {s['demonstrated']}")
    print(f"Gaps: {s['gaps']}")

    if "clusters" in output:
        print("\n--- Gap Clusters ---")
        for cat, subs in sorted(output["clusters"].items()):
            print(f"\n  {cat}:")
            for sub, cids in sorted(subs.items()):
                print(f"    {sub}: {', '.join(cids)}")

    if "priorities" in output:
        print("\n--- Intervention Priorities ---")
        for i, p in enumerate(output["priorities"], 1):
            if p["gap_count"] == 0:
                continue
            print(f"  {i}. {p['category']} > {p['subcategory']}")
            print(f"     {p['gap_count']} gaps / {p['total']} total (ratio: {p['gap_ratio']})")
            print(f"     Gap criteria: {', '.join(p['gap_criteria'])}")

    if "zpd_candidates" in output:
        print("\n--- ZPD Candidates (most productive intervention targets) ---")
        if not output["zpd_candidates"]:
            print("  No ZPD candidates found (all subcategories are either fully demonstrated or fully gapped)")
        for c in output["zpd_candidates"]:
            print(f"  {c['category']} > {c['subcategory']}")
            print(f"    {c['demonstrated']}/{c['total']} demonstrated, {c['gap_count']} gaps to close")
            print(f"    Gaps: {', '.join(c['gap_criteria'])}")

    if "comparison" in output:
        comp = output["comparison"]
        print(f"\n--- Longitudinal Comparison ---")
        print(f"  Current:  {comp['current_model']} ({comp['current_timestamp']})")
        print(f"  Previous: {comp['previous_model']} ({comp['previous_timestamp']})")
        print(f"\n  Newly demonstrated: {len(comp['newly_demonstrated'])}")
        for cid in comp["newly_demonstrated"]:
            print(f"    + {cid}")
        print(f"  Regressions: {len(comp['regressions'])}")
        for cid in comp["regressions"]:
            print(f"    - {cid}")
        print(f"  Stable demonstrated: {comp['stable_demonstrated']}")
        print(f"  Stable gaps: {len(comp['stable_gaps'])}")
        print(f"\n  Category deltas:")
        for cat, delta in sorted(comp["category_deltas"].items()):
            sign = "+" if delta["demonstrated_delta"] > 0 else ""
            print(f"    {cat}: {delta['previous']} -> {delta['current']} ({sign}{delta['demonstrated_delta']})")


if __name__ == "__main__":
    main()
