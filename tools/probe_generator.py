#!/usr/bin/env python3
"""AI-SETT Probe Generator

Parses AI-SETT-FRAMEWORK.md to extract all criteria, then generates YAML
probe templates for criteria that lack probes. Also reports coverage.

Usage:
    python -m tools.probe_generator \
        --framework docs/AI-SETT-FRAMEWORK.md \
        --output probes/

    python -m tools.probe_generator \
        --framework docs/AI-SETT-FRAMEWORK.md \
        --coverage probes/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

from tools.assessment_runner import CATEGORY_MAP


# ---------------------------------------------------------------------------
# Framework parsing
# ---------------------------------------------------------------------------

def parse_framework(framework_path: str) -> list[dict]:
    """Parse AI-SETT-FRAMEWORK.md to extract all criteria.

    Returns list of {id, description, verification, category, subcategory, section_number}.
    """
    path = Path(framework_path)
    if not path.exists():
        print(f"Error: Framework file not found: {framework_path}", file=sys.stderr)
        sys.exit(1)

    text = path.read_text()
    criteria = []

    # Track current section context
    current_category = ""
    current_subcategory = ""
    current_section = ""

    for line in text.split("\n"):
        # Match top-level category headers: "## 1. Understanding"
        cat_match = re.match(r"^##\s+(\d+)\.\s+(.+)", line)
        if cat_match:
            current_section = cat_match.group(1)
            current_category = cat_match.group(2).strip()
            continue

        # Match subcategory headers: "### 1.1 Basic requests"
        subcat_match = re.match(r"^###\s+(\d+\.\d+)\s+(.+)", line)
        if subcat_match:
            current_subcategory = subcat_match.group(2).strip()
            continue

        # Match criterion rows: "| U.BR.01 | Description | Verification |"
        criterion_match = re.match(
            r"^\|\s*([A-Z]+\.[A-Z]{2}\.\d{2})\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|",
            line,
        )
        if criterion_match:
            cid = criterion_match.group(1)
            description = criterion_match.group(2).strip()
            verification = criterion_match.group(3).strip()
            criteria.append({
                "id": cid,
                "description": description,
                "verification": verification,
                "category": current_category,
                "subcategory": current_subcategory,
                "section": current_section,
            })

    return criteria


# ---------------------------------------------------------------------------
# Coverage analysis
# ---------------------------------------------------------------------------

def scan_existing_probes(probes_path: str) -> set[str]:
    """Scan existing probe YAML files and return the set of criterion IDs covered."""
    if yaml is None:
        print("Error: PyYAML required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    path = Path(probes_path)
    covered = set()

    if not path.exists():
        return covered

    for yaml_file in path.glob("**/*.yaml"):
        try:
            with open(yaml_file) as f:
                for doc in yaml.safe_load_all(f):
                    if doc is None:
                        continue
                    if isinstance(doc, list):
                        for probe in doc:
                            covered.update(probe.get("criteria_tested", []))
                    else:
                        covered.update(doc.get("criteria_tested", []))
        except Exception as e:
            print(f"Warning: could not parse {yaml_file}: {e}", file=sys.stderr)

    return covered


def coverage_report(criteria: list[dict], covered: set[str]) -> dict:
    """Generate a coverage report.

    Returns dict with overall stats and per-category breakdown.
    """
    total = len(criteria)
    covered_count = sum(1 for c in criteria if c["id"] in covered)
    uncovered = [c for c in criteria if c["id"] not in covered]

    # Per-category breakdown
    categories: dict[str, dict] = {}
    for c in criteria:
        cat = c["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "covered": 0, "uncovered": []}
        categories[cat]["total"] += 1
        if c["id"] in covered:
            categories[cat]["covered"] += 1
        else:
            categories[cat]["uncovered"].append(c["id"])

    return {
        "total_criteria": total,
        "covered": covered_count,
        "uncovered_count": total - covered_count,
        "coverage_pct": round(covered_count / total * 100, 1) if total > 0 else 0,
        "categories": categories,
        "uncovered_criteria": uncovered,
    }


# ---------------------------------------------------------------------------
# Probe template generation
# ---------------------------------------------------------------------------

def _criterion_to_directory(criterion: dict) -> str:
    """Map a criterion to a probe directory name."""
    cat = criterion["category"].lower()
    # Normalize separators that would create subdirectories
    for ch in "/\\":
        cat = cat.replace(ch, "-")
    cat = cat.replace(" & ", "-").replace(" ", "-")
    return cat


def _criterion_to_probe_id(cid: str) -> str:
    """Convert criterion ID to probe ID (e.g. U.BR.01 -> probe_U_BR_01)."""
    return "probe_" + cid.replace(".", "_")


def _yaml_escape(s: str) -> str:
    """Escape a string for safe use in double-quoted YAML values."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _generate_single_probe_yaml(criterion: dict) -> str:
    """Generate a YAML probe template for a single criterion."""
    cid = criterion["id"]
    desc = _yaml_escape(criterion["description"])
    verify = _yaml_escape(criterion["verification"])
    probe_id = _criterion_to_probe_id(cid)

    # Build a template — user fills in the input and refines evaluation
    return (
        f'id: "{probe_id}"\n'
        f'name: "{desc}"\n'
        f'criteria_tested:\n'
        f'  - "{cid}"\n'
        f'\n'
        f'# TODO: Write a prompt that tests this criterion\n'
        f'input: ""\n'
        f'\n'
        f'expected_behaviors:\n'
        f'  - "{verify}"\n'
        f'\n'
        f'anti_patterns:\n'
        f'  - "TODO: Define what failure looks like"\n'
        f'\n'
        f'evaluation:\n'
        f'  {cid}:\n'
        f'    check: "{desc}"\n'
        f'    pass: "{verify}"\n'
    )


def generate_probe_templates(
    criteria: list[dict],
    covered: set[str],
    output_dir: str,
    overwrite: bool = False,
) -> dict:
    """Generate YAML probe templates for uncovered criteria.

    Groups probes by category into subdirectory files.
    Returns stats on what was generated.
    """
    if yaml is None:
        print("Error: PyYAML required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    output_path = Path(output_dir)
    uncovered = [c for c in criteria if c["id"] not in covered]

    # Group by directory (category)
    by_dir: dict[str, list[dict]] = {}
    for c in uncovered:
        dirname = _criterion_to_directory(c)
        by_dir.setdefault(dirname, []).append(c)

    files_written = []
    criteria_generated = 0

    for dirname, crit_list in sorted(by_dir.items()):
        dir_path = output_path / dirname
        dir_path.mkdir(parents=True, exist_ok=True)

        # Group by subcategory within directory
        by_subcat: dict[str, list[dict]] = {}
        for c in crit_list:
            sub = c["subcategory"].lower()
            for ch in "/\\":
                sub = sub.replace(ch, "-")
            sub = sub.replace(" & ", "_").replace(" ", "_")
            by_subcat.setdefault(sub, []).append(c)

        for subcat_slug, sub_criteria in sorted(by_subcat.items()):
            filename = f"{subcat_slug}_generated.yaml"
            filepath = dir_path / filename

            if filepath.exists() and not overwrite:
                continue

            # Generate multi-document YAML
            docs = []
            for c in sorted(sub_criteria, key=lambda x: x["id"]):
                docs.append(_generate_single_probe_yaml(c))
                criteria_generated += 1

            content = "\n---\n".join(docs)
            filepath.write_text(content)
            files_written.append(str(filepath))

    return {
        "files_written": files_written,
        "criteria_generated": criteria_generated,
        "directories": list(by_dir.keys()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI-SETT Probe Generator — parse framework and generate YAML templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show coverage of existing probes
  %(prog)s --framework docs/AI-SETT-FRAMEWORK.md --coverage probes/

  # Generate templates for uncovered criteria
  %(prog)s --framework docs/AI-SETT-FRAMEWORK.md --output probes/

  # Overwrite existing generated templates
  %(prog)s --framework docs/AI-SETT-FRAMEWORK.md --output probes/ --overwrite
        """,
    )
    parser.add_argument("--framework", required=True, help="Path to AI-SETT-FRAMEWORK.md")
    parser.add_argument("--output", help="Output directory for generated probe templates")
    parser.add_argument("--coverage", help="Probes directory to check coverage against")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated files")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if not args.output and not args.coverage:
        parser.error("Specify --output (to generate) or --coverage (to report) or both")

    # Parse framework
    criteria = parse_framework(args.framework)
    print(f"Parsed {len(criteria)} criteria from framework", file=sys.stderr)

    # Gather covered criteria
    covered = set()
    if args.coverage:
        covered = scan_existing_probes(args.coverage)
    elif args.output:
        # Also scan output dir for existing coverage
        covered = scan_existing_probes(args.output)

    # Coverage report
    if args.coverage:
        report = coverage_report(criteria, covered)

        if args.json:
            # Simplify for JSON output
            json_report = {
                "total_criteria": report["total_criteria"],
                "covered": report["covered"],
                "uncovered": report["uncovered_count"],
                "coverage_pct": report["coverage_pct"],
                "categories": {
                    cat: {"total": d["total"], "covered": d["covered"], "uncovered": d["uncovered"]}
                    for cat, d in report["categories"].items()
                },
            }
            import json
            print(json.dumps(json_report, indent=2))
        else:
            print(f"\n--- Coverage Report ---")
            print(f"Total criteria: {report['total_criteria']}")
            print(f"Covered: {report['covered']} ({report['coverage_pct']}%)")
            print(f"Uncovered: {report['uncovered_count']}")
            print()
            for cat in sorted(report["categories"]):
                d = report["categories"][cat]
                pct = round(d["covered"] / d["total"] * 100, 1) if d["total"] > 0 else 0
                print(f"  {cat}: {d['covered']}/{d['total']} ({pct}%)")
                if d["uncovered"] and len(d["uncovered"]) <= 10:
                    print(f"    Missing: {', '.join(d['uncovered'])}")
                elif d["uncovered"]:
                    print(f"    Missing: {', '.join(d['uncovered'][:5])}... +{len(d['uncovered'])-5} more")

    # Generate templates
    if args.output:
        stats = generate_probe_templates(criteria, covered, args.output, overwrite=args.overwrite)
        print(f"\n--- Generation Results ---", file=sys.stderr)
        print(f"Templates generated: {stats['criteria_generated']}", file=sys.stderr)
        print(f"Files written: {len(stats['files_written'])}", file=sys.stderr)
        for f in stats["files_written"]:
            print(f"  {f}", file=sys.stderr)


if __name__ == "__main__":
    main()
