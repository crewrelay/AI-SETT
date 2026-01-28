#!/usr/bin/env python3
"""AI-SETT Profile Visualizer

Generates self-contained HTML reports with SVG radar charts and heatmaps.
No JavaScript dependencies — pure SVG geometry and inline CSS.

Usage:
    python -m tools.profile_visualizer --input results/gpt4o.json --output results/profile.html
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from tools import __version__


def load_result(path: str) -> dict:
    """Load an assessment result JSON file."""
    with open(path) as f:
        data = json.load(f)
    if data.get("version") != "1.0.0":
        print(f"Warning: expected version 1.0.0, got {data.get('version')}", file=sys.stderr)
    return data


# ---------------------------------------------------------------------------
# SVG Radar Chart
# ---------------------------------------------------------------------------

def generate_radar_svg(profile: dict, size: int = 400) -> str:
    """Generate an SVG radar chart showing demonstrated ratio per category.

    Each axis = one category. The polygon shows the demonstrated/total ratio.
    """
    categories = sorted(profile.keys())
    n = len(categories)

    if n < 3:
        return '<text x="200" y="200" fill="#8b949e" text-anchor="middle">Need 3+ categories for radar chart</text>'

    cx, cy = size // 2, size // 2
    radius = size // 2 - 60

    # Calculate angles (evenly spaced, starting from top)
    angles = [(-math.pi / 2 + 2 * math.pi * i / n) for i in range(n)]

    # Grid rings
    rings_svg = []
    for ring in [0.25, 0.5, 0.75, 1.0]:
        r = radius * ring
        points = " ".join(
            f"{cx + r * math.cos(a):.1f},{cy + r * math.sin(a):.1f}"
            for a in angles
        )
        rings_svg.append(
            f'<polygon points="{points}" fill="none" stroke="#30363d" stroke-width="1" />'
        )

    # Axis lines and labels
    axes_svg = []
    for i, (cat, angle) in enumerate(zip(categories, angles)):
        ex = cx + radius * math.cos(angle)
        ey = cy + radius * math.sin(angle)
        axes_svg.append(
            f'<line x1="{cx}" y1="{cy}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="#30363d" stroke-width="1" />'
        )

        # Label position (pushed out past the ring)
        lx = cx + (radius + 30) * math.cos(angle)
        ly = cy + (radius + 30) * math.sin(angle)

        # Text anchor based on position
        if abs(math.cos(angle)) < 0.1:
            anchor = "middle"
        elif math.cos(angle) > 0:
            anchor = "start"
        else:
            anchor = "end"

        # Truncate long names
        label = cat[:18]
        axes_svg.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" fill="#8b949e" font-size="11" '
            f'text-anchor="{anchor}" dominant-baseline="middle">{label}</text>'
        )

    # Data polygon
    data_points = []
    for cat, angle in zip(categories, angles):
        cat_data = profile[cat]
        total = cat_data["total"]
        ratio = cat_data["demonstrated"] / total if total > 0 else 0
        r = radius * ratio
        data_points.append(f"{cx + r * math.cos(angle):.1f},{cy + r * math.sin(angle):.1f}")

    data_poly = (
        f'<polygon points="{" ".join(data_points)}" '
        f'fill="rgba(88,166,255,0.2)" stroke="#58a6ff" stroke-width="2" />'
    )

    # Data points (dots)
    dots_svg = []
    for cat, angle in zip(categories, angles):
        cat_data = profile[cat]
        total = cat_data["total"]
        ratio = cat_data["demonstrated"] / total if total > 0 else 0
        r = radius * ratio
        px = cx + r * math.cos(angle)
        py = cy + r * math.sin(angle)
        dots_svg.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="#58a6ff" />'
        )

    svg = (
        f'<svg viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">\n'
        + "\n".join(rings_svg) + "\n"
        + "\n".join(axes_svg) + "\n"
        + data_poly + "\n"
        + "\n".join(dots_svg) + "\n"
        + "</svg>"
    )
    return svg


# ---------------------------------------------------------------------------
# Heatmap cells
# ---------------------------------------------------------------------------

def generate_heatmap_cells(profile: dict) -> str:
    """Generate HTML heatmap cells for each subcategory."""
    cells = []

    for cat_name in sorted(profile):
        cat = profile[cat_name]
        for sub_name in sorted(cat.get("subcategories", {})):
            sub = cat["subcategories"][sub_name]
            total = sub["total"]
            demo = sub["demonstrated"]
            ratio = demo / total if total > 0 else 0
            pct = round(ratio * 100)

            # Color class
            if ratio >= 0.8:
                bar_class = "bar-green"
            elif ratio >= 0.5:
                bar_class = "bar-yellow"
            else:
                bar_class = "bar-red"

            cell = (
                f'<div class="heatmap-cell">'
                f'<div class="label">{sub_name}</div>'
                f'<div class="sub-label">{cat_name} — {demo}/{total} ({pct}%)</div>'
                f'<div class="bar-bg"><div class="bar-fill {bar_class}" style="width:{pct}%"></div></div>'
                f'</div>'
            )
            cells.append(cell)

    return "\n".join(cells)


# ---------------------------------------------------------------------------
# Gap table
# ---------------------------------------------------------------------------

def generate_gap_table(profile: dict) -> str:
    """Generate an HTML table of all gap criteria."""
    rows = []
    for cat_name in sorted(profile):
        cat = profile[cat_name]
        for sub_name in sorted(cat.get("subcategories", {})):
            sub = cat["subcategories"][sub_name]
            for cid in sorted(sub.get("criteria", {})):
                if not sub["criteria"][cid]:
                    rows.append(
                        f'<tr>'
                        f'<td><code>{cid}</code></td>'
                        f'<td>{cat_name}</td>'
                        f'<td>{sub_name}</td>'
                        f'<td><span class="tag tag-gap">gap</span></td>'
                        f'</tr>'
                    )

    if not rows:
        return '<p style="color: var(--green);">No gaps identified in tested criteria.</p>'

    return (
        f'<table>'
        f'<thead><tr><th>Criterion</th><th>Category</th><th>Subcategory</th><th>Status</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        f'</table>'
    )


def generate_demonstrated_table(profile: dict) -> str:
    """Generate an HTML table of demonstrated criteria by category."""
    rows = []
    for cat_name in sorted(profile):
        cat = profile[cat_name]
        total = cat["total"]
        demo = cat["demonstrated"]
        ratio = demo / total if total > 0 else 0
        pct = round(ratio * 100)

        if ratio >= 0.8:
            tag_class = "tag-demo"
            tag_text = "strong"
        elif ratio >= 0.5:
            tag_class = "tag-partial"
            tag_text = "partial"
        else:
            tag_class = "tag-gap"
            tag_text = "needs work"

        rows.append(
            f'<tr>'
            f'<td>{cat_name}</td>'
            f'<td>{demo}/{total}</td>'
            f'<td>{pct}%</td>'
            f'<td><span class="tag {tag_class}">{tag_text}</span></td>'
            f'</tr>'
        )

    return (
        f'<table>'
        f'<thead><tr><th>Category</th><th>Demonstrated/Total</th><th>Ratio</th><th>Status</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        f'</table>'
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(result: dict) -> str:
    """Generate a complete self-contained HTML report."""
    template_path = Path(__file__).parent / "templates" / "report.html"
    with open(template_path) as f:
        template = f.read()

    profile = result["profile"]
    meta = result.get("metadata", {})

    # Generate components
    radar_svg = generate_radar_svg(profile)
    heatmap_cells = generate_heatmap_cells(profile)
    gap_table = generate_gap_table(profile)
    demonstrated_table = generate_demonstrated_table(profile)

    # Template substitution (simple {{ var }} replacement)
    html = template
    replacements = {
        "{{ model }}": result.get("model", "Unknown"),
        "{{ provider }}": result.get("provider", "Unknown"),
        "{{ timestamp }}": result.get("timestamp", "Unknown")[:10],
        "{{ probe_count }}": str(meta.get("probe_count", 0)),
        "{{ criteria_tested }}": str(meta.get("criteria_tested", 0)),
        "{{ radar_svg }}": radar_svg,
        "{{ heatmap_cells }}": heatmap_cells,
        "{{ gap_table }}": gap_table,
        "{{ demonstrated_table }}": demonstrated_table,
        "{{ version }}": __version__,
    }
    for key, value in replacements.items():
        html = html.replace(key, value)

    return html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI-SETT Profile Visualizer — generate HTML profile reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input results/gpt4o.json --output results/profile.html
  %(prog)s --input results/gpt4o.json  # outputs to stdout
        """,
    )
    parser.add_argument("--input", required=True, help="Path to assessment result JSON")
    parser.add_argument("--output", help="Output HTML path (default: stdout)")

    args = parser.parse_args()

    result = load_result(args.input)
    html = generate_report(result)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)
        print(f"Report written to: {output_path}", file=sys.stderr)
    else:
        print(html)


if __name__ == "__main__":
    main()
