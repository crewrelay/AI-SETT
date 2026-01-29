"""Multi-judge scoring for AI-SETT assessment results.

Re-scores an existing assessment result using one or more LLM judges,
then computes inter-rater agreement (Cohen's kappa) when multiple judges
are used.

Usage:
    # Single judge (re-score with Sonnet)
    python -m tools.multi_judge \
        --input results/gpt4o.json \
        --judge anthropic:claude-sonnet-4-20250514 \
        --output results/gpt4o_judged.json

    # Multiple judges
    python -m tools.multi_judge \
        --input results/gpt4o.json \
        --judge anthropic:claude-sonnet-4-20250514 \
        --judge openai:gpt-4o \
        --output results/gpt4o_multi_judged.json

    # Only re-score criteria that needed heuristic evaluation
    python -m tools.multi_judge \
        --input results/gpt4o.json \
        --judge openai:gpt-4o \
        --heuristic-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Judge:
    provider: str
    model: str
    api_key: str = ""
    base_url: str | None = None


@dataclass
class JudgeVerdict:
    judge_id: str  # "provider:model"
    criterion_id: str
    probe_id: str
    passed: bool
    evidence: str


@dataclass
class AgreementReport:
    total_criteria: int = 0
    agreed: int = 0
    disagreed: int = 0
    kappa: float = 0.0
    judge_ids: list[str] = field(default_factory=list)
    disagreements: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Heuristic detection
# ---------------------------------------------------------------------------

# Rule evaluator pass conditions that ARE deterministic (no LLM needed)
DETERMINISTIC_PATTERNS = [
    "contains ",
    "does not contain",
    "word count",
    "under ",
    "not contain",
    "phrase not present",
    "no content beyond",
    "no formal language",
]


def is_heuristic(pass_condition: str) -> bool:
    """Check if a pass condition requires LLM judgment (not deterministic)."""
    lower = pass_condition.lower()
    for pattern in DETERMINISTIC_PATTERNS:
        if pattern in lower:
            return False
    return True


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_with_judge(
    judge: Judge,
    probe_id: str,
    response_text: str,
    criterion_id: str,
    criterion_spec: dict,
) -> JudgeVerdict:
    """Score a single criterion using an LLM judge."""
    from tools.evaluators.llm_evaluator import LLMEvaluator

    evaluator = LLMEvaluator(
        provider_name=judge.provider,
        model=judge.model,
        api_key=judge.api_key,
        base_url=judge.base_url,
    )

    passed, evidence = evaluator.evaluate(response_text, criterion_spec)

    return JudgeVerdict(
        judge_id=f"{judge.provider}:{judge.model}",
        criterion_id=criterion_id,
        probe_id=probe_id,
        passed=passed,
        evidence=evidence,
    )


def collect_scoring_tasks(
    result_data: dict,
    probes_path: str | None,
    heuristic_only: bool,
) -> list[dict]:
    """Build list of (probe_id, response, criterion_id, criterion_spec) to score.

    Loads probe YAML to get evaluation specs (the result JSON has responses
    but not the original evaluation criteria).
    """
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    # Build lookup: probe_id -> evaluation specs
    probe_specs: dict[str, dict] = {}

    if probes_path:
        probe_path = Path(probes_path)
        yaml_files = []
        if probe_path.is_file():
            yaml_files = [probe_path]
        elif probe_path.is_dir():
            yaml_files = sorted(probe_path.glob("**/*.yaml"))

        for yf in yaml_files:
            if "_generated" in yf.name:
                continue
            with open(yf) as f:
                for doc in yaml.safe_load_all(f):
                    if not doc:
                        continue
                    pid = doc.get("id", "")
                    if pid and doc.get("evaluation"):
                        probe_specs[pid] = doc["evaluation"]

    # Build scoring tasks
    tasks = []
    for probe_result in result_data.get("probe_results", []):
        probe_id = probe_result.get("probe_id", "")
        response = probe_result.get("response", "")

        if not response or probe_result.get("error"):
            continue

        eval_specs = probe_specs.get(probe_id, {})
        if not eval_specs:
            # Fallback: use criteria from the result itself
            for cid in probe_result.get("criteria_results", {}):
                tasks.append({
                    "probe_id": probe_id,
                    "response": response,
                    "criterion_id": cid,
                    "spec": {"check": cid, "pass": "Criterion demonstrated"},
                })
            continue

        for cid, spec in eval_specs.items():
            if heuristic_only and not is_heuristic(spec.get("pass", "")):
                continue
            tasks.append({
                "probe_id": probe_id,
                "response": response,
                "criterion_id": cid,
                "spec": spec,
            })

    return tasks


# ---------------------------------------------------------------------------
# Agreement computation
# ---------------------------------------------------------------------------

def compute_kappa(verdicts_by_judge: dict[str, dict[str, bool]]) -> AgreementReport:
    """Compute Cohen's kappa for inter-rater agreement.

    For >2 judges, computes pairwise kappa and averages (Fleiss approximation).
    """
    judge_ids = sorted(verdicts_by_judge.keys())

    if len(judge_ids) < 2:
        # Single judge — no agreement to compute
        single = judge_ids[0] if judge_ids else "none"
        verdicts = verdicts_by_judge.get(single, {})
        return AgreementReport(
            total_criteria=len(verdicts),
            agreed=len(verdicts),
            disagreed=0,
            kappa=1.0,
            judge_ids=judge_ids,
            disagreements=[],
        )

    # Get all criteria scored by ALL judges
    all_criteria = set.intersection(
        *[set(v.keys()) for v in verdicts_by_judge.values()]
    )

    agreed = 0
    disagreed = 0
    disagreements = []

    for cid in sorted(all_criteria):
        votes = [verdicts_by_judge[j][cid] for j in judge_ids]
        if all(v == votes[0] for v in votes):
            agreed += 1
        else:
            disagreed += 1
            disagreements.append({
                "criterion_id": cid,
                "votes": {j: verdicts_by_judge[j][cid] for j in judge_ids},
            })

    total = agreed + disagreed

    # Compute Cohen's kappa (for 2 judges) or average pairwise kappa
    kappa_values = []

    for i in range(len(judge_ids)):
        for k in range(i + 1, len(judge_ids)):
            j1, j2 = judge_ids[i], judge_ids[k]
            v1 = verdicts_by_judge[j1]
            v2 = verdicts_by_judge[j2]
            common = set(v1.keys()) & set(v2.keys())

            if not common:
                continue

            # Build contingency counts
            n = len(common)
            a = 0  # both pass
            b = 0  # j1 pass, j2 fail
            c = 0  # j1 fail, j2 pass
            d = 0  # both fail

            for cid in common:
                p1, p2 = v1[cid], v2[cid]
                if p1 and p2:
                    a += 1
                elif p1 and not p2:
                    b += 1
                elif not p1 and p2:
                    c += 1
                else:
                    d += 1

            # Observed agreement
            po = (a + d) / n if n > 0 else 0

            # Expected agreement by chance
            pe = (
                ((a + b) / n) * ((a + c) / n)
                + ((c + d) / n) * ((b + d) / n)
            ) if n > 0 else 0

            # Kappa
            if pe == 1.0:
                kappa = 1.0
            else:
                kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

            kappa_values.append(kappa)

    avg_kappa = sum(kappa_values) / len(kappa_values) if kappa_values else 0.0

    return AgreementReport(
        total_criteria=total,
        agreed=agreed,
        disagreed=disagreed,
        kappa=round(avg_kappa, 4),
        judge_ids=judge_ids,
        disagreements=disagreements,
    )


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_multi_judge(
    result_data: dict,
    judges: list[Judge],
    probes_path: str | None = None,
    heuristic_only: bool = False,
    concurrency: int = 2,
    verbose: bool = False,
) -> dict:
    """Re-score assessment results with multiple judges.

    Returns enriched result dict with judge verdicts and agreement report.
    """
    tasks = collect_scoring_tasks(result_data, probes_path, heuristic_only)

    if not tasks:
        print("No criteria to score.", file=sys.stderr)
        return result_data

    total_calls = len(tasks) * len(judges)
    print(f"Scoring {len(tasks)} criteria with {len(judges)} judge(s) = {total_calls} API calls")

    # Collect all verdicts
    all_verdicts: list[JudgeVerdict] = []
    completed = 0
    start_time = time.time()

    def _score_one(judge, task):
        return score_with_judge(
            judge=judge,
            probe_id=task["probe_id"],
            response_text=task["response"],
            criterion_id=task["criterion_id"],
            criterion_spec=task["spec"],
        )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for judge in judges:
            for task in tasks:
                future = executor.submit(_score_one, judge, task)
                futures[future] = (judge, task)

        for future in as_completed(futures):
            completed += 1
            try:
                verdict = future.result()
                all_verdicts.append(verdict)
                if verbose:
                    status = "PASS" if verdict.passed else "FAIL"
                    print(f"  [{completed}/{total_calls}] {verdict.judge_id} | {verdict.criterion_id}: {status}")
            except Exception as e:
                judge, task = futures[future]
                print(f"  [{completed}/{total_calls}] ERROR {judge.provider}:{judge.model} | {task['criterion_id']}: {e}",
                      file=sys.stderr)

    elapsed = time.time() - start_time
    print(f"Completed {completed} judgments in {elapsed:.1f}s")

    # Organize verdicts by judge -> criterion -> bool
    verdicts_by_judge: dict[str, dict[str, bool]] = {}
    evidence_by_judge: dict[str, dict[str, str]] = {}

    for v in all_verdicts:
        verdicts_by_judge.setdefault(v.judge_id, {})[v.criterion_id] = v.passed
        evidence_by_judge.setdefault(v.judge_id, {})[v.criterion_id] = v.evidence

    # Compute agreement
    agreement = compute_kappa(verdicts_by_judge)

    # Build consensus (majority vote across judges)
    all_criteria = set()
    for verdicts in verdicts_by_judge.values():
        all_criteria.update(verdicts.keys())

    consensus: dict[str, bool] = {}
    for cid in sorted(all_criteria):
        votes = [verdicts_by_judge[j].get(cid) for j in verdicts_by_judge if cid in verdicts_by_judge[j]]
        votes = [v for v in votes if v is not None]
        if votes:
            consensus[cid] = sum(votes) > len(votes) / 2

    # Enrich result data
    judge_output = {
        "judges": [f"{j.provider}:{j.model}" for j in judges],
        "scoring_mode": "heuristic_only" if heuristic_only else "all_criteria",
        "criteria_scored": len(tasks),
        "total_api_calls": total_calls,
        "duration_seconds": round(elapsed, 2),
        "verdicts_by_judge": {
            jid: {cid: {"passed": p, "evidence": evidence_by_judge.get(jid, {}).get(cid, "")}
                  for cid, p in verdicts.items()}
            for jid, verdicts in verdicts_by_judge.items()
        },
        "consensus": consensus,
        "agreement": {
            "total_criteria": agreement.total_criteria,
            "agreed": agreement.agreed,
            "disagreed": agreement.disagreed,
            "cohens_kappa": agreement.kappa,
            "interpretation": _interpret_kappa(agreement.kappa),
            "disagreements": agreement.disagreements[:20],  # cap for readability
        },
    }

    result_data["judge_scoring"] = judge_output

    # Optionally override the profile with consensus
    if consensus:
        # Update probe_results with consensus verdicts
        for probe_result in result_data.get("probe_results", []):
            for cid in probe_result.get("criteria_results", {}):
                if cid in consensus:
                    probe_result["criteria_results"][cid] = consensus[cid]

    return result_data


def _interpret_kappa(kappa: float) -> str:
    """Interpret Cohen's kappa value."""
    if kappa < 0:
        return "less than chance agreement"
    elif kappa < 0.21:
        return "slight agreement"
    elif kappa < 0.41:
        return "fair agreement"
    elif kappa < 0.61:
        return "moderate agreement"
    elif kappa < 0.81:
        return "substantial agreement"
    else:
        return "almost perfect agreement"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ENV_KEY_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
}


def parse_judge(spec: str) -> Judge:
    """Parse 'provider:model' into a Judge with API key from env."""
    if ":" not in spec:
        raise ValueError(f"Judge spec must be 'provider:model', got '{spec}'")

    provider, model = spec.split(":", 1)
    env_var = ENV_KEY_MAP.get(provider, f"{provider.upper()}_API_KEY")
    api_key = os.environ.get(env_var, "")

    return Judge(provider=provider, model=model, api_key=api_key)


def main():
    parser = argparse.ArgumentParser(
        description="AI-SETT Multi-Judge Scorer — re-score results with one or more LLM judges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single judge
  %(prog)s --input results/gpt4o.json \\
    --judge anthropic:claude-sonnet-4-20250514

  # Multiple judges with agreement analysis
  %(prog)s --input results/gpt4o.json \\
    --judge anthropic:claude-sonnet-4-20250514 \\
    --judge openai:gpt-4o \\
    --output results/gpt4o_judged.json

  # Only re-score heuristic criteria (skip deterministic ones)
  %(prog)s --input results/gpt4o.json \\
    --judge openai:gpt-4o-mini \\
    --heuristic-only

  # View agreement report only
  %(prog)s --input results/gpt4o_judged.json --report
        """,
    )
    parser.add_argument("--input", required=True, help="Path to assessment result JSON")
    parser.add_argument("--judge", action="append", dest="judges",
                        help="Judge spec as 'provider:model' (can repeat for multiple judges)")
    parser.add_argument("--probes", help="Path to probe YAML files (needed for evaluation specs)")
    parser.add_argument("--output", help="Output path (default: overwrite input)")
    parser.add_argument("--heuristic-only", action="store_true",
                        help="Only re-score criteria that need heuristic/LLM evaluation")
    parser.add_argument("--concurrency", type=int, default=2, help="Max concurrent API calls (default: 2)")
    parser.add_argument("--verbose", action="store_true", help="Print each verdict")
    parser.add_argument("--report", action="store_true",
                        help="Print agreement report from a previously judged result (no API calls)")

    args = parser.parse_args()

    # Load result
    with open(args.input) as f:
        result_data = json.load(f)

    # Report mode: just print existing judge data
    if args.report:
        judge_data = result_data.get("judge_scoring")
        if not judge_data:
            print("No judge scoring data found in this result file.", file=sys.stderr)
            sys.exit(1)

        agreement = judge_data["agreement"]
        print(f"Judges: {', '.join(judge_data['judges'])}")
        print(f"Criteria scored: {judge_data['criteria_scored']}")
        print(f"Mode: {judge_data['scoring_mode']}")
        print()
        print(f"Agreement: {agreement['agreed']}/{agreement['total_criteria']} "
              f"({100 * agreement['agreed'] / agreement['total_criteria']:.1f}%)" if agreement['total_criteria'] else "N/A")
        print(f"Cohen's kappa: {agreement['cohens_kappa']} ({agreement['interpretation']})")
        print()

        if agreement["disagreements"]:
            print(f"Disagreements ({agreement['disagreed']}):")
            for d in agreement["disagreements"]:
                votes_str = ", ".join(f"{j}: {'PASS' if v else 'FAIL'}" for j, v in d["votes"].items())
                print(f"  {d['criterion_id']}: {votes_str}")

        sys.exit(0)

    # Scoring mode
    if not args.judges:
        parser.error("At least one --judge is required (e.g., --judge openai:gpt-4o)")

    judges = [parse_judge(spec) for spec in args.judges]

    # Auto-detect probes path
    probes_path = args.probes
    if not probes_path:
        candidates = [Path("probes"), Path("../probes")]
        for c in candidates:
            if c.is_dir():
                probes_path = str(c)
                break

    if not probes_path:
        print("Warning: no --probes path specified and 'probes/' not found. "
              "Evaluation specs will be limited.", file=sys.stderr)

    result_data = run_multi_judge(
        result_data=result_data,
        judges=judges,
        probes_path=probes_path,
        heuristic_only=args.heuristic_only,
        concurrency=args.concurrency,
        verbose=args.verbose,
    )

    # Save
    output_path = args.output or args.input
    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to {output_path}")

    # Print summary
    judge_data = result_data.get("judge_scoring", {})
    agreement = judge_data.get("agreement", {})
    if agreement.get("total_criteria"):
        print(f"\nAgreement: {agreement['agreed']}/{agreement['total_criteria']} "
              f"({100 * agreement['agreed'] / agreement['total_criteria']:.1f}%)")
        print(f"Cohen's kappa: {agreement.get('cohens_kappa', 'N/A')} "
              f"({agreement.get('interpretation', '')})")

        if agreement.get("disagreed", 0) > 0:
            print(f"\n{agreement['disagreed']} disagreements — see output file for details")


if __name__ == "__main__":
    main()
