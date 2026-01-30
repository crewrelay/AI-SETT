#!/usr/bin/env python3
"""AI-SETT Conversation Harvester

Extracts training examples from real Claude conversations and files them into
the AI-SETT training data library by criterion. Uses an LLM classifier for
two-stage classification: category triage then criterion-level matching.

Usage:
    # Harvest from Claude Code transcripts (main use case)
    python -m tools.conversation_harvester \
        --input docs/claude-transcripts/all-conversations.jsonl \
        --framework docs/AI-SETT-FRAMEWORK.md \
        --classifier anthropic:claude-3-5-haiku-20241022 \
        --output training_data/ --split-by-category --append

    # Dry run — show extraction stats
    python -m tools.conversation_harvester \
        --input docs/claude-transcripts/all-conversations.jsonl --dry-run

    # Single session, verbose
    python -m tools.conversation_harvester \
        --input ~/.claude/projects/-Users-Israel-crewrelay-backend/session.jsonl \
        --framework docs/AI-SETT-FRAMEWORK.md \
        --classifier anthropic:claude-3-5-haiku-20241022 --verbose

    # Higher quality threshold
    python -m tools.conversation_harvester \
        --input all-conversations.jsonl \
        --framework docs/AI-SETT-FRAMEWORK.md \
        --classifier anthropic:claude-3-5-haiku-20241022 \
        --min-quality 0.8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tools.training_data_generator import (
    CATEGORY_MAP,
    FORMATTERS,
    GeneratorSpec,
    TrainingExample,
    criterion_category,
    extract_json_array,
    init_provider,
    parse_framework,
    parse_generator_spec,
    write_manifest,
    write_split_output,
)
from tools.providers.base import CompletionRequest, Message


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TurnPair:
    """A user→assistant turn pair extracted from a conversation."""
    session_id: str
    user_text: str
    assistant_text: str
    user_uuid: str
    assistant_uuid: str
    timestamp: str
    tool_use_ratio: float = 0.0
    code_ratio: float = 0.0


@dataclass
class ClassifiedTurn:
    """A turn pair after classification."""
    turn: TurnPair
    categories: list[str]
    triage_quality: float
    criteria: list[dict]  # [{"criterion_id": ..., "quality": ..., "reasoning": ...}]


# ---------------------------------------------------------------------------
# Format detection and streaming
# ---------------------------------------------------------------------------

def detect_format(path: Path) -> str:
    """Detect whether a file is Claude Code JSONL or Claude.ai Web JSON."""
    with open(path) as f:
        first_line = ""
        for line in f:
            line = line.strip()
            if line:
                first_line = line
                break

    if not first_line:
        return "empty"

    # JSON array or object with "conversations" key → web export
    if first_line.startswith("[") or first_line.startswith("{"):
        try:
            obj = json.loads(first_line)
            if isinstance(obj, list):
                return "web_json"
            if isinstance(obj, dict):
                if "conversations" in obj:
                    return "web_json"
                if "type" in obj:
                    return "code_jsonl"
                return "web_json"
        except json.JSONDecodeError:
            pass

    # JSONL with type field → Claude Code
    try:
        obj = json.loads(first_line)
        if isinstance(obj, dict) and "type" in obj:
            return "code_jsonl"
    except json.JSONDecodeError:
        pass

    return "unknown"


def stream_code_jsonl(path: Path):
    """Stream records from a Claude Code JSONL file, line by line."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_web_json(path: Path) -> list[dict]:
    """Load conversations from a Claude.ai web export JSON file."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "conversations" in data:
        return data["conversations"]
    return [data]


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def extract_user_text(message: dict) -> str | None:
    """Extract human-readable text from a user message.

    Returns None if the message is tool_result-only (no human text).
    """
    content = message.get("content", "")

    if isinstance(content, str):
        return _strip_system_boilerplate(content)

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                # Skip tool_result blocks
        combined = " ".join(t.strip() for t in texts if t.strip())
        if not combined:
            return None
        return _strip_system_boilerplate(combined)

    return None


def extract_assistant_text(message: dict) -> tuple[str, float, float]:
    """Extract text from an assistant message.

    Returns: (text, tool_use_ratio, code_ratio)
    """
    content = message.get("content", "")

    if isinstance(content, str):
        code_ratio = _estimate_code_ratio(content)
        return content.strip(), 0.0, code_ratio

    if isinstance(content, list):
        texts = []
        total_blocks = 0
        tool_blocks = 0
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype in ("text",):
                texts.append(block.get("text", ""))
                total_blocks += 1
            elif btype == "tool_use":
                tool_blocks += 1
                total_blocks += 1
            elif btype == "thinking":
                # Skip thinking blocks — not part of visible response
                pass
            else:
                total_blocks += 1

        combined = "\n".join(t.strip() for t in texts if t.strip())
        tool_ratio = tool_blocks / total_blocks if total_blocks > 0 else 0.0
        code_ratio = _estimate_code_ratio(combined)
        return combined, tool_ratio, code_ratio

    return "", 0.0, 0.0


# Patterns to strip from user messages
_BOILERPLATE_PATTERNS = [
    re.compile(r'<system-reminder>.*?</system-reminder>', re.DOTALL),
    re.compile(r'<command-name>.*?</command-name>', re.DOTALL),
    re.compile(r'<user-prompt-submit-hook>.*?</user-prompt-submit-hook>', re.DOTALL),
]


def _strip_system_boilerplate(text: str) -> str:
    """Remove system boilerplate tags from user text."""
    for pat in _BOILERPLATE_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


def _estimate_code_ratio(text: str) -> float:
    """Estimate what fraction of text is code (indented or fenced)."""
    if not text.strip():
        return 0.0
    lines = text.split("\n")
    code_lines = 0
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            code_lines += 1
            continue
        if in_fence:
            code_lines += 1
        elif line.startswith("    ") or line.startswith("\t"):
            code_lines += 1
    return code_lines / len(lines) if lines else 0.0


# ---------------------------------------------------------------------------
# Turn reconstruction
# ---------------------------------------------------------------------------

def reconstruct_turns_code(records) -> list[TurnPair]:
    """Reconstruct user→assistant turn pairs from Claude Code JSONL records.

    Groups by sessionId. For each real user message (one with human text,
    not just tool_result), follows the entire assistant response chain —
    collecting all text blocks from all assistant messages until the next
    real user message. This captures the full substantive response even
    when Claude produces many intermediate tool_use/thinking messages.
    """
    # Group by session, keep user + assistant records
    sessions: dict[str, list[dict]] = {}
    for rec in records:
        rtype = rec.get("type")
        if rtype not in ("user", "assistant"):
            continue
        if rec.get("isSidechain"):
            continue
        sid = rec.get("sessionId", "")
        if not sid:
            continue
        sessions.setdefault(sid, []).append(rec)

    pairs: list[TurnPair] = []

    for sid, recs in sessions.items():
        # Sort by timestamp
        recs.sort(key=lambda r: r.get("timestamp", ""))

        # Find real user messages (ones with human text, not tool_result-only)
        # and collect all assistant text between them
        i = 0
        while i < len(recs):
            rec = recs[i]
            if rec.get("type") != "user":
                i += 1
                continue

            user_text = extract_user_text(rec.get("message", {}))
            if user_text is None:
                # tool_result-only user message — skip
                i += 1
                continue

            user_uuid = rec.get("uuid", "")
            user_ts = rec.get("timestamp", "")

            # Collect all assistant text blocks until the next real user msg
            all_asst_texts: list[str] = []
            total_blocks = 0
            tool_blocks = 0
            last_asst_uuid = ""
            last_asst_ts = ""

            j = i + 1
            while j < len(recs):
                r = recs[j]
                if r.get("type") == "user":
                    # Check if this is a real user message or tool_result
                    next_user_text = extract_user_text(r.get("message", {}))
                    if next_user_text is not None:
                        # Real user message — stop here
                        break
                    # tool_result — keep going
                    j += 1
                    continue

                if r.get("type") == "assistant":
                    text, tr, _ = extract_assistant_text(r.get("message", {}))
                    if text:
                        all_asst_texts.append(text)
                    # Count blocks for tool ratio
                    content = r.get("message", {}).get("content", [])
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                btype = block.get("type", "")
                                if btype == "tool_use":
                                    tool_blocks += 1
                                    total_blocks += 1
                                elif btype == "text":
                                    total_blocks += 1
                                elif btype != "thinking":
                                    total_blocks += 1
                    last_asst_uuid = r.get("uuid", "")
                    last_asst_ts = r.get("timestamp", "")

                j += 1

            i = j  # Skip to next real user message

            # Combine all assistant text
            combined_text = "\n\n".join(t for t in all_asst_texts if t)
            if not combined_text:
                continue

            tool_ratio = tool_blocks / total_blocks if total_blocks > 0 else 0.0
            code_ratio = _estimate_code_ratio(combined_text)

            pairs.append(TurnPair(
                session_id=sid,
                user_text=user_text,
                assistant_text=combined_text,
                user_uuid=user_uuid,
                assistant_uuid=last_asst_uuid,
                timestamp=last_asst_ts or user_ts,
                tool_use_ratio=tool_ratio,
                code_ratio=code_ratio,
            ))

    return pairs


def reconstruct_turns_web(conversations: list[dict]) -> list[TurnPair]:
    """Reconstruct turn pairs from Claude.ai web export JSON."""
    pairs: list[TurnPair] = []

    for conv in conversations:
        messages = conv.get("chat_messages", [])
        conv_id = conv.get("uuid", conv.get("id", "unknown"))

        i = 0
        while i < len(messages) - 1:
            msg = messages[i]
            next_msg = messages[i + 1]

            if msg.get("sender") == "human" and next_msg.get("sender") == "assistant":
                user_text = msg.get("text", "").strip()
                asst_text = next_msg.get("text", "").strip()

                if user_text and asst_text:
                    pairs.append(TurnPair(
                        session_id=conv_id,
                        user_text=user_text,
                        assistant_text=asst_text,
                        user_uuid=msg.get("uuid", ""),
                        assistant_uuid=next_msg.get("uuid", ""),
                        timestamp=next_msg.get("created_at", ""),
                        tool_use_ratio=0.0,
                        code_ratio=_estimate_code_ratio(asst_text),
                    ))
                i += 2
            else:
                i += 1

    return pairs


# ---------------------------------------------------------------------------
# Filtering (pre-classification, no API calls)
# ---------------------------------------------------------------------------

def filter_turns(
    pairs: list[TurnPair],
    min_user_words: int = 5,
    min_assistant_words: int = 20,
    max_tool_ratio: float = 0.6,
    max_code_ratio: float = 0.7,
) -> tuple[list[TurnPair], dict[str, int]]:
    """Filter turn pairs, returning (kept, filter_stats)."""
    stats: dict[str, int] = {
        "total": len(pairs),
        "short_user": 0,
        "short_assistant": 0,
        "tool_heavy": 0,
        "code_heavy": 0,
        "kept": 0,
    }

    kept: list[TurnPair] = []

    for pair in pairs:
        user_words = len(pair.user_text.split())
        asst_words = len(pair.assistant_text.split())

        if user_words < min_user_words:
            stats["short_user"] += 1
            continue
        if asst_words < min_assistant_words:
            stats["short_assistant"] += 1
            continue
        if pair.tool_use_ratio > max_tool_ratio:
            stats["tool_heavy"] += 1
            continue
        if pair.code_ratio > max_code_ratio:
            stats["code_heavy"] += 1
            continue

        kept.append(pair)

    stats["kept"] = len(kept)
    return kept, stats


# ---------------------------------------------------------------------------
# Category descriptions for triage
# ---------------------------------------------------------------------------

CATEGORY_DESCRIPTIONS = {
    "understanding": "Comprehending user requests — basic, complex, ambiguous, implicit, contextual",
    "calibration": "Response calibration — length, depth, format, tone, knowing when to stop",
    "generation": "Content generation — technical/creative/persuasive/academic/business writing, code, structured output, summarization, translation",
    "knowledge": "Domain knowledge — programming, APIs, security, devops, databases, math, science, business, legal, medical, history, arts, education, PM, design",
    "reasoning": "Logical, mathematical, causal, analogical reasoning, critical thinking, problem solving",
    "boundaries": "Appropriate refusals, avoiding over-refusal, uncertainty, professional boundaries, safety",
    "interaction": "Multi-turn coherence, error handling, clarification and repair",
    "tool_use": "Web search, code execution, file handling, API calling, computation, image understanding, tool selection",
    "emotional_intelligence": "Emotional recognition, empathetic response, difficult conversations, social awareness",
    "metacognition": "Self-awareness, learning/adaptation, strategy selection",
    "learning": "In-context learning, instruction following, domain-specific learning, error-based learning, transfer",
    "pedagogy": "Diagnostic assessment, explanation quality, scaffolding, feedback, adaptation, misconception handling, checking understanding, domain teaching",
}


# ---------------------------------------------------------------------------
# Two-stage classification
# ---------------------------------------------------------------------------

TRIAGE_SYSTEM = """You classify conversation turn pairs into AI behavioral assessment categories.

Given a batch of user→assistant turn pairs, classify each into one or more of these categories:

{categories}

For each turn pair, return:
- idx: the 0-based index in the batch
- categories: list of matching category slugs (1-3 max)
- quality: 0.0-1.0 score for how well this pair demonstrates interesting behavior (not routine coding/tool use)

Skip pairs that are pure boilerplate (file reads, tool outputs, status updates).

Return a JSON array. Example:
[{{"idx": 0, "categories": ["understanding", "calibration"], "quality": 0.8}}, ...]

Return ONLY the JSON array. No markdown fences, no explanation."""


CRITERION_SYSTEM = """You match a conversation turn pair to specific behavioral criteria.

The user→assistant exchange demonstrates behavior from the "{category}" category.
Match it to the most relevant criteria from this list:

{criteria_table}

For each matching criterion, return:
- criterion_id: the ID (e.g. "U.BR.01")
- quality: 0.0-1.0 score for how clearly the pair demonstrates this criterion
- reasoning: brief explanation (1 sentence)

Return a JSON array. Example:
[{{"criterion_id": "U.BR.01", "quality": 0.85, "reasoning": "User asks a straightforward question and assistant answers directly"}}]

Return ONLY the JSON array. No markdown fences, no explanation."""


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, adding ellipsis if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


def triage_classify(
    pairs: list[TurnPair],
    provider,
    spec: GeneratorSpec,
    batch_size: int = 10,
    verbose: bool = False,
) -> list[tuple[TurnPair, list[str], float]]:
    """Stage 1: Classify turn pairs into categories (batched).

    Returns: [(pair, categories, quality), ...]
    """
    # Build category descriptions
    cat_lines = "\n".join(
        f"- {slug}: {desc}" for slug, desc in sorted(CATEGORY_DESCRIPTIONS.items())
    )
    system_prompt = TRIAGE_SYSTEM.format(categories=cat_lines)

    results: list[tuple[TurnPair, list[str], float]] = []

    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start:batch_start + batch_size]

        # Build batch prompt
        lines = []
        for i, pair in enumerate(batch):
            user_trunc = _truncate(pair.user_text, 500)
            asst_trunc = _truncate(pair.assistant_text, 1000)
            lines.append(f"[{i}] USER: {user_trunc}\nASSISTANT: {asst_trunc}")

        user_prompt = "\n\n".join(lines)

        request = CompletionRequest(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ],
            model=spec.model,
            temperature=0.0,
            max_tokens=2048,
        )

        try:
            response = provider.complete(request)
        except Exception as e:
            print(f"  Triage batch error: {e}", file=sys.stderr)
            continue

        items = extract_json_array(response.content)
        if not items:
            if verbose:
                print(f"  No JSON from triage batch starting at {batch_start}", file=sys.stderr)
            continue

        # Map results back to pairs
        for item in items:
            if not isinstance(item, dict):
                continue
            idx = item.get("idx")
            if not isinstance(idx, int) or idx < 0 or idx >= len(batch):
                continue
            cats = item.get("categories", [])
            if not isinstance(cats, list):
                continue
            # Validate category slugs
            valid_cats = [c for c in cats if c in CATEGORY_DESCRIPTIONS]
            quality = float(item.get("quality", 0.0))
            results.append((batch[idx], valid_cats, quality))

        if verbose:
            batch_end = min(batch_start + batch_size, len(pairs))
            classified = len([i for i in items if isinstance(i, dict)])
            print(f"  Triage batch {batch_start}-{batch_end}: {classified}/{len(batch)} classified")

    return results


def criterion_classify(
    pair: TurnPair,
    categories: list[str],
    criteria_db: dict[str, dict],
    provider,
    spec: GeneratorSpec,
) -> list[dict]:
    """Stage 2: Match a turn pair to specific criteria within its categories.

    Returns: [{"criterion_id": ..., "quality": ..., "reasoning": ...}, ...]
    """
    all_matches: list[dict] = []

    for category in categories:
        # Get criteria for this category
        cat_criteria = {}
        for cid, info in criteria_db.items():
            cat, _ = criterion_category(cid)
            if cat == category:
                cat_criteria[cid] = info

        if not cat_criteria:
            continue

        # Build criteria table
        table_lines = ["| ID | Criterion | Verify |"]
        table_lines.append("|---|---|---|")
        for cid in sorted(cat_criteria):
            info = cat_criteria[cid]
            table_lines.append(
                f"| {cid} | {info.get('criterion', '')} | {info.get('verify', '')} |"
            )
        criteria_table = "\n".join(table_lines)

        system_prompt = CRITERION_SYSTEM.format(
            category=category,
            criteria_table=criteria_table,
        )

        user_prompt = (
            f"USER: {pair.user_text}\n\n"
            f"ASSISTANT: {pair.assistant_text}"
        )

        request = CompletionRequest(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ],
            model=spec.model,
            temperature=0.0,
            max_tokens=1024,
        )

        try:
            response = provider.complete(request)
        except Exception as e:
            print(f"  Criterion classify error ({category}): {e}", file=sys.stderr)
            continue

        items = extract_json_array(response.content)
        for item in items:
            if not isinstance(item, dict):
                continue
            cid = item.get("criterion_id", "")
            # Validate criterion exists in our database
            if cid not in criteria_db:
                continue
            all_matches.append({
                "criterion_id": cid,
                "quality": float(item.get("quality", 0.0)),
                "reasoning": item.get("reasoning", ""),
            })

    return all_matches


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def harvest(
    input_path: str,
    framework_path: str | None,
    classifier_spec: GeneratorSpec | None,
    output_path: str,
    output_format: str,
    split_by_category: bool,
    append: bool,
    min_quality: float,
    min_user_words: int,
    min_assistant_words: int,
    max_tool_ratio: float,
    max_code_ratio: float = 0.7,
    concurrency: int = 4,
    limit: int = 0,
    dry_run: bool = False,
    verbose: bool = False,
):
    """Main harvest pipeline."""
    path = Path(input_path)
    if not path.is_file():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Detect format
    fmt = detect_format(path)
    print(f"Input: {input_path}")
    print(f"Format: {fmt}")

    if fmt == "empty":
        print("Error: Input file is empty", file=sys.stderr)
        sys.exit(1)
    if fmt == "unknown":
        print("Error: Could not detect input format", file=sys.stderr)
        sys.exit(1)

    # Reconstruct turn pairs
    print("Reconstructing turn pairs...")
    if fmt == "code_jsonl":
        records = stream_code_jsonl(path)
        pairs = reconstruct_turns_code(records)
    else:
        conversations = load_web_json(path)
        pairs = reconstruct_turns_web(conversations)

    sessions = len(set(p.session_id for p in pairs))
    print(f"Turn pairs: {len(pairs)} from {sessions} sessions")

    # Filter
    pairs, stats = filter_turns(
        pairs,
        min_user_words=min_user_words,
        min_assistant_words=min_assistant_words,
        max_tool_ratio=max_tool_ratio,
        max_code_ratio=max_code_ratio,
    )

    print(f"\nFilter results:")
    print(f"  Total:           {stats['total']}")
    print(f"  Short user:      {stats['short_user']}")
    print(f"  Short assistant:  {stats['short_assistant']}")
    print(f"  Tool-heavy:      {stats['tool_heavy']}")
    print(f"  Code-heavy:      {stats['code_heavy']}")
    print(f"  Kept:            {stats['kept']}")

    # Apply limit
    if limit > 0 and len(pairs) > limit:
        pairs = pairs[:limit]
        print(f"  Limited to:      {limit}")

    if dry_run:
        print("\n--- Dry run: no API calls ---")
        # Show sample turns
        if verbose and pairs:
            print("\nSample turn pairs:")
            for i, pair in enumerate(pairs[:5]):
                print(f"\n  [{i}] Session: {pair.session_id[:12]}...")
                print(f"      User ({len(pair.user_text.split())}w): {_truncate(pair.user_text, 120)}")
                print(f"      Asst ({len(pair.assistant_text.split())}w): {_truncate(pair.assistant_text, 120)}")
                print(f"      Tool ratio: {pair.tool_use_ratio:.2f}, Code ratio: {pair.code_ratio:.2f}")
        return

    # Parse framework and init classifier
    if not framework_path:
        print("Error: --framework required for classification", file=sys.stderr)
        sys.exit(1)

    criteria_db = parse_framework(framework_path)
    print(f"\nFramework: {len(criteria_db)} criteria")

    if not classifier_spec:
        print("Error: --classifier required for classification", file=sys.stderr)
        sys.exit(1)

    provider = init_provider(classifier_spec)
    print(f"Classifier: {classifier_spec.provider}:{classifier_spec.model}")

    # Stage 1: Category triage
    print(f"\nStage 1: Category triage ({len(pairs)} pairs)...")
    triaged = triage_classify(
        pairs, provider, classifier_spec, batch_size=10, verbose=verbose
    )
    print(f"  Triaged: {len(triaged)} pairs with categories")

    # Filter by triage quality
    triaged = [(p, cats, q) for p, cats, q in triaged if q >= min_quality and cats]
    print(f"  After quality filter (>= {min_quality}): {len(triaged)}")

    if not triaged:
        print("No pairs passed triage. Try lowering --min-quality.")
        return

    # Stage 2: Criterion-level classification (parallel)
    print(f"\nStage 2: Criterion classification ({len(triaged)} pairs)...")
    classified: list[ClassifiedTurn] = []
    completed = 0

    def _classify_one(item):
        pair, cats, quality = item
        criteria = criterion_classify(pair, cats, criteria_db, provider, classifier_spec)
        return ClassifiedTurn(
            turn=pair,
            categories=cats,
            triage_quality=quality,
            criteria=criteria,
        )

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_classify_one, item): item for item in triaged
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result.criteria:
                        classified.append(result)
                    completed += 1
                    if verbose and completed % 20 == 0:
                        print(f"  [{completed}/{len(triaged)}] classified")
                except Exception as e:
                    completed += 1
                    print(f"  Classification error: {e}", file=sys.stderr)
    else:
        for item in triaged:
            result = _classify_one(item)
            if result.criteria:
                classified.append(result)
            completed += 1
            if verbose and completed % 20 == 0:
                print(f"  [{completed}/{len(triaged)}] classified")

    print(f"  Classified: {len(classified)} pairs with criteria matches")

    # Convert to TrainingExamples
    examples: list[TrainingExample] = []
    for ct in classified:
        for match in ct.criteria:
            quality = float(match.get("quality", 0.0))
            if quality < min_quality:
                continue
            cid = match["criterion_id"]
            info = criteria_db.get(cid, {})

            examples.append(TrainingExample(
                criterion_id=cid,
                behavioral_target=info.get("criterion", ""),
                system_prompt="",  # Harvested conversations have no system prompt
                user_input=ct.turn.user_text,
                ideal_output=ct.turn.assistant_text,
                generator_model="human:harvested",
                scenario_tag=match.get("reasoning", "")[:60],
                quality_score=quality,
            ))

    print(f"\nTraining examples: {len(examples)}")

    if not examples:
        print("No examples passed quality threshold.")
        return

    # Deduplicate by (criterion_id, user_input hash) — skip exact duplicates
    seen: set[tuple[str, str]] = set()
    deduped: list[TrainingExample] = []
    for ex in examples:
        key = (ex.criterion_id, ex.user_input[:200])
        if key not in seen:
            seen.add(key)
            deduped.append(ex)
    if len(deduped) < len(examples):
        print(f"Dedup removed {len(examples) - len(deduped)} duplicates")
    examples = deduped

    # Write output
    formatter = FORMATTERS[output_format]

    if split_by_category:
        output_dir = output_path
        manifest_data = write_split_output(
            examples, output_dir, output_format, formatter, append=append
        )
        manifest_path = write_manifest(
            manifest_data, output_dir, output_format,
            [classifier_spec] if classifier_spec else [],
        )
        print(f"\nWrote {len(examples)} examples across {len(manifest_data)} criteria")
        print(f"Library: {output_dir}/")
        print(f"Manifest: {manifest_path}")

        # Category summary
        cats: dict[str, int] = {}
        for info in manifest_data.values():
            cats[info["category"]] = cats.get(info["category"], 0) + info["example_count"]
        for cat in sorted(cats):
            print(f"  {cat}: {cats[cat]} examples")
    else:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"
        with open(out_path, mode) as f:
            for ex in examples:
                line = json.dumps(formatter(ex), ensure_ascii=False)
                f.write(line + "\n")

        print(f"\nWrote {len(examples)} examples to {output_path} ({output_format})")

    # Per-criterion summary
    if verbose:
        by_cid: dict[str, list[TrainingExample]] = {}
        for ex in examples:
            by_cid.setdefault(ex.criterion_id, []).append(ex)

        print("\n--- Per-criterion summary ---")
        for cid in sorted(by_cid):
            cid_examples = by_cid[cid]
            avg_q = sum(ex.quality_score for ex in cid_examples) / len(cid_examples)
            print(f"  {cid}: {len(cid_examples)} examples, avg quality: {avg_q:.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI-SETT Conversation Harvester — extract training examples from real conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Harvest from Claude Code transcripts
  %(prog)s \\
    --input docs/claude-transcripts/all-conversations.jsonl \\
    --framework docs/AI-SETT-FRAMEWORK.md \\
    --classifier anthropic:claude-3-5-haiku-20241022 \\
    --output training_data/ --split-by-category --append

  # Dry run — show extraction stats
  %(prog)s \\
    --input docs/claude-transcripts/all-conversations.jsonl --dry-run

  # Higher quality threshold
  %(prog)s \\
    --input all-conversations.jsonl \\
    --framework docs/AI-SETT-FRAMEWORK.md \\
    --classifier anthropic:claude-3-5-haiku-20241022 \\
    --min-quality 0.8
        """,
    )

    parser.add_argument("--input", required=True,
                        help="Conversation file (JSONL or JSON)")
    parser.add_argument("--framework", default="docs/AI-SETT-FRAMEWORK.md",
                        help="Path to AI-SETT-FRAMEWORK.md (default: docs/AI-SETT-FRAMEWORK.md)")
    parser.add_argument("--classifier",
                        help="Classifier spec as 'provider:model' (required unless --dry-run)")

    # Output
    parser.add_argument("--output", default="training_data",
                        help="Output directory or file (default: training_data)")
    parser.add_argument("--format", default="raw_jsonl", choices=list(FORMATTERS),
                        help="Output format (default: raw_jsonl)")
    parser.add_argument("--split-by-category", action="store_true",
                        help="Write category/subcategory/criterion tree")
    parser.add_argument("--append", action="store_true",
                        help="Add to existing library")

    # Quality & filtering
    parser.add_argument("--min-quality", type=float, default=0.7,
                        help="Minimum classifier quality score (default: 0.7)")
    parser.add_argument("--min-user-words", type=int, default=5,
                        help="Skip short user messages (default: 5)")
    parser.add_argument("--min-assistant-words", type=int, default=20,
                        help="Skip short assistant responses (default: 20)")
    parser.add_argument("--max-tool-ratio", type=float, default=0.6,
                        help="Skip tool-heavy exchanges (default: 0.6)")

    # Processing
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Parallel classifier calls (default: 4)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max turn pairs to process, 0=all (default: 0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stats only, no API calls")
    parser.add_argument("--verbose", action="store_true",
                        help="Detailed output")

    args = parser.parse_args()

    # Validate
    if not args.dry_run and not args.classifier:
        parser.error("--classifier is required (unless using --dry-run)")

    # Parse classifier spec
    classifier_spec = None
    if args.classifier:
        classifier_spec = parse_generator_spec(args.classifier)

    harvest(
        input_path=args.input,
        framework_path=args.framework,
        classifier_spec=classifier_spec,
        output_path=args.output,
        output_format=args.format,
        split_by_category=args.split_by_category,
        append=args.append,
        min_quality=args.min_quality,
        min_user_words=args.min_user_words,
        min_assistant_words=args.min_assistant_words,
        max_tool_ratio=args.max_tool_ratio,
        concurrency=args.concurrency,
        limit=args.limit,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
