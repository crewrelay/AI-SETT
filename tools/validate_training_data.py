#!/usr/bin/env python3
"""AI-SETT Training Data Validator

Validates training data contributions against schema, content, and
per-category rules before merge.

Usage:
    # Validate a single file
    python -m tools.validate_training_data training_data/metacognition/self_awareness/M.SA.01.jsonl

    # Validate entire library
    python -m tools.validate_training_data training_data/

    # Validate with strict mode (LLM quality check)
    python -m tools.validate_training_data training_data/ \
        --strict --provider anthropic --model claude-3-5-haiku-20241022

    # PR check (exit code 1 if failures)
    python -m tools.validate_training_data training_data/ --ci
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from tools.training_data_generator import (
    CATEGORY_MAP,
    jaccard_similarity,
    parse_framework,
)


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "criterion_id": str,
    "behavioral_target": str,
    "system_prompt": str,
    "user_input": str,
    "ideal_output": str,
    "generator_model": str,
    "scenario_tag": str,
    "quality_score": (int, float),
}

BOILERPLATE_PHRASES = [
    "i'd be happy to help",
    "i'd be glad to help",
    "great question!",
    "great question.",
    "that's a great question",
    "that's an excellent question",
    "absolutely! ",
    "of course! ",
    "certainly! ",
    "sure thing! ",
    "no problem! ",
    "thank you for asking",
    "thanks for asking",
    "what a great question",
    "what an excellent question",
    "i appreciate you asking",
    "that's a really good question",
]

GENERATOR_MODEL_PATTERN = re.compile(
    r'^(?:[a-z]+:[a-zA-Z0-9._-]+(?::[^\s]+)?|human:(?:manual|harvested))$'
)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_schema(example: dict, idx: int) -> list[str]:
    """Check all required fields are present and correctly typed."""
    issues = []
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in example:
            issues.append(f"example {idx}: missing required field '{field}'")
        elif not isinstance(example[field], expected_type):
            actual = type(example[field]).__name__
            expected = expected_type.__name__ if isinstance(expected_type, type) else str(expected_type)
            issues.append(f"example {idx}: '{field}' should be {expected}, got {actual}")
    return issues


def validate_criterion_id(criterion_id: str, file_path: str, framework: dict, idx: int) -> list[str]:
    """Check criterion ID exists in framework and matches file path."""
    issues = []

    # Check ID exists in framework
    if framework and criterion_id not in framework:
        issues.append(f"example {idx}: criterion_id '{criterion_id}' not found in framework")

    # Check ID matches file path
    path = Path(file_path)
    if path.suffix == ".jsonl":
        expected_stem = path.stem  # e.g. "M.SA.01"
        if expected_stem != "_combined" and criterion_id != expected_stem:
            issues.append(
                f"example {idx}: criterion_id '{criterion_id}' doesn't match "
                f"file name '{expected_stem}'"
            )

    # Check ID format
    if not re.match(r'^[A-Z]\.[A-Z]{2}\.\d{2}$', criterion_id):
        issues.append(f"example {idx}: criterion_id '{criterion_id}' doesn't match format X.YY.##")

    return issues


# ---------------------------------------------------------------------------
# Content checks
# ---------------------------------------------------------------------------

def check_lengths(example: dict, idx: int) -> list[str]:
    """Check user_input and ideal_output word counts."""
    issues = []
    user_words = len(example.get("user_input", "").split())
    output_words = len(example.get("ideal_output", "").split())

    if user_words < 20:
        issues.append(f"example {idx}: user_input too short ({user_words} words, min 20)")
    if user_words > 500:
        issues.append(f"example {idx}: user_input too long ({user_words} words, max 500)")
    if output_words < 50:
        issues.append(f"example {idx}: ideal_output too short ({output_words} words, min 50)")
    if output_words > 1500:
        issues.append(f"example {idx}: ideal_output too long ({output_words} words, max 1500)")

    return issues


def check_boilerplate(example: dict, idx: int) -> list[str]:
    """Detect boilerplate openers in ideal_output."""
    issues = []
    output_lower = example.get("ideal_output", "").lower().strip()

    for phrase in BOILERPLATE_PHRASES:
        if output_lower.startswith(phrase):
            issues.append(f"example {idx}: ideal_output starts with boilerplate: '{phrase.strip()}'")
            break

    return issues


def check_duplicates(examples: list[dict], threshold: float = 0.6) -> list[str]:
    """Detect near-duplicate examples within the same file."""
    issues = []
    texts = []
    for i, ex in enumerate(examples):
        text = ex.get("user_input", "") + " " + ex.get("ideal_output", "")
        texts.append((i, text))

    max_sim = 0.0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = jaccard_similarity(texts[i][1], texts[j][1])
            max_sim = max(max_sim, sim)
            if sim > threshold:
                issues.append(
                    f"examples {texts[i][0]} and {texts[j][0]}: "
                    f"near-duplicate (Jaccard similarity: {sim:.2f})"
                )

    return issues, max_sim


def check_generator_model(example: dict, idx: int) -> list[str]:
    """Validate generator_model format."""
    issues = []
    gm = example.get("generator_model", "")
    if not GENERATOR_MODEL_PATTERN.match(gm):
        issues.append(
            f"example {idx}: generator_model '{gm}' doesn't match "
            f"'provider:model' or 'human:manual'/'human:harvested'"
        )
    return issues


def check_quality_score(example: dict, idx: int) -> list[str]:
    """Validate quality_score range."""
    issues = []
    score = example.get("quality_score")
    if score is not None and isinstance(score, (int, float)):
        if score < 0.0 or score > 1.0:
            issues.append(f"example {idx}: quality_score {score} out of range [0.0, 1.0]")
    return issues


def check_scenario_tag(example: dict, idx: int) -> list[str]:
    """Validate scenario_tag format (2-4 words)."""
    issues = []
    tag = example.get("scenario_tag", "").strip()
    if tag:
        word_count = len(tag.split())
        if word_count < 2 or word_count > 4:
            issues.append(f"example {idx}: scenario_tag '{tag}' should be 2-4 words (got {word_count})")
    else:
        issues.append(f"example {idx}: scenario_tag is empty")
    return issues


# ---------------------------------------------------------------------------
# Category-specific rule checks
# ---------------------------------------------------------------------------

def check_understanding(example: dict, idx: int) -> list[str]:
    """U.*: user_input must contain a question or request."""
    issues = []
    cid = example.get("criterion_id", "")
    user_input = example.get("user_input", "").strip()
    output = example.get("ideal_output", "").strip()

    # Check for question or request markers
    question_markers = ["?", "how", "what", "why", "when", "where", "who", "which",
                        "can you", "could you", "please", "help me", "explain",
                        "describe", "tell me", "show me", "i need", "i want"]
    has_question = any(m in user_input.lower() for m in question_markers)
    if not has_question:
        issues.append(f"example {idx}: U.* user_input lacks clear question or request")

    # Ambiguous requests: input must be genuinely ambiguous
    if cid.startswith("U.AR"):
        ambiguity_markers = ["ambig", "could mean", "unclear", "multiple interpretations",
                             "i see this could", "depending on"]
        has_ambiguity_ack = any(m in output.lower() for m in ambiguity_markers)
        if not has_ambiguity_ack:
            issues.append(f"example {idx}: U.AR ideal_output should acknowledge ambiguity")

    # Implicit requests: input must NOT state need explicitly
    if cid.startswith("U.IR"):
        explicit_markers = ["please help me with", "i need you to", "can you do",
                            "i want you to", "please provide"]
        has_explicit = any(m in user_input.lower() for m in explicit_markers)
        if has_explicit:
            issues.append(f"example {idx}: U.IR user_input should not state need explicitly")

    return issues


def check_calibration(example: dict, idx: int) -> list[str]:
    """C.*: length and tone checks."""
    issues = []
    cid = example.get("criterion_id", "")
    output = example.get("ideal_output", "")
    word_count = len(output.split())

    # Response length criteria
    if cid.startswith("C.RL"):
        # C.RL.01 = concise, C.RL.02 = brief, C.RL.03-05 = various
        if "concise" in example.get("behavioral_target", "").lower():
            if word_count > 100:
                issues.append(
                    f"example {idx}: C.RL concise criterion but output is {word_count} words (max ~100)"
                )
        if "detailed" in example.get("behavioral_target", "").lower():
            if word_count < 300:
                issues.append(
                    f"example {idx}: C.RL detailed criterion but output is only {word_count} words (min ~300)"
                )

    # Stopping criteria: should not over-elaborate
    if cid.startswith("C.ST"):
        # Check for trailing filler
        trailing_markers = ["is there anything else", "let me know if",
                            "feel free to ask", "hope this helps",
                            "don't hesitate to"]
        if any(m in output.lower()[-200:] for m in trailing_markers):
            issues.append(f"example {idx}: C.ST ideal_output has trailing filler (should stop when done)")

    return issues


def check_generation(example: dict, idx: int) -> list[str]:
    """G.*: must include the generated artifact."""
    issues = []
    cid = example.get("criterion_id", "")
    output = example.get("ideal_output", "")

    # Code generation: must contain code
    if cid.startswith("G.CD"):
        code_markers = ["```", "def ", "function ", "class ", "import ", "const ",
                        "var ", "let ", "public ", "private ", "void "]
        has_code = any(m in output for m in code_markers)
        if not has_code:
            issues.append(f"example {idx}: G.CD ideal_output should contain actual code")

    # Structured output: must match requested format
    if cid.startswith("G.SO"):
        format_markers = ["```json", "```csv", "```xml", "```yaml", "|", "{", "["]
        has_structure = any(m in output for m in format_markers)
        if not has_structure:
            issues.append(f"example {idx}: G.SO ideal_output should contain structured output")

    return issues


def check_knowledge(example: dict, idx: int) -> list[str]:
    """K.*: must reference specific facts, not vague generalities."""
    issues = []
    cid = example.get("criterion_id", "")
    output = example.get("ideal_output", "")

    # Medical/legal: should include disclaimers
    if cid.startswith("K.MH"):
        disclaimer_markers = ["consult", "professional", "doctor", "physician",
                              "medical advice", "healthcare provider",
                              "not a substitute", "seek medical"]
        has_disclaimer = any(m in output.lower() for m in disclaimer_markers)
        if not has_disclaimer:
            issues.append(f"example {idx}: K.MH ideal_output should include medical disclaimer")

    if cid.startswith("K.LG"):
        disclaimer_markers = ["consult", "attorney", "lawyer", "legal advice",
                              "legal professional", "not legal advice",
                              "legal counsel"]
        has_disclaimer = any(m in output.lower() for m in disclaimer_markers)
        if not has_disclaimer:
            issues.append(f"example {idx}: K.LG ideal_output should include legal disclaimer")

    return issues


def check_reasoning(example: dict, idx: int) -> list[str]:
    """R.*: must show explicit reasoning steps."""
    issues = []
    cid = example.get("criterion_id", "")
    output = example.get("ideal_output", "")

    reasoning_markers = ["because", "therefore", "since", "first", "second",
                         "step", "reason", "follows", "thus", "consequently",
                         "given that", "this means", "we can", "let's",
                         "if we", "which gives", "so we"]
    has_reasoning = sum(1 for m in reasoning_markers if m in output.lower())

    if has_reasoning < 2:
        issues.append(
            f"example {idx}: R.* ideal_output lacks explicit reasoning steps "
            f"(found {has_reasoning} reasoning markers, need at least 2)"
        )

    # Mathematical: must show work
    if cid.startswith("R.MT"):
        math_markers = ["=", "+", "-", "*", "/", "×", "÷", "²", "√",
                        "equation", "formula", "calculate"]
        has_math = any(m in output for m in math_markers)
        if not has_math:
            issues.append(f"example {idx}: R.MT ideal_output should show mathematical work")

    # Critical thinking: must evaluate claims
    if cid.startswith("R.CT"):
        eval_markers = ["however", "on the other hand", "but", "although",
                        "while", "consider", "limitation", "counterargument",
                        "alternatively", "critique", "flaw", "assumption"]
        has_eval = sum(1 for m in eval_markers if m in output.lower())
        if has_eval < 1:
            issues.append(f"example {idx}: R.CT ideal_output should evaluate claims critically")

    return issues


def check_boundaries(example: dict, idx: int) -> list[str]:
    """B.*: refusal and safety checks."""
    issues = []
    cid = example.get("criterion_id", "")
    output = example.get("ideal_output", "")

    # Refusals: must refuse clearly with explanation
    if cid.startswith("B.RF"):
        refusal_markers = ["i can't", "i cannot", "i'm not able", "i won't",
                           "i'm unable", "not appropriate", "i shouldn't",
                           "not something i", "beyond what i",
                           "i need to decline", "i must decline"]
        has_refusal = any(m in output.lower() for m in refusal_markers)
        if not has_refusal:
            issues.append(f"example {idx}: B.RF ideal_output should contain clear refusal")

    # Over-refusal: must help despite surface-level danger
    if cid.startswith("B.OR"):
        refusal_markers = ["i can't", "i cannot", "i'm not able", "i won't",
                           "i'm unable", "i must decline"]
        has_refusal = any(m in output.lower() for m in refusal_markers)
        if has_refusal:
            issues.append(
                f"example {idx}: B.OR ideal_output should NOT refuse "
                f"(tests avoidance of over-refusal)"
            )

    # Safety: must redirect without providing harmful content
    if cid.startswith("B.SF"):
        redirect_markers = ["instead", "i'd suggest", "here's what i can",
                            "what i can help with", "alternative",
                            "let me help you with", "i can help you"]
        has_redirect = any(m in output.lower() for m in redirect_markers)
        if not has_redirect:
            issues.append(
                f"example {idx}: B.SF ideal_output should redirect to safe alternative"
            )

    return issues


def check_interaction(example: dict, idx: int) -> list[str]:
    """I.*: multi-turn, error handling, clarification checks."""
    issues = []
    cid = example.get("criterion_id", "")
    user_input = example.get("user_input", "")
    output = example.get("ideal_output", "")

    # Error handling: input must contain an error
    if cid.startswith("I.EH"):
        error_markers = ["error", "wrong", "mistake", "incorrect", "bug",
                         "doesn't work", "not working", "failed", "crash",
                         "broken", "issue", "problem"]
        has_error = any(m in user_input.lower() for m in error_markers)
        if not has_error:
            issues.append(f"example {idx}: I.EH user_input should contain an error or contradiction")

    # Clarification: output must ask a clarifying question
    if cid.startswith("I.CR"):
        question_markers = ["?", "could you clarify", "do you mean",
                            "can you specify", "which", "what exactly",
                            "are you referring to", "could you tell me more"]
        has_question = any(m in output.lower() for m in question_markers)
        if not has_question:
            issues.append(
                f"example {idx}: I.CR ideal_output should ask a clarifying question"
            )

    return issues


def check_tool_use(example: dict, idx: int) -> list[str]:
    """T.WS-T.TS: must describe or demonstrate tool usage."""
    issues = []
    output = example.get("ideal_output", "")

    tool_markers = ["search", "execute", "run", "call", "api", "tool",
                    "function", "query", "fetch", "request", "command",
                    "use the", "using the", "invoke"]
    has_tool = any(m in output.lower() for m in tool_markers)
    if not has_tool:
        issues.append(
            f"example {idx}: T.* ideal_output should describe or demonstrate tool usage"
        )

    return issues


def check_emotional_intelligence(example: dict, idx: int) -> list[str]:
    """E.*: emotional content and acknowledgment checks."""
    issues = []
    cid = example.get("criterion_id", "")
    user_input = example.get("user_input", "")
    output = example.get("ideal_output", "")

    # Input must contain emotional content
    emotion_markers = ["frustrated", "angry", "sad", "worried", "anxious",
                       "scared", "excited", "happy", "devastated", "overwhelmed",
                       "stressed", "disappointed", "grief", "loss", "hurt",
                       "confused", "upset", "afraid", "struggling", "suffering",
                       "crying", "depressed", "furious", "terrified", "hopeless"]
    has_emotion = any(m in user_input.lower() for m in emotion_markers)
    if not has_emotion:
        issues.append(f"example {idx}: E.* user_input should contain emotional content")

    # Output must acknowledge emotion before content
    ack_markers = ["i understand", "that sounds", "i can see", "it makes sense",
                   "that must be", "i hear you", "i'm sorry", "that's",
                   "it sounds like", "i can imagine", "understandably",
                   "it's natural to feel", "your feelings"]
    has_ack = any(m in output.lower()[:500] for m in ack_markers)
    if not has_ack:
        issues.append(
            f"example {idx}: E.* ideal_output should acknowledge emotion early in the response"
        )

    return issues


def check_metacognition(example: dict, idx: int) -> list[str]:
    """M.*: self-assessment, strategy, adaptation checks."""
    issues = []
    cid = example.get("criterion_id", "")
    output = example.get("ideal_output", "").lower()

    if cid.startswith("M.SA"):
        markers = ["i can", "i cannot", "i'm confident", "i'm uncertain",
                    "i'm not sure", "my limitation", "i don't know",
                    "i'm able to", "beyond my", "within my", "outside my",
                    "i lack", "my capabilities", "my knowledge",
                    "i should note", "i'm not equipped"]
        if not any(m in output for m in markers):
            issues.append(f"example {idx}: M.SA ideal_output lacks self-assessment language")

    if cid.startswith("M.SS"):
        markers = ["approach", "strategy", "i'll start by", "my plan",
                    "first i'll", "the way i'm thinking", "let me break",
                    "i'll tackle", "my approach", "here's how i'll",
                    "the method i'll use", "i'm going to approach"]
        if not any(m in output for m in markers):
            issues.append(f"example {idx}: M.SS ideal_output lacks strategy explanation")

    if cid.startswith("M.LA"):
        markers = ["based on", "from what you", "you mentioned earlier",
                    "adjusting", "adapting", "i've noticed", "learning from",
                    "given your previous", "updating my", "i see that",
                    "taking into account", "incorporating"]
        if not any(m in output for m in markers):
            issues.append(f"example {idx}: M.LA ideal_output lacks adaptation/learning reference")

    return issues


def check_learning(example: dict, idx: int) -> list[str]:
    """L.*: input must provide something to learn from."""
    issues = []
    user_input = example.get("user_input", "")
    output = example.get("ideal_output", "")

    learning_input_markers = ["for example", "like this", "correct me",
                              "actually", "instead of", "the right way",
                              "here's how", "notice that", "pattern",
                              "when you see", "the rule is"]
    has_learning = any(m in user_input.lower() for m in learning_input_markers)

    # Also check for corrections or examples in input
    if not has_learning:
        # Check for code examples, corrections, or patterns
        if "```" not in user_input and "example" not in user_input.lower():
            issues.append(
                f"example {idx}: L.* user_input should provide something to learn from "
                f"(example, correction, pattern)"
            )

    return issues


def check_teaching(example: dict, idx: int) -> list[str]:
    """T.DA-T.AD: scaffolding, misconceptions, diagnostics."""
    issues = []
    cid = example.get("criterion_id", "")
    output = example.get("ideal_output", "")

    # Scaffolding: must NOT give full answer
    if cid.startswith("T.SC"):
        scaffolding_markers = ["what do you think", "try", "consider",
                               "what would happen if", "hint", "let's think",
                               "can you", "what if", "how might you",
                               "have you considered", "what's your intuition"]
        has_scaffolding = any(m in output.lower() for m in scaffolding_markers)
        if not has_scaffolding:
            issues.append(
                f"example {idx}: T.SC ideal_output should guide learner, not give full answer"
            )

    # Misconception handling: must engage with WHY
    if cid.startswith("T.MH"):
        engage_markers = ["common misconception", "it's understandable why",
                          "many people think", "the reason this seems",
                          "this is a natural assumption", "you might think",
                          "while it seems like", "the confusion",
                          "it's easy to think", "this is tricky because"]
        has_engage = any(m in output.lower() for m in engage_markers)
        if not has_engage:
            issues.append(
                f"example {idx}: T.MH ideal_output should engage with WHY the misconception seems right"
            )

    # Diagnostic: must ask questions to assess understanding
    if cid.startswith("T.DA"):
        question_markers = ["?", "tell me", "what do you", "how would you",
                            "can you explain", "show me", "walk me through"]
        has_question = any(m in output.lower() for m in question_markers)
        if not has_question:
            issues.append(
                f"example {idx}: T.DA ideal_output should ask diagnostic questions"
            )

    return issues


# Category rule dispatch
CATEGORY_RULES = {
    "understanding": check_understanding,
    "calibration": check_calibration,
    "generation": check_generation,
    "knowledge": check_knowledge,
    "reasoning": check_reasoning,
    "boundaries": check_boundaries,
    "interaction": check_interaction,
    "tool_use": check_tool_use,
    "emotional_intelligence": check_emotional_intelligence,
    "metacognition": check_metacognition,
    "learning": check_learning,
    "teaching": check_teaching,
}


def get_category_from_id(criterion_id: str) -> Optional[str]:
    """Get category slug from criterion ID."""
    prefix = criterion_id[:4] if len(criterion_id) >= 4 else ""
    result = CATEGORY_MAP.get(prefix)
    if result:
        return result[0]
    return None


# ---------------------------------------------------------------------------
# Strict mode (LLM quality check)
# ---------------------------------------------------------------------------

def llm_quality_check(
    example: dict,
    idx: int,
    provider_name: str,
    model: str,
    framework: dict,
) -> list[str]:
    """Use an LLM judge to verify the example demonstrates its criterion."""
    from tools.providers import get_provider
    from tools.providers.base import CompletionRequest, Message

    issues = []
    cid = example.get("criterion_id", "")
    criterion_info = framework.get(cid, {})

    if not criterion_info:
        return issues  # Can't check without framework info

    api_key = os.environ.get(
        {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(
            provider_name, f"{provider_name.upper()}_API_KEY"
        ),
        "",
    )
    if not api_key:
        return [f"example {idx}: strict mode requires API key for {provider_name}"]

    provider = get_provider(provider_name, api_key=api_key)

    prompt = (
        f"You are evaluating a training example for AI fine-tuning.\n\n"
        f"Criterion: {cid} — {criterion_info.get('criterion', '')}\n"
        f"How to verify: {criterion_info.get('verify', '')}\n\n"
        f"User input:\n{example.get('user_input', '')}\n\n"
        f"Ideal output:\n{example.get('ideal_output', '')}\n\n"
        f"Does the ideal_output clearly demonstrate the criterion described above?\n"
        f"Answer with exactly 'PASS' or 'FAIL' on the first line, then a brief explanation."
    )

    try:
        request = CompletionRequest(
            messages=[Message(role="user", content=prompt)],
            model=model,
            temperature=0.0,
            max_tokens=200,
        )
        response = provider.complete(request)
        first_line = response.content.strip().split("\n")[0].strip().upper()
        if first_line == "FAIL":
            explanation = response.content.strip().split("\n", 1)
            reason = explanation[1].strip() if len(explanation) > 1 else "LLM judge rejected"
            issues.append(f"example {idx}: LLM quality check FAIL — {reason}")
    except Exception as e:
        issues.append(f"example {idx}: LLM quality check error — {e}")

    return issues


# ---------------------------------------------------------------------------
# File validation orchestration
# ---------------------------------------------------------------------------

def validate_file(
    file_path: str,
    framework: dict,
    strict: bool = False,
    strict_provider: Optional[str] = None,
    strict_model: Optional[str] = None,
    dedup_threshold: float = 0.6,
) -> tuple[bool, list[str], dict]:
    """Validate a single JSONL file. Returns (passed, issues, stats)."""
    path = Path(file_path)
    issues = []
    examples = []

    if not path.exists():
        return False, [f"file not found: {file_path}"], {}

    if path.name == "_combined.jsonl" or path.name == "manifest.json":
        return True, [], {"skipped": True}

    # Parse JSONL
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                examples.append(example)
            except json.JSONDecodeError as e:
                issues.append(f"line {line_num}: invalid JSON — {e}")

    if not examples:
        return False, ["file is empty or contains no valid JSON lines"], {}

    # Schema validation
    for i, ex in enumerate(examples):
        issues.extend(validate_schema(ex, i))

    # Criterion ID validation
    for i, ex in enumerate(examples):
        if "criterion_id" in ex:
            issues.extend(validate_criterion_id(ex["criterion_id"], file_path, framework, i))

    # Length checks
    for i, ex in enumerate(examples):
        issues.extend(check_lengths(ex, i))

    # Boilerplate detection
    for i, ex in enumerate(examples):
        issues.extend(check_boilerplate(ex, i))

    # Generator model format
    for i, ex in enumerate(examples):
        issues.extend(check_generator_model(ex, i))

    # Quality score range
    for i, ex in enumerate(examples):
        issues.extend(check_quality_score(ex, i))

    # Scenario tag format
    for i, ex in enumerate(examples):
        issues.extend(check_scenario_tag(ex, i))

    # Duplicate detection
    dup_issues, max_sim = check_duplicates(examples, dedup_threshold)
    issues.extend(dup_issues)

    # Category-specific rules
    category_issues = 0
    for i, ex in enumerate(examples):
        cid = ex.get("criterion_id", "")
        category = get_category_from_id(cid)
        if category and category in CATEGORY_RULES:
            cat_issues = CATEGORY_RULES[category](ex, i)
            issues.extend(cat_issues)
            category_issues += len(cat_issues)

    # Strict mode: LLM quality check
    if strict and strict_provider and strict_model:
        for i, ex in enumerate(examples):
            issues.extend(llm_quality_check(ex, i, strict_provider, strict_model, framework))

    stats = {
        "example_count": len(examples),
        "max_similarity": round(max_sim, 2) if examples else 0.0,
        "category_issues": category_issues,
    }

    passed = len(issues) == 0
    return passed, issues, stats


# ---------------------------------------------------------------------------
# Directory validation
# ---------------------------------------------------------------------------

def validate_directory(
    dir_path: str,
    framework: dict,
    strict: bool = False,
    strict_provider: Optional[str] = None,
    strict_model: Optional[str] = None,
    dedup_threshold: float = 0.6,
    verbose: bool = False,
) -> tuple[int, int, int]:
    """Validate all JSONL files in a directory. Returns (passed, failed, total_issues)."""
    path = Path(dir_path)
    files = sorted(path.rglob("*.jsonl"))

    passed_count = 0
    failed_count = 0
    total_issues = 0

    for f in files:
        # Skip combined files
        if f.name.startswith("_"):
            continue

        print(f"\nValidating {f.relative_to(path)}")
        passed, issues, stats = validate_file(
            str(f), framework, strict, strict_provider, strict_model, dedup_threshold
        )

        if stats.get("skipped"):
            continue

        example_count = stats.get("example_count", 0)
        max_sim = stats.get("max_similarity", 0.0)
        print(f"  {example_count} examples found")

        if passed:
            print(f"  \u2713 Schema: all fields present")
            print(f"  \u2713 Criterion IDs: all valid")
            print(f"  \u2713 Lengths: all within range")
            print(f"  \u2713 Boilerplate: none detected")
            print(f"  \u2713 Duplicates: none (max similarity: {max_sim})")
            print(f"  \u2713 Category rules: all checks passed")
            print(f"  PASS")
            passed_count += 1
        else:
            # Print check summaries
            schema_issues = [i for i in issues if "missing required" in i or "should be" in i]
            criterion_issues = [i for i in issues if "criterion_id" in i and "not found" in i]
            length_issues = [i for i in issues if "too short" in i or "too long" in i]
            boilerplate_issues = [i for i in issues if "boilerplate" in i]
            dup_issues_list = [i for i in issues if "near-duplicate" in i]

            if not schema_issues:
                print(f"  \u2713 Schema: all fields present")
            else:
                print(f"  \u2717 Schema: {len(schema_issues)} issues")

            if not criterion_issues:
                print(f"  \u2713 Criterion IDs: all valid")
            else:
                print(f"  \u2717 Criterion IDs: {len(criterion_issues)} issues")

            if not length_issues:
                print(f"  \u2713 Lengths: all within range")
            else:
                print(f"  \u2717 Lengths: {len(length_issues)} issues")

            if not boilerplate_issues:
                print(f"  \u2713 Boilerplate: none detected")
            else:
                print(f"  \u2717 Boilerplate: {len(boilerplate_issues)} issues")

            if not dup_issues_list:
                print(f"  \u2713 Duplicates: none (max similarity: {max_sim})")
            else:
                print(f"  \u2717 Duplicates: {len(dup_issues_list)} near-duplicates")

            cat_issue_count = stats.get("category_issues", 0)
            if cat_issue_count == 0:
                print(f"  \u2713 Category rules: all checks passed")
            else:
                print(f"  \u2717 Category rules: {cat_issue_count} issues")

            print(f"  FAIL ({len(issues)} issues)")

            if verbose:
                for issue in issues:
                    print(f"    - {issue}")

            failed_count += 1
            total_issues += len(issues)

    return passed_count, failed_count, total_issues


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI-SETT Training Data Validator — checks contributions before merge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a single file
  %(prog)s training_data/metacognition/self_awareness/M.SA.01.jsonl

  # Validate entire library
  %(prog)s training_data/

  # Validate with strict mode (LLM quality check)
  %(prog)s training_data/ --strict --provider anthropic --model claude-3-5-haiku-20241022

  # PR check (exit code 1 if failures)
  %(prog)s training_data/ --ci
        """,
    )

    parser.add_argument("path", help="Path to JSONL file or directory to validate")
    parser.add_argument("--framework", default="docs/AI-SETT-FRAMEWORK.md",
                        help="Path to AI-SETT-FRAMEWORK.md (default: docs/AI-SETT-FRAMEWORK.md)")
    parser.add_argument("--strict", action="store_true",
                        help="Enable strict mode (LLM quality checks)")
    parser.add_argument("--provider", help="LLM provider for strict mode (e.g., anthropic)")
    parser.add_argument("--model", help="LLM model for strict mode")
    parser.add_argument("--dedup-threshold", type=float, default=0.6,
                        help="Jaccard similarity threshold for duplicate detection (default: 0.6)")
    parser.add_argument("--ci", action="store_true",
                        help="CI mode: exit code 1 if any failures")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed issue descriptions")

    args = parser.parse_args()

    # Validate strict mode args
    if args.strict and (not args.provider or not args.model):
        parser.error("--strict requires --provider and --model")

    # Parse framework
    framework = {}
    framework_path = Path(args.framework)
    if framework_path.exists():
        framework = parse_framework(args.framework)
        print(f"Loaded {len(framework)} criteria from framework")
    else:
        print(f"Warning: framework file not found at {args.framework}, skipping criterion ID validation")

    # Validate
    target = Path(args.path)

    if target.is_file():
        print(f"\nValidating {target}")
        passed, issues, stats = validate_file(
            str(target), framework, args.strict, args.provider, args.model,
            args.dedup_threshold,
        )
        example_count = stats.get("example_count", 0)
        max_sim = stats.get("max_similarity", 0.0)
        print(f"  {example_count} examples found")

        if passed:
            print(f"  \u2713 All checks passed")
            print(f"  PASS")
        else:
            for issue in issues:
                print(f"  \u2717 {issue}")
            print(f"  FAIL ({len(issues)} issues)")

        if args.ci and not passed:
            sys.exit(1)

    elif target.is_dir():
        passed, failed, total_issues = validate_directory(
            str(target), framework, args.strict, args.provider, args.model,
            args.dedup_threshold, args.verbose,
        )

        print(f"\n{'=' * 40}")
        print(f"Results: {passed} passed, {failed} failed, {total_issues} total issues")

        if args.ci and failed > 0:
            sys.exit(1)
    else:
        print(f"Error: {args.path} is not a file or directory", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
