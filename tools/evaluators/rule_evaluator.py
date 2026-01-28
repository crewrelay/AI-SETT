"""Deterministic rule-based evaluation for AI-SETT criteria.

Supports: word_count, contains, not_contains, regex, regex_absent,
          response_length, contains_question, and custom compound checks.
"""

from __future__ import annotations

import re


def evaluate_criterion(response: str, criterion: dict) -> tuple[bool, str]:
    """Evaluate a single criterion against a response.

    Args:
        response: The model's response text.
        criterion: Dict with 'check' and 'pass' fields from probe YAML.

    Returns:
        (passed: bool, evidence: str)
    """
    check = criterion.get("check", "")
    pass_cond = criterion.get("pass", "")

    # Try to infer the rule type from the pass condition
    lower_pass = pass_cond.lower()

    # Word count checks
    wc_match = re.search(r"word count\s*[≤<>≥]=?\s*(\d+)", lower_pass)
    if wc_match:
        return _check_word_count(response, pass_cond, int(wc_match.group(1)))

    # "Under N words" pattern
    under_match = re.search(r"under\s+(\d+)\s+words", lower_pass)
    if under_match:
        limit = int(under_match.group(1))
        count = len(response.split())
        passed = count < limit
        return passed, f"Word count: {count} (limit: <{limit})"

    # Contains check
    if "contains" in lower_pass and "not" not in lower_pass.split("contains")[0][-5:]:
        return _check_contains(response, pass_cond)

    # Does not contain / absence check
    if "not present" in lower_pass or "absent" in lower_pass or "does not" in lower_pass:
        return _check_not_contains(response, pass_cond)

    # No + something pattern (e.g., "No elaboration beyond answer")
    if lower_pass.startswith("no "):
        return _check_absence_pattern(response, pass_cond)

    # Contains question mark
    if "?" in lower_pass and ("question" in lower_pass or "seeking" in lower_pass):
        return _check_contains_question(response)

    # Phrase not present
    if "phrase not present" in lower_pass or "phrase absent" in lower_pass:
        return _check_phrase_absent(response, pass_cond)

    # Regex check
    if lower_pass.startswith("regex:"):
        pattern = pass_cond[6:].strip()
        return _check_regex(response, pattern)

    # Fallback: try a simple keyword extraction from the pass condition
    return _check_heuristic(response, check, pass_cond)


def _check_word_count(response: str, condition: str, limit: int) -> tuple[bool, str]:
    """Check word count against a limit."""
    count = len(response.split())
    # Determine comparison operator
    if "≤" in condition or "<=" in condition:
        passed = count <= limit
        op = "<="
    elif "≥" in condition or ">=" in condition:
        passed = count >= limit
        op = ">="
    elif "<" in condition:
        passed = count < limit
        op = "<"
    elif ">" in condition:
        passed = count > limit
        op = ">"
    else:
        passed = count <= limit
        op = "<="
    return passed, f"Word count: {count} ({op} {limit})"


def _check_contains(response: str, condition: str) -> tuple[bool, str]:
    """Check that response contains a keyword extracted from condition."""
    # Try to extract quoted string
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", condition)
    if quoted:
        targets = [q[0] or q[1] for q in quoted]
        for target in targets:
            if target.lower() in response.lower():
                return True, f"Found '{target}' in response"
        return False, f"Missing: {targets}"

    # Extract key terms after "Contains"
    match = re.search(r"[Cc]ontains\s+(.+)", condition)
    if match:
        term = match.group(1).strip().rstrip(".")
        if term.lower() in response.lower():
            return True, f"Found '{term}' in response"
        return False, f"Missing '{term}'"

    return False, "Could not parse contains condition"


def _check_not_contains(response: str, condition: str) -> tuple[bool, str]:
    """Check that response does NOT contain something."""
    # Extract quoted strings
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", condition)
    if quoted:
        targets = [q[0] or q[1] for q in quoted]
        for target in targets:
            if target.lower() in response.lower():
                return False, f"Found unwanted '{target}'"
        return True, f"Confirmed absent: {targets}"

    # Check common anti-patterns
    patterns = {
        "let me know": "let me know",
        "capability": "can help|i can|capable of",
        "elaboration": r"\.\s+\w.*\.\s+\w",  # multiple sentences as proxy
    }
    for key, pattern in patterns.items():
        if key in condition.lower():
            if re.search(pattern, response, re.IGNORECASE):
                return False, f"Found pattern matching '{key}'"
            return True, f"Pattern '{key}' absent"

    return True, "Absence check passed (heuristic)"


def _check_absence_pattern(response: str, condition: str) -> tuple[bool, str]:
    """Handle 'No X' style conditions."""
    # Extract what should be absent
    what = condition[3:].strip().rstrip(".")
    lower_resp = response.lower()

    # Common patterns
    if "formal language" in what.lower():
        formal_phrases = ["i would be delighted", "certainly", "i shall", "allow me to"]
        found = [p for p in formal_phrases if p in lower_resp]
        if found:
            return False, f"Found formal language: {found}"
        return True, "No formal language detected"

    if "elaboration" in what.lower() or "content beyond" in what.lower():
        sentences = [s.strip() for s in response.split(".") if s.strip()]
        if len(sentences) > 2:
            return False, f"Found {len(sentences)} sentences (possible elaboration)"
        return True, f"Brief response ({len(sentences)} sentences)"

    if "facts about" in what.lower():
        # Very simple heuristic
        word_count = len(response.split())
        if word_count > 30:
            return False, f"Response seems elaborate ({word_count} words)"
        return True, f"Concise response ({word_count} words)"

    # Generic: short response = likely no extras
    word_count = len(response.split())
    return word_count < 50, f"Word count: {word_count} (checking for absence of: {what})"


def _check_contains_question(response: str) -> tuple[bool, str]:
    """Check that response contains a question."""
    if "?" in response:
        # Find the question
        sentences = re.split(r'[.!?]+', response)
        questions = [s.strip() for s in response.split("?") if s.strip()]
        return True, f"Contains question(s): found '?' in response"
    return False, "No question mark found in response"


def _check_phrase_absent(response: str, condition: str) -> tuple[bool, str]:
    """Check that a specific phrase is not in the response."""
    # Try to extract the phrase
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", condition)
    if quoted:
        targets = [q[0] or q[1] for q in quoted]
        for target in targets:
            if target.lower() in response.lower():
                return False, f"Found phrase '{target}'"
        return True, f"Phrase(s) absent: {targets}"
    return True, "Phrase absence check passed"


def _check_regex(response: str, pattern: str) -> tuple[bool, str]:
    """Check response against a regex pattern."""
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return True, f"Regex matched: '{match.group()}'"
    return False, f"Regex not matched: /{pattern}/"


def _check_heuristic(response: str, check: str, pass_cond: str) -> tuple[bool, str]:
    """Fallback heuristic evaluation.

    Tries to extract meaning from the check/pass descriptions and make
    a reasonable determination. Returns (False, reason) when unsure to
    flag for manual review.
    """
    lower_resp = response.lower()
    lower_check = check.lower()
    lower_pass = pass_cond.lower()

    # "Asks what to fix" or similar
    if "asks" in lower_pass and ("what" in lower_pass or "which" in lower_pass):
        if "?" in response:
            return True, "Response contains a question (heuristic match)"
        return False, "No question found (heuristic)"

    # "Response addresses X"
    if "addresses" in lower_pass or "responds to" in lower_pass:
        # Can't fully evaluate without knowing the topic — pass with note
        if len(response.strip()) > 0:
            return True, "Response is non-empty (heuristic — needs manual review)"
        return False, "Empty response"

    # If we can't determine, flag for review
    return False, f"Could not evaluate automatically: check='{check}', pass='{pass_cond}'"
