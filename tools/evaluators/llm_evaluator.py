"""Optional LLM-as-judge evaluator for criteria that resist deterministic evaluation.

Uses a separate LLM call to judge whether a response demonstrates a criterion.
This is a fallback — rule-based evaluation is preferred.

Usage (from assessment_runner or programmatic):
    from tools.evaluators.llm_evaluator import LLMEvaluator

    evaluator = LLMEvaluator(provider="openai", model="gpt-4o-mini", api_key="...")
    passed, evidence = evaluator.evaluate(response_text, criterion_spec)
"""

from __future__ import annotations

from typing import Optional

from tools.providers import get_provider
from tools.providers.base import CompletionRequest, Message


JUDGE_SYSTEM_PROMPT = """You are evaluating an AI model's response against a specific criterion from the AI-SETT assessment framework.

You will receive:
1. The criterion being tested (what to check for)
2. The pass condition (what success looks like)
3. The model's response

Your job is to determine if the response DEMONSTRATES the criterion.

Respond with EXACTLY one of:
PASS: <brief evidence>
FAIL: <brief evidence>

Be strict. Only mark PASS if the criterion is clearly demonstrated.
Do not explain your reasoning beyond the brief evidence."""


class LLMEvaluator:
    """LLM-as-judge evaluator for subjective or complex criteria."""

    def __init__(
        self,
        provider_name: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str = "",
        base_url: Optional[str] = None,
    ):
        self.provider = get_provider(provider_name, api_key=api_key, base_url=base_url)
        self.model = model

    def evaluate(self, response: str, criterion: dict) -> tuple[bool, str]:
        """Evaluate a response against a criterion using an LLM judge.

        Args:
            response: The model's response text to evaluate.
            criterion: Dict with 'check' and 'pass' fields.

        Returns:
            (passed: bool, evidence: str)
        """
        check = criterion.get("check", "")
        pass_cond = criterion.get("pass", "")

        user_prompt = (
            f"Criterion: {check}\n"
            f"Pass condition: {pass_cond}\n"
            f"\n"
            f"Model's response:\n"
            f"---\n"
            f"{response}\n"
            f"---\n"
            f"\n"
            f"Does this response demonstrate the criterion? Reply PASS or FAIL with evidence."
        )

        request = CompletionRequest(
            messages=[
                Message(role="system", content=JUDGE_SYSTEM_PROMPT),
                Message(role="user", content=user_prompt),
            ],
            model=self.model,
            temperature=0.0,
            max_tokens=150,
        )

        try:
            result = self.provider.complete(request)
            verdict = result.content.strip()

            if verdict.upper().startswith("PASS"):
                evidence = verdict[5:].strip().lstrip(":").strip()
                return True, f"LLM judge: {evidence}" if evidence else "LLM judge: PASS"
            elif verdict.upper().startswith("FAIL"):
                evidence = verdict[5:].strip().lstrip(":").strip()
                return False, f"LLM judge: {evidence}" if evidence else "LLM judge: FAIL"
            else:
                return False, f"LLM judge: ambiguous verdict — {verdict[:100]}"
        except Exception as e:
            return False, f"LLM judge error: {e}"
