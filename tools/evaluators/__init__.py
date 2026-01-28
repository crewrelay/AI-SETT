"""Evaluator registry — deterministic and LLM-based evaluation."""

from __future__ import annotations

from typing import Callable

# type: evaluation function (response_text, criterion_spec) -> (bool, evidence_str)
EvalFn = Callable[[str, dict], tuple[bool, str]]

_REGISTRY: dict[str, EvalFn] = {}


def register(name: str):
    """Decorator to register an evaluation function."""
    def decorator(fn: EvalFn):
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_evaluator(name: str) -> EvalFn:
    """Get a registered evaluator by name."""
    _ensure_loaded()
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown evaluator '{name}'. Available: {available}")
    return _REGISTRY[name]


def list_evaluators() -> list[str]:
    """Return names of all registered evaluators."""
    _ensure_loaded()
    return sorted(_REGISTRY)


def evaluate(response: str, criterion: dict) -> tuple[bool, str]:
    """Evaluate a response against a criterion spec.

    The criterion dict should have a 'check' field that maps to a rule type.
    Falls back to 'contains' check if no specific type given.
    """
    _ensure_loaded()
    from .rule_evaluator import evaluate_criterion
    return evaluate_criterion(response, criterion)


_loaded = False


def _ensure_loaded():
    global _loaded
    if _loaded:
        return
    _loaded = True
    from . import rule_evaluator  # noqa: F401
