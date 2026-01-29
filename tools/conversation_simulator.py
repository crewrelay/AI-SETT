#!/usr/bin/env python3
"""AI-SETT Conversation Simulator

Uses a cheap model (e.g. Haiku) as a simulated user for dynamic multi-turn
assessment probes. The simulated user follows a rubric defined in the probe
YAML to role-play realistic conversations.

Usage:
    python -m tools.conversation_simulator \
        --probe probes/interaction/debugging_sim.yaml \
        --provider anthropic --model claude-sonnet-4-20250514 \
        --user-provider anthropic --user-model claude-3-5-haiku-20241022 \
        --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None

from tools.providers import get_provider
from tools.providers.base import CompletionRequest, CompletionResponse, Message


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Result of a simulated conversation."""
    probe_id: str
    simulation: bool  # always True
    conversation: list[dict]  # [{role, content}, ...]
    turns_completed: int
    max_turns: int
    total_latency_ms: float
    user_model: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------

def load_simulated_probe(probe_path: str) -> dict:
    """Load a single simulated probe from a YAML file."""
    if yaml is None:
        print("Error: PyYAML required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    path = Path(probe_path)
    if not path.is_file():
        print(f"Error: '{probe_path}' is not a file", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        probe = yaml.safe_load(f)

    if not probe:
        print(f"Error: Empty probe file '{probe_path}'", file=sys.stderr)
        sys.exit(1)

    if probe.get("type") != "simulated":
        print(f"Error: Probe type is '{probe.get('type', 'single_turn')}', expected 'simulated'",
              file=sys.stderr)
        sys.exit(1)

    if "simulation_rubric" not in probe:
        print(f"Error: Probe missing 'simulation_rubric'", file=sys.stderr)
        sys.exit(1)

    return probe


def load_simulated_probes(probe_path: str) -> list[dict]:
    """Load all simulated probes from a file or directory."""
    if yaml is None:
        print("Error: PyYAML required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    path = Path(probe_path)
    probes = []

    files: list[Path] = []
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("**/*.yaml"))
    else:
        return probes

    for yaml_file in files:
        with open(yaml_file) as f:
            for doc in yaml.safe_load_all(f):
                if doc is None:
                    continue
                if isinstance(doc, list):
                    for d in doc:
                        if isinstance(d, dict) and d.get("type") == "simulated":
                            probes.append(d)
                elif isinstance(doc, dict) and doc.get("type") == "simulated":
                    probes.append(doc)

    return probes


# ---------------------------------------------------------------------------
# Simulated user prompt
# ---------------------------------------------------------------------------

USER_SYSTEM_PROMPT = """You are role-playing as {user_role}.

Your goal: {user_goal}

Behavior: {user_behavior}

{confusion_section}

Rules:
- Stay in character at all times
- Generate ONLY the user's next message — nothing else
- Do not explain what you are doing or break character
- React naturally to the assistant's responses
- If the assistant solves your problem, you may express gratitude or ask a follow-up
- Keep messages concise and natural (1-3 sentences typical)"""


def build_user_system_prompt(rubric: dict) -> str:
    """Build the system prompt for the simulated user."""
    confusion_points = rubric.get("confusion_points", [])
    if confusion_points:
        confusion_section = "Confusion points (things you don't understand or get wrong):\n"
        for point in confusion_points:
            confusion_section += f"- {point}\n"
    else:
        confusion_section = ""

    return USER_SYSTEM_PROMPT.format(
        user_role=rubric.get("user_role", "a user"),
        user_goal=rubric.get("user_goal", "get help with a task"),
        user_behavior=rubric.get("user_behavior", "asks questions naturally"),
        confusion_section=confusion_section,
    )


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def run_simulation(
    probe: dict,
    test_provider,
    test_model: str,
    user_provider,
    user_model: str,
    test_temperature: float = 0.0,
    user_temperature: float = 0.7,
    verbose: bool = False,
) -> SimulationResult:
    """Run a simulated multi-turn conversation.

    Args:
        probe: Probe dict with type="simulated" and simulation_rubric
        test_provider: Provider instance for the model under test
        test_model: Model identifier for the model under test
        user_provider: Provider instance for the simulated user
        user_model: Model identifier for the simulated user
        test_temperature: Temperature for model under test
        user_temperature: Temperature for simulated user (higher = more natural)
        verbose: Print conversation as it happens
    """
    probe_id = probe.get("id", "unknown")
    rubric = probe.get("simulation_rubric", {})
    max_turns = rubric.get("max_turns", 6)
    opening_message = rubric.get("opening_message", "Hi, I need help.")

    # Build system prompts
    user_system = build_user_system_prompt(rubric)
    test_system = probe.get("system", "")

    # Conversation state
    conversation: list[dict] = []
    # Messages for the model under test
    test_messages: list[Message] = []
    # Messages for the simulated user (includes assistant's responses from its perspective)
    user_messages: list[Message] = [Message(role="system", content=user_system)]

    if test_system:
        test_messages.append(Message(role="system", content=test_system))

    total_latency = 0.0
    turns_completed = 0

    # Opening message from simulated user
    current_user_msg = opening_message

    try:
        for turn_num in range(max_turns):
            # Record user message
            conversation.append({"role": "user", "content": current_user_msg})
            test_messages.append(Message(role="user", content=current_user_msg))
            # For user model context: the user said this, assistant will respond
            user_messages.append(Message(role="assistant", content=current_user_msg))

            if verbose:
                print(f"\n[Turn {turn_num + 1}/{max_turns}]")
                print(f"  User: {current_user_msg}")

            # Get response from model under test
            test_request = CompletionRequest(
                messages=test_messages,
                model=test_model,
                temperature=test_temperature,
            )
            test_response = test_provider.complete(test_request)
            total_latency += test_response.latency_ms

            assistant_msg = test_response.content
            conversation.append({"role": "assistant", "content": assistant_msg})
            test_messages.append(Message(role="assistant", content=assistant_msg))

            if verbose:
                # Truncate long responses for display
                display = assistant_msg[:200] + "..." if len(assistant_msg) > 200 else assistant_msg
                print(f"  Assistant: {display}")

            turns_completed = turn_num + 1

            # If this is the last turn, don't generate another user message
            if turn_num >= max_turns - 1:
                break

            # Generate next user message from simulated user
            user_messages.append(Message(role="user", content=assistant_msg))

            user_request = CompletionRequest(
                messages=user_messages,
                model=user_model,
                temperature=user_temperature,
                max_tokens=256,
            )
            user_response = user_provider.complete(user_request)
            current_user_msg = user_response.content.strip()

            # Basic sanity: if the simulated user produces nothing, stop
            if not current_user_msg:
                if verbose:
                    print("  (Simulated user produced empty message, stopping)")
                break

    except Exception as e:
        return SimulationResult(
            probe_id=probe_id,
            simulation=True,
            conversation=conversation,
            turns_completed=turns_completed,
            max_turns=max_turns,
            total_latency_ms=total_latency,
            user_model=user_model,
            error=str(e),
        )

    return SimulationResult(
        probe_id=probe_id,
        simulation=True,
        conversation=conversation,
        turns_completed=turns_completed,
        max_turns=max_turns,
        total_latency_ms=total_latency,
        user_model=user_model,
    )


def simulation_result_to_probe_result(
    sim_result: SimulationResult,
    probe: dict,
) -> dict:
    """Convert a SimulationResult to the standard probe_result format for assessment_runner."""
    from tools.evaluators import evaluate

    # Build the full response text for evaluation (all assistant messages concatenated)
    assistant_messages = [
        turn["content"] for turn in sim_result.conversation if turn["role"] == "assistant"
    ]
    full_response = "\n\n".join(assistant_messages)

    # Evaluate each criterion against the full conversation
    criteria_results = {}
    evidence = {}

    evaluation = probe.get("evaluation", {})
    for criterion_id, spec in evaluation.items():
        passed, ev = evaluate(full_response, spec)
        criteria_results[criterion_id] = passed
        evidence[criterion_id] = ev

    return {
        "probe_id": sim_result.probe_id,
        "input": sim_result.conversation,
        "response": full_response,
        "criteria_results": criteria_results,
        "evidence": evidence,
        "latency_ms": sim_result.total_latency_ms,
        "simulation": True,
        "turns_completed": sim_result.turns_completed,
        "max_turns": sim_result.max_turns,
        "user_model": sim_result.user_model,
        "conversation": sim_result.conversation,
        "error": sim_result.error,
    }


# ---------------------------------------------------------------------------
# CLI (standalone mode)
# ---------------------------------------------------------------------------

def _init_provider(provider_name: str, base_url: Optional[str] = None):
    """Initialize a provider with API key from environment."""
    env_var = f"{provider_name.upper()}_API_KEY"
    api_key = os.environ.get(env_var, "")
    if not api_key:
        print(f"Error: Set {env_var} environment variable", file=sys.stderr)
        sys.exit(1)
    return get_provider(provider_name, api_key=api_key, base_url=base_url)


def main():
    parser = argparse.ArgumentParser(
        description="AI-SETT Conversation Simulator — Haiku-as-user for multi-turn probes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a simulated conversation
  %(prog)s --probe probes/interaction/debugging_sim.yaml \\
    --provider anthropic --model claude-sonnet-4-20250514 \\
    --user-provider anthropic --user-model claude-3-5-haiku-20241022 \\
    --verbose

  # With custom temperatures
  %(prog)s --probe probes/interaction/sim_probe.yaml \\
    --provider openai --model gpt-4o \\
    --user-provider anthropic --user-model claude-3-5-haiku-20241022 \\
    --temperature 0.0 --user-temperature 0.8
        """,
    )
    parser.add_argument("--probe", required=True, help="Path to probe YAML with simulation_rubric")
    parser.add_argument("--provider", required=True, help="Provider for model under test")
    parser.add_argument("--model", required=True, help="Model under test")
    parser.add_argument("--user-provider", required=True, help="Provider for simulated user")
    parser.add_argument("--user-model", required=True, help="Model for simulated user")
    parser.add_argument("--base-url", help="Custom base URL for model under test")
    parser.add_argument("--user-base-url", help="Custom base URL for simulated user")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for model under test (default: 0.0)")
    parser.add_argument("--user-temperature", type=float, default=0.7,
                        help="Temperature for simulated user (default: 0.7)")
    parser.add_argument("--verbose", action="store_true", help="Print conversation as it happens")

    args = parser.parse_args()

    # Load probe
    probe = load_simulated_probe(args.probe)
    print(f"Loaded simulated probe: {probe.get('id', '?')} — {probe.get('name', '?')}")

    rubric = probe.get("simulation_rubric", {})
    print(f"  Role: {rubric.get('user_role', '?')}")
    print(f"  Goal: {rubric.get('user_goal', '?')}")
    print(f"  Max turns: {rubric.get('max_turns', 6)}")

    # Initialize providers
    test_provider = _init_provider(args.provider, args.base_url)
    user_provider = _init_provider(args.user_provider, args.user_base_url)

    print(f"\nModel under test: {args.provider}/{args.model}")
    print(f"Simulated user: {args.user_provider}/{args.user_model}")
    print("---")

    # Run simulation
    result = run_simulation(
        probe=probe,
        test_provider=test_provider,
        test_model=args.model,
        user_provider=user_provider,
        user_model=args.user_model,
        test_temperature=args.temperature,
        user_temperature=args.user_temperature,
        verbose=args.verbose,
    )

    # Print results
    print(f"\n--- Simulation Complete ---")
    print(f"Turns completed: {result.turns_completed}/{result.max_turns}")
    print(f"Total latency: {result.total_latency_ms:.0f}ms")

    if result.error:
        print(f"Error: {result.error}")

    if not args.verbose:
        print("\nConversation:")
        for turn in result.conversation:
            role = turn["role"].upper()
            content = turn["content"]
            print(f"  [{role}]: {content[:150]}{'...' if len(content) > 150 else ''}")

    # If the probe has evaluation criteria, evaluate
    if probe.get("evaluation"):
        probe_result = simulation_result_to_probe_result(result, probe)
        print(f"\n--- Evaluation ---")
        for cid, passed in probe_result["criteria_results"].items():
            status = "PASS" if passed else "FAIL"
            ev = probe_result["evidence"].get(cid, "")
            print(f"  {cid}: {status} — {ev}")

    # Output JSON
    output = {
        "probe_id": result.probe_id,
        "simulation": True,
        "turns_completed": result.turns_completed,
        "max_turns": result.max_turns,
        "total_latency_ms": result.total_latency_ms,
        "user_model": result.user_model,
        "conversation": result.conversation,
    }
    if result.error:
        output["error"] = result.error

    print(f"\nJSON output:")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
