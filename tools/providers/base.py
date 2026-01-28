"""Abstract base for model providers and shared data structures."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class CompletionRequest:
    """Request to a model provider."""
    messages: list[Message]
    model: str
    temperature: float = 0.0
    max_tokens: int = 2048
    stop: list[str] = field(default_factory=list)


@dataclass
class CompletionResponse:
    """Response from a model provider."""
    content: str
    model: str
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    raw: Optional[dict] = None


class ModelProvider(ABC):
    """Abstract base class for model providers.

    All providers use httpx for HTTP calls — no provider SDKs.
    """

    name: str = ""

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.extra = kwargs

    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send a completion request and return the response."""
        ...

    def _timed(self, fn):
        """Helper to time a callable and return (result, latency_ms)."""
        start = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - start) * 1000
        return result, elapsed
