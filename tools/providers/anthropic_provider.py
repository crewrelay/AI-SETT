"""Anthropic Messages API provider."""

from __future__ import annotations

from typing import Optional

import httpx

from . import register
from .base import CompletionRequest, CompletionResponse, ModelProvider


@register("anthropic")
class AnthropicProvider(ModelProvider):
    """Anthropic Messages API — system message extracted to top-level param."""

    DEFAULT_BASE_URL = "https://api.anthropic.com"
    API_VERSION = "2023-06-01"

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        url = f"{self.base_url}/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json",
        }

        # Separate system message from conversation messages
        system_text = None
        messages = []
        for m in request.messages:
            if m.role == "system":
                system_text = m.content
            else:
                messages.append({"role": m.role, "content": m.content})

        payload = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if system_text:
            payload["system"] = system_text
        if request.stop:
            payload["stop_sequences"] = request.stop

        def _call():
            resp = httpx.post(url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            return resp.json()

        data, latency_ms = self._timed(_call)

        # Extract text from content blocks
        content_blocks = data.get("content", [])
        text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
        usage = data.get("usage", {})

        return CompletionResponse(
            content=text,
            model=data.get("model", request.model),
            latency_ms=latency_ms,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            finish_reason=data.get("stop_reason"),
            raw=data,
        )
