"""Mistral AI provider (OpenAI-compatible format)."""

from __future__ import annotations

from typing import Optional

import httpx

from . import register
from .base import CompletionRequest, CompletionResponse, ModelProvider


@register("mistral")
class MistralProvider(ModelProvider):
    """Mistral chat completions API (OpenAI-compatible)."""

    DEFAULT_BASE_URL = "https://api.mistral.ai/v1"

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.stop:
            payload["stop"] = request.stop

        def _call():
            resp = httpx.post(url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            return resp.json()

        data, latency_ms = self._timed(_call)

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return CompletionResponse(
            content=choice["message"]["content"],
            model=data.get("model", request.model),
            latency_ms=latency_ms,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )
