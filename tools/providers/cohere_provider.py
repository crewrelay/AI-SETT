"""Cohere Chat API v2 provider."""

from __future__ import annotations

from typing import Optional

import httpx

from . import register
from .base import CompletionRequest, CompletionResponse, ModelProvider


@register("cohere")
class CohereProvider(ModelProvider):
    """Cohere v2 chat API — uses message + chat_history format."""

    DEFAULT_BASE_URL = "https://api.cohere.com"

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        url = f"{self.base_url}/v2/chat"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build chat_history and extract the last user message
        chat_history = []
        preamble = None
        last_user_message = ""

        for m in request.messages:
            if m.role == "system":
                preamble = m.content
            elif m.role == "user":
                # Push previous user message to history if there was one
                if last_user_message:
                    chat_history.append({"role": "user", "content": last_user_message})
                last_user_message = m.content
            elif m.role == "assistant":
                chat_history.append({"role": "assistant", "content": m.content})

        payload = {
            "model": request.model,
            "message": last_user_message,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if chat_history:
            payload["chat_history"] = chat_history
        if preamble:
            payload["preamble"] = preamble
        if request.stop:
            payload["stop_sequences"] = request.stop

        def _call():
            resp = httpx.post(url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            return resp.json()

        data, latency_ms = self._timed(_call)

        text = data.get("text", "")
        meta = data.get("meta", {})
        tokens = meta.get("tokens", {})

        return CompletionResponse(
            content=text,
            model=request.model,
            latency_ms=latency_ms,
            input_tokens=tokens.get("input_tokens"),
            output_tokens=tokens.get("output_tokens"),
            finish_reason=data.get("finish_reason"),
            raw=data,
        )
