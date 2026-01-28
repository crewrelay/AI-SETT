"""Google Gemini API provider."""

from __future__ import annotations

from typing import Optional

import httpx

from . import register
from .base import CompletionRequest, CompletionResponse, ModelProvider


@register("google")
class GoogleProvider(ModelProvider):
    """Google Generative Language API (Gemini)."""

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        url = (
            f"{self.base_url}/v1beta/models/{request.model}"
            f":generateContent?key={self.api_key}"
        )
        headers = {"Content-Type": "application/json"}

        # Map messages to Gemini's contents format
        contents = []
        system_instruction = None

        for m in request.messages:
            if m.role == "system":
                system_instruction = m.content
            else:
                role = "user" if m.role == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": m.content}],
                })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}],
            }
        if request.stop:
            payload["generationConfig"]["stopSequences"] = request.stop

        def _call():
            resp = httpx.post(url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            return resp.json()

        data, latency_ms = self._timed(_call)

        # Extract text from candidates
        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)

        usage = data.get("usageMetadata", {})

        return CompletionResponse(
            content=text,
            model=request.model,
            latency_ms=latency_ms,
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
            finish_reason=candidates[0].get("finishReason") if candidates else None,
            raw=data,
        )
