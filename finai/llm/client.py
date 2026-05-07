"""Thin Anthropic SDK wrapper.

Centralizes:
- API key check + degradation when LLM is disabled
- model selection (default vs fast)
- prompt-cache breakpoint placement (frozen instructions in system, volatile
  per-request data in messages, top-level cache_control)
- structured output via messages.parse + Pydantic
- usage logging so cost stays observable

This wrapper assumes attribution-style requests: short outputs, JSON-shaped,
single-turn. We do NOT enable streaming (max_tokens stays small).
"""
from __future__ import annotations

import logging
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from finai.config import settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """Raised when caller asks for LLM output but it is disabled / not configured."""


class LLMClient:
    def __init__(self) -> None:
        self._client: anthropic.Anthropic | None = None
        self._enabled = settings.llm_enabled and bool(settings.llm_anthropic_api_key)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _ensure(self) -> anthropic.Anthropic:
        if not self._enabled:
            raise LLMUnavailable("LLM disabled or ANTHROPIC_API_KEY missing")
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=settings.llm_anthropic_api_key)
        return self._client

    def parse(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        model: str | None = None,
        max_tokens: int | None = None,
        cache_system: bool = True,
    ) -> T:
        """Single-shot structured-output call.

        `system` is treated as the cache prefix — keep it stable across calls
        in the same batch (e.g. one shared rubric per report). `user` carries
        the volatile per-request payload.
        """
        client = self._ensure()
        mdl = model or settings.llm_model
        sys_block: list[dict] = [{"type": "text", "text": system}]
        if cache_system:
            sys_block[0]["cache_control"] = {"type": "ephemeral"}

        resp = client.messages.parse(
            model=mdl,
            max_tokens=max_tokens or settings.llm_max_tokens,
            system=sys_block,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )

        usage = resp.usage
        log.info(
            "llm.parse model=%s in=%s cached_in=%s cache_read=%s out=%s",
            mdl,
            usage.input_tokens,
            getattr(usage, "cache_creation_input_tokens", 0),
            getattr(usage, "cache_read_input_tokens", 0),
            usage.output_tokens,
        )
        return resp.parsed_output


_default: LLMClient | None = None


def get_client() -> LLMClient:
    global _default
    if _default is None:
        _default = LLMClient()
    return _default
