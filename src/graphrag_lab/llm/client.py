"""Structured LLM calls with a content-addressed disk cache.

Every LLM use in the pipeline (extraction, query classification fallback,
claim adjudication, synthesis) goes through `StructuredLLM.generate`, which
returns a validated pydantic object via Instructor.

Responses are cached on disk keyed by a hash of everything that determines
the output: model, temperature, max_tokens, system prompt, user message, and
the response model's JSON schema. Re-running an experiment therefore costs
nothing and returns identical results, and a prompt or schema change
naturally invalidates exactly the affected entries.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Generic, TypeVar

import anthropic
import instructor
import structlog
from pydantic import BaseModel

from graphrag_lab.core.config import get_settings

logger = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMResponse(Generic[T]):
    value: T
    # Input + output tokens of the original (uncached) call; 0 if unreported.
    tokens: int
    cached: bool


def _total_tokens(completion: object) -> int:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return 0
    return int(getattr(usage, "input_tokens", 0) or 0) + int(
        getattr(usage, "output_tokens", 0) or 0
    )


def cache_key(
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    system: str,
    user: str,
    response_model: type[BaseModel],
) -> str:
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system": system,
        "user": user,
        "schema": response_model.model_json_schema(),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class StructuredLLM:
    """Instructor-backed structured generation with an optional disk cache.

    `client` is injectable for tests; otherwise an Anthropic client is built
    lazily on first uncached call, so cache-only runs need no API key.
    """

    def __init__(
        self,
        cache_dir: Path | None,
        temperature: float = 0.0,
        client: instructor.AsyncInstructor | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._temperature = temperature
        self._client = client

    def _get_client(self) -> instructor.AsyncInstructor:
        if self._client is None:
            api_key = get_settings().anthropic_api_key
            if api_key is None:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set and this LLM call is not in the cache"
                )
            self._client = instructor.from_anthropic(
                anthropic.AsyncAnthropic(api_key=api_key.get_secret_value())
            )
        return self._client

    def _path_for(self, key: str) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / key[:2] / f"{key}.json"

    def _read(self, path: Path | None, response_model: type[T]) -> LLMResponse[T] | None:
        if path is None or not path.exists():
            return None
        try:
            record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            value = response_model.model_validate(record["value"])
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("llm_cache.unreadable_entry", path=str(path), error=str(exc))
            return None
        return LLMResponse(value=value, tokens=int(record.get("tokens", 0)), cached=True)

    @staticmethod
    def _write(path: Path | None, model: str, value: BaseModel, tokens: int) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"model": model, "tokens": tokens, "value": value.model_dump(mode="json")}
        # Write-then-rename so a crash never leaves a truncated cache entry.
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False)
        os.replace(tmp, path)

    async def generate(
        self,
        *,
        model: str,
        system: str,
        user: str,
        response_model: type[T],
        max_tokens: int,
    ) -> LLMResponse[T]:
        key = cache_key(
            model=model,
            temperature=self._temperature,
            max_tokens=max_tokens,
            system=system,
            user=user,
            response_model=response_model,
        )
        path = self._path_for(key)
        hit = self._read(path, response_model)
        if hit is not None:
            return hit

        value, completion = await self._get_client().chat.completions.create_with_completion(
            model=model,
            max_tokens=max_tokens,
            temperature=self._temperature,
            system=system,
            response_model=response_model,
            messages=[{"role": "user", "content": user}],
        )
        tokens = _total_tokens(completion)
        self._write(path, model, value, tokens)
        return LLMResponse(value=value, tokens=tokens, cached=False)


@lru_cache
def get_llm() -> StructuredLLM:
    """Process-wide StructuredLLM configured from settings."""
    settings = get_settings()
    cache_dir = settings.llm_cache_dir if settings.llm_cache_enabled else None
    return StructuredLLM(cache_dir=cache_dir, temperature=settings.llm_temperature)
