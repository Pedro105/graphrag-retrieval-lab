"""StructuredLLM: cache hits skip the API, cache keys track every input that
shapes the output, and a disabled cache always calls through. Fully offline."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel

from graphrag_lab.llm.client import StructuredLLM, cache_key


class Answer(BaseModel):
    text: str


class OtherAnswer(BaseModel):
    text: str
    score: float = 0.0


def _fake_client(text: str = "hi", input_tokens: int = 10, output_tokens: int = 5) -> MagicMock:
    completion = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    )
    client = MagicMock()
    client.chat.completions.create_with_completion = AsyncMock(
        return_value=(Answer(text=text), completion)
    )
    return client


CALL = dict(model="m", system="sys", user="hello", response_model=Answer, max_tokens=64)


async def test_miss_calls_api_then_hit_reads_disk(tmp_path: Path) -> None:
    client = _fake_client()
    llm = StructuredLLM(cache_dir=tmp_path, client=client)

    first = await llm.generate(**CALL)
    second = await llm.generate(**CALL)

    assert first.value == Answer(text="hi") and first.cached is False and first.tokens == 15
    assert second.value == Answer(text="hi") and second.cached is True and second.tokens == 15
    client.chat.completions.create_with_completion.assert_awaited_once()


async def test_cache_survives_a_new_instance(tmp_path: Path) -> None:
    await StructuredLLM(cache_dir=tmp_path, client=_fake_client()).generate(**CALL)

    fresh_client = _fake_client(text="different")
    hit = await StructuredLLM(cache_dir=tmp_path, client=fresh_client).generate(**CALL)

    assert hit.value.text == "hi"
    fresh_client.chat.completions.create_with_completion.assert_not_awaited()


async def test_disabled_cache_always_calls_through(tmp_path: Path) -> None:
    client = _fake_client()
    llm = StructuredLLM(cache_dir=None, client=client)

    await llm.generate(**CALL)
    await llm.generate(**CALL)

    assert client.chat.completions.create_with_completion.await_count == 2
    assert not any(tmp_path.iterdir())


async def test_corrupt_entry_is_treated_as_a_miss(tmp_path: Path) -> None:
    client = _fake_client()
    llm = StructuredLLM(cache_dir=tmp_path, client=client)
    await llm.generate(**CALL)
    (entry,) = tmp_path.glob("*/*.json")
    entry.write_text("{not json")

    result = await llm.generate(**CALL)

    assert result.cached is False
    assert client.chat.completions.create_with_completion.await_count == 2


async def test_temperature_is_passed_to_the_api(tmp_path: Path) -> None:
    client = _fake_client()
    await StructuredLLM(cache_dir=None, temperature=0.0, client=client).generate(**CALL)
    assert client.chat.completions.create_with_completion.await_args.kwargs["temperature"] == 0.0


def test_cache_key_changes_with_every_output_shaping_input() -> None:
    base = dict(
        model="m", temperature=0.0, max_tokens=64, system="s", user="u", response_model=Answer
    )
    key = cache_key(**base)  # type: ignore[arg-type]
    variants = [
        {"model": "m2"},
        {"temperature": 0.5},
        {"max_tokens": 128},
        {"system": "s2"},
        {"user": "u2"},
        {"response_model": OtherAnswer},
    ]
    for change in variants:
        assert cache_key(**{**base, **change}) != key, change  # type: ignore[arg-type]
    assert cache_key(**base) == key  # type: ignore[arg-type]
