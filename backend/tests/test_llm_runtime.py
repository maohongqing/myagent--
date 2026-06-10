from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import BaseModel

from myagent.backend.app.agents.llm import (
    QA_LANE,
    ChatModelBundle,
    configure_llm_runtime,
    retry_ainvoke,
    retry_structured_ainvoke,
    structured_ainvoke,
)


@pytest.mark.asyncio
async def test_retry_ainvoke_respects_global_concurrency_limit():
    active = 0
    peak = 0

    class FakeModel:
        async def ainvoke(self, _messages):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return "ok"

    configure_llm_runtime(max_concurrency=5)
    await asyncio.gather(*(retry_ainvoke(FakeModel(), [("user", str(index))], attempts=1) for index in range(12)))

    assert peak <= 5


@pytest.mark.asyncio
async def test_retry_ainvoke_uses_fallback_after_primary_failure_then_primary_next_call():
    calls: list[str] = []

    class FailingPrimary:
        async def ainvoke(self, _messages):
            calls.append("primary")
            raise RuntimeError("primary unavailable")

    class FallbackModel:
        async def ainvoke(self, _messages):
            calls.append("fallback")
            return "ok"

    bundle = ChatModelBundle(primary=FailingPrimary(), fallback=FallbackModel())

    assert await retry_ainvoke(bundle, [("user", "first")], attempts=1) == "ok"
    assert await retry_ainvoke(bundle, [("user", "second")], attempts=1) == "ok"
    assert calls == ["primary", "fallback", "primary", "fallback"]


@pytest.mark.asyncio
async def test_retry_ainvoke_tries_primary_three_then_fallback_three_before_failing():
    calls: list[str] = []

    class FailingPrimary:
        async def ainvoke(self, _messages):
            calls.append("primary")
            raise RuntimeError("primary unavailable")

    class FailingFallback:
        async def ainvoke(self, _messages):
            calls.append("fallback")
            raise RuntimeError("fallback unavailable")

    bundle = ChatModelBundle(primary=FailingPrimary(), fallback=FailingFallback())

    with pytest.raises(RuntimeError, match="Primary LLM failed after configured attempts"):
        await retry_ainvoke(bundle, [("user", "fail")])

    assert calls == ["primary", "primary", "primary", "fallback", "fallback", "fallback"]


@pytest.mark.asyncio
async def test_llm_runtime_uses_separate_main_and_qa_lanes():
    active_main = 0
    active_qa = 0
    peak_main = 0
    peak_qa = 0
    peak_total = 0

    class FakeModel:
        def __init__(self, lane: str):
            self.lane = lane

        async def ainvoke(self, _messages):
            nonlocal active_main, active_qa, peak_main, peak_qa, peak_total
            if self.lane == QA_LANE:
                active_qa += 1
                peak_qa = max(peak_qa, active_qa)
            else:
                active_main += 1
                peak_main = max(peak_main, active_main)
            peak_total = max(peak_total, active_main + active_qa)
            await asyncio.sleep(0.02)
            if self.lane == QA_LANE:
                active_qa -= 1
            else:
                active_main -= 1
            return "ok"

    configure_llm_runtime(main_max_concurrency=4, qa_max_concurrency=1)
    await asyncio.gather(
        *(retry_ainvoke(FakeModel("main"), [("user", str(index))], attempts=1) for index in range(12)),
        *(retry_ainvoke(FakeModel(QA_LANE), [("user", str(index))], attempts=1, lane=QA_LANE) for index in range(4)),
    )

    assert peak_main <= 4
    assert peak_qa <= 1
    assert peak_total <= 5


@pytest.mark.asyncio
async def test_llm_runtime_global_gate_caps_total_even_when_lane_limits_are_higher():
    active = 0
    peak = 0

    class FakeModel:
        async def ainvoke(self, _messages):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return "ok"

    configure_llm_runtime(max_concurrency=2, main_max_concurrency=5, qa_max_concurrency=5, retry_max_concurrency=5)
    await asyncio.gather(
        *(retry_ainvoke(FakeModel(), [("user", f"main-{index}")], attempts=1) for index in range(4)),
        *(retry_ainvoke(FakeModel(), [("user", f"qa-{index}")], attempts=1, lane=QA_LANE) for index in range(4)),
    )

    assert peak <= 2


@pytest.mark.asyncio
async def test_failed_structured_retry_uses_low_concurrency_then_recovers_normal_lane():
    class MiniSchema(BaseModel):
        value: str

    retry_active = 0
    retry_peak = 0
    normal_active = 0
    normal_peak = 0
    attempts_by_key: dict[str, int] = {}

    class FlakyStructuredModel:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, messages):
            nonlocal retry_active, retry_peak
            key = messages[-1][1]
            attempts_by_key[key] = attempts_by_key.get(key, 0) + 1
            if attempts_by_key[key] == 1:
                raise RuntimeError("transient timeout")
            retry_active += 1
            retry_peak = max(retry_peak, retry_active)
            await asyncio.sleep(0.02)
            retry_active -= 1
            return {"value": key}

    class NormalStructuredModel:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, messages):
            nonlocal normal_active, normal_peak
            normal_active += 1
            normal_peak = max(normal_peak, normal_active)
            await asyncio.sleep(0.02)
            normal_active -= 1
            return {"value": messages[-1][1]}

    configure_llm_runtime(main_max_concurrency=4, qa_max_concurrency=1, retry_max_concurrency=1)
    await asyncio.gather(
        *(retry_structured_ainvoke(FlakyStructuredModel(), MiniSchema, [("user", str(index))], attempts=2) for index in range(5))
    )
    await asyncio.gather(
        *(retry_structured_ainvoke(NormalStructuredModel(), MiniSchema, [("user", str(index))], attempts=1) for index in range(4))
    )

    assert retry_peak <= 1
    assert normal_peak > 1


@pytest.mark.asyncio
async def test_structured_ainvoke_falls_back_to_json_mode_for_choices_shape_error():
    class MiniSchema(BaseModel):
        value: str

    class StructuredWrapper:
        async def ainvoke(self, _messages):
            raise AttributeError("'str' object has no attribute 'choices'")

    class GatewayModel:
        def with_structured_output(self, _schema):
            return StructuredWrapper()

        async def ainvoke(self, _messages):
            return '{"value": "ok"}'

    result = await structured_ainvoke(GatewayModel(), MiniSchema, [("user", "return json")])

    assert result.value == "ok"


@pytest.mark.asyncio
async def test_structured_ainvoke_falls_back_to_json_mode_for_model_dump_shape_error():
    class MiniSchema(BaseModel):
        value: str

    class StructuredWrapper:
        async def ainvoke(self, _messages):
            raise AttributeError("'str' object has no attribute 'model_dump'")

    class GatewayModel:
        def with_structured_output(self, _schema):
            return StructuredWrapper()

        async def ainvoke(self, _messages):
            return '{"value": "ok"}'

    result = await structured_ainvoke(GatewayModel(), MiniSchema, [("user", "return json")])

    assert result.value == "ok"


@pytest.mark.asyncio
async def test_structured_ainvoke_falls_back_to_json_mode_for_empty_structured_response():
    class MiniSchema(BaseModel):
        value: str

    class StructuredWrapper:
        async def ainvoke(self, _messages):
            raise json.JSONDecodeError("Expecting value", "", 0)

    class GatewayModel:
        def with_structured_output(self, _schema):
            return StructuredWrapper()

        async def ainvoke(self, _messages):
            return '{"value": "ok"}'

    result = await structured_ainvoke(GatewayModel(), MiniSchema, [("user", "return json")])

    assert result.value == "ok"


@pytest.mark.asyncio
async def test_structured_ainvoke_accepts_dict_result():
    class MiniSchema(BaseModel):
        value: str

    class DictStructuredModel:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _messages):
            return {"value": "ok"}

    result = await structured_ainvoke(DictStructuredModel(), MiniSchema, [("user", "return json")])

    assert result.value == "ok"


@pytest.mark.asyncio
async def test_structured_ainvoke_parses_json_string_result():
    class MiniSchema(BaseModel):
        value: str

    class StringStructuredModel:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _messages):
            return '{"value": "ok"}'

    result = await structured_ainvoke(StringStructuredModel(), MiniSchema, [("user", "return json")])

    assert result.value == "ok"


@pytest.mark.asyncio
async def test_retry_structured_ainvoke_rejects_non_json_string_result():
    class MiniSchema(BaseModel):
        value: str

    calls = 0

    class TextStructuredModel:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _messages):
            nonlocal calls
            calls += 1
            return "I cannot produce that JSON."

    with pytest.raises(ValueError, match="Structured output returned non-JSON string for MiniSchema"):
        await retry_structured_ainvoke(TextStructuredModel(), MiniSchema, [("user", "return json")], attempts=2)

    assert calls == 2
