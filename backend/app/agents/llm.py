from __future__ import annotations

import json
import re
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from myagent.backend.app.config import Settings

T = TypeVar("T", bound=BaseModel)
LLM_RETRY_ATTEMPTS = 3
_DEFAULT_LLM_CONCURRENCY = 5
LLMLane = str
MAIN_LANE = "main"
QA_LANE = "qa"
RETRY_LANE = "retry"
_llm_semaphores: dict[LLMLane, asyncio.Semaphore] = {}
_llm_semaphore_limits: dict[LLMLane, int] = {}
_llm_semaphore_loops: dict[LLMLane, asyncio.AbstractEventLoop] = {}
_llm_global_semaphore: asyncio.Semaphore | None = None
_llm_global_limit: int = _DEFAULT_LLM_CONCURRENCY
_llm_global_loop: asyncio.AbstractEventLoop | None = None


@dataclass(frozen=True)
class ChatModelBundle:
    primary: object | None
    fallback: object | None = None

    @property
    def available(self) -> bool:
        return self.primary is not None


def configure_llm_runtime(
    *,
    max_concurrency: int | None = None,
    main_max_concurrency: int | None = None,
    qa_max_concurrency: int | None = None,
    retry_max_concurrency: int | None = None,
) -> None:
    """Configure process-local LLM runtime controls."""
    global _llm_global_semaphore, _llm_global_limit, _llm_global_loop
    global_limit = max(1, int(max_concurrency or _DEFAULT_LLM_CONCURRENCY))
    _llm_global_limit = global_limit
    _llm_global_semaphore = asyncio.Semaphore(global_limit)
    try:
        _llm_global_loop = asyncio.get_running_loop()
    except RuntimeError:
        _llm_global_loop = None
    if main_max_concurrency is None:
        main_limit = min(3, global_limit)
    else:
        main_limit = min(global_limit, max(1, int(main_max_concurrency)))
    qa_limit = min(global_limit, max(1, int(qa_max_concurrency or 1)))
    retry_limit = min(global_limit, max(1, int(retry_max_concurrency or 1)))
    _configure_lane(MAIN_LANE, main_limit)
    _configure_lane(QA_LANE, qa_limit)
    _configure_lane(RETRY_LANE, retry_limit)


def _default_lane_limit(lane: LLMLane) -> int:
    if lane in {QA_LANE, RETRY_LANE}:
        return 1
    return _DEFAULT_LLM_CONCURRENCY


def _configure_lane(lane: LLMLane, limit: int, *, force: bool = False) -> None:
    if force or lane not in _llm_semaphores or limit != _llm_semaphore_limits.get(lane):
        _llm_semaphores[lane] = asyncio.Semaphore(limit)
        _llm_semaphore_limits[lane] = limit
        try:
            _llm_semaphore_loops[lane] = asyncio.get_running_loop()
        except RuntimeError:
            _llm_semaphore_loops.pop(lane, None)


def _runtime_semaphore(lane: LLMLane = MAIN_LANE) -> asyncio.Semaphore:
    if not _llm_semaphores:
        configure_llm_runtime(max_concurrency=_DEFAULT_LLM_CONCURRENCY)
    if lane not in _llm_semaphores:
        _configure_lane(lane, _default_lane_limit(lane))
    loop = asyncio.get_running_loop()
    if _llm_semaphore_loops.get(lane) is not loop:
        _configure_lane(
            lane,
            _llm_semaphore_limits.get(lane, _default_lane_limit(lane)),
            force=True,
        )
    return _llm_semaphores[lane]


def _runtime_global_semaphore() -> asyncio.Semaphore:
    global _llm_global_semaphore, _llm_global_loop
    if _llm_global_semaphore is None:
        configure_llm_runtime(max_concurrency=_DEFAULT_LLM_CONCURRENCY)
    loop = asyncio.get_running_loop()
    if _llm_global_loop is not loop:
        _llm_global_semaphore = asyncio.Semaphore(_llm_global_limit)
        _llm_global_loop = loop
    assert _llm_global_semaphore is not None
    return _llm_global_semaphore


@asynccontextmanager
async def _runtime_gate(lane: LLMLane = MAIN_LANE):
    async with _runtime_global_semaphore():
        async with _runtime_semaphore(lane):
            yield


def get_chat_model(settings: Settings) -> ChatModelBundle | None:
    primary = _build_chat_model(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        settings=settings,
    )
    if primary is None:
        return None
    fallback = _build_chat_model(
        api_key=settings.openai_fallback_api_key,
        base_url=settings.openai_fallback_base_url,
        model=settings.openai_fallback_model or settings.openai_model,
        settings=settings,
    )
    return ChatModelBundle(primary=primary, fallback=fallback)


def _build_chat_model(*, api_key: str | None, base_url: str | None, model: str | None, settings: Settings):
    if not api_key:
        return None
    try:
        import httpx
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None

    kwargs = {
        "model": model or settings.openai_model,
        "api_key": api_key,
        "temperature": 0.2,
        "timeout": settings.openai_timeout_seconds,
        "max_retries": settings.openai_max_retries,
    }
    if base_url:
        kwargs["base_url"] = base_url
    if not settings.openai_trust_env:
        kwargs["http_async_client"] = httpx.AsyncClient(
            trust_env=False,
            timeout=settings.openai_timeout_seconds,
        )
    return ChatOpenAI(**kwargs)


def friendly_llm_error(exc: Exception) -> str:
    text = str(exc)
    lower = text.lower()
    if "apiconnectionerror" in lower or "connection error" in lower or "endofstream" in lower:
        return (
            "模型服务连接失败。请检查 OPENAI_BASE_URL、网络代理、VPN/系统代理和网关可用性；"
            "如果当前系统代理会断开 TLS，请在 backend/.env 设置 OPENAI_TRUST_ENV=false 后重启后端。"
        )
    if "response_format" in lower:
        return "模型网关不支持 response_format，系统已尝试普通 JSON 模式降级。"
    if "choices" in lower and "attribute" in lower:
        return (
            "模型网关返回格式不是标准 OpenAI Chat Completions 响应，解析器没有拿到 choices 字段；"
            "系统会优先尝试普通 JSON 模式降级。"
        )
    return text


async def structured_ainvoke(model, schema: type[T], messages: list[tuple[str, str]], *, lane: LLMLane = MAIN_LANE) -> T:
    async with _runtime_gate(lane):
        try:
            structured = model.with_structured_output(schema)
            result = await structured.ainvoke(messages)
        except Exception as exc:
            if not _should_try_json_mode_fallback(exc):
                raise
            result = await _json_mode_fallback(model, schema, messages)
    return _coerce_structured_result(schema, result)


def _coerce_structured_result(schema: type[T], result) -> T:
    if isinstance(result, schema):
        return result
    if isinstance(result, str):
        try:
            result = _extract_json_object(result)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Structured output returned non-JSON string for {schema.__name__}") from exc
    return schema.model_validate(result)


async def retry_structured_ainvoke(
    model,
    schema: type[T],
    messages: list[tuple[str, str]],
    *,
    attempts: int = LLM_RETRY_ATTEMPTS,
    lane: LLMLane = MAIN_LANE,
) -> T:
    return await _with_fallback_models(
        model,
        lambda active_model: _retry_structured_single_model(active_model, schema, messages, attempts=attempts, lane=lane),
    )


async def _retry_structured_single_model(
    model,
    schema: type[T],
    messages: list[tuple[str, str]],
    *,
    attempts: int,
    lane: LLMLane,
) -> T:
    last_exc: Exception | None = None
    for index in range(max(1, attempts)):
        try:
            attempt_lane = lane if index == 0 else RETRY_LANE
            return await structured_ainvoke(model, schema, messages, lane=attempt_lane)
        except Exception as exc:
            last_exc = exc
            if index + 1 >= attempts:
                break
            await asyncio.sleep(min(8.0, 0.75 * (2**index)))
    assert last_exc is not None
    raise last_exc


async def retry_ainvoke(
    model,
    messages: list[tuple[str, str]],
    *,
    attempts: int = LLM_RETRY_ATTEMPTS,
    lane: LLMLane = MAIN_LANE,
):
    return await _with_fallback_models(
        model,
        lambda active_model: _retry_text_single_model(active_model, messages, attempts=attempts, lane=lane),
    )


async def _retry_text_single_model(model, messages: list[tuple[str, str]], *, attempts: int, lane: LLMLane):
    last_exc: Exception | None = None
    for index in range(max(1, attempts)):
        try:
            attempt_lane = lane if index == 0 else RETRY_LANE
            async with _runtime_gate(attempt_lane):
                return await model.ainvoke(messages)
        except Exception as exc:
            last_exc = exc
            if index + 1 >= attempts:
                break
            await asyncio.sleep(min(8.0, 0.75 * (2**index)))
    assert last_exc is not None
    raise last_exc


async def _with_fallback_models(model, call):
    bundle = model if isinstance(model, ChatModelBundle) else ChatModelBundle(primary=model)
    if bundle.primary is None:
        raise RuntimeError("No LLM model is configured.")
    try:
        return await call(bundle.primary)
    except Exception as primary_exc:
        if bundle.fallback is None:
            raise
        try:
            return await call(bundle.fallback)
        except Exception as fallback_exc:
            raise RuntimeError(
                "Primary LLM failed after configured attempts; fallback LLM also failed after configured attempts. "
                f"Primary error: {friendly_llm_error(primary_exc)}; fallback error: {friendly_llm_error(fallback_exc)}"
            ) from fallback_exc


def _is_response_format_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "response_format" in text or "structured output" in text


def _is_chat_completion_shape_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "choices" in text and "attribute" in text


def _is_structured_output_object_shape_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "model_dump" in text and "attribute" in text


def _is_json_parse_error(exc: Exception) -> bool:
    return isinstance(exc, json.JSONDecodeError)


def _should_try_json_mode_fallback(exc: Exception) -> bool:
    return (
        _is_response_format_error(exc)
        or _is_chat_completion_shape_error(exc)
        or _is_structured_output_object_shape_error(exc)
        or _is_json_parse_error(exc)
    )


async def _json_mode_fallback(model, schema: type[T], messages: list[tuple[str, str]]) -> dict:
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    fallback_messages = [
        (
            "system",
            "You must return only valid JSON. Do not wrap it in markdown. "
            "The JSON must satisfy this schema:\n"
            f"{schema_json}",
        ),
        *messages,
    ]
    response = await model.ainvoke(fallback_messages)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "\n".join(str(item) for item in content)
    return _extract_json_object(str(content))


def _extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise
