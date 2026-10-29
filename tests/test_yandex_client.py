import asyncio
import json

import httpx
import pytest

from agent import AgentError, GroqConnectionError, QuotaPauseError
from provider_config import CONFIG_NAMES, WebConfig, load_settings
from yandex_client import YandexClient

FOLDER = "synthetic-folder"
MODEL = f"gpt://{FOLDER}/aliceai-llm-flash"
CONFIG = WebConfig("synthetic-yandex-key", "synthetic-maps", MODEL, "yandex", FOLDER)


@pytest.mark.parametrize("model", ["aliceai-llm-flash", MODEL, MODEL + "/latest"])
def test_yandex_config_needs_no_groq_key_and_does_not_mutate_environment(monkeypatch, tmp_path, model):
    for name in CONFIG_NAMES:
        monkeypatch.delenv(name, raising=False)
    config = load_settings({"LLM_PROVIDER": "yandex", "YANDEX_API_KEY": CONFIG.llm_key,
                            "YANDEX_FOLDER_ID": FOLDER, "YANDEX_MODEL": model,
                            "GOOGLE_MAPS_API_KEY": CONFIG.maps_key}, tmp_path / "absent.env")
    assert config.provider == "yandex" and config.model == MODEL + "/latest"
    assert "synthetic-yandex-key" not in repr(config)
    assert config.headers["x-data-logging-enabled"] == "false"
    assert config.headers["OpenAI-Project"] == FOLDER
    assert config.redact("synthetic-yandex-key synthetic-maps") == "[REDACTED] [REDACTED]"
    import os
    assert "YANDEX_API_KEY" not in os.environ


def test_explicit_yandex_selection_does_not_fall_back_to_groq(monkeypatch, tmp_path):
    for name in CONFIG_NAMES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(AgentError, match="YANDEX_API_KEY"):
        load_settings({"LLM_PROVIDER": "yandex", "GROQ_API_KEY": "synthetic-groq",
                       "GOOGLE_MAPS_API_KEY": CONFIG.maps_key}, tmp_path / "absent.env")


def test_yandex_tool_round_trip_uses_expected_headers_fields_and_usage():
    sent = []
    def handler(request):
        assert request.url.host == "ai.api.cloud.yandex.net"
        assert request.headers["Authorization"] == "Bearer synthetic-yandex-key"
        assert request.headers["x-data-logging-enabled"] == "false"
        body = json.loads(request.content)
        sent.append(body)
        assert body["model"] == MODEL and body["max_tokens"] == 1600
        assert not {"max_completion_tokens", "include_reasoning", "reasoning_effort", "store"} & body.keys()
        message = {"role": "assistant", "content": "Синтетический результат", "reasoning": "private-reasoning"}
        if len(sent) == 1:
            message.update(content=None, tool_calls=[{"id": "one", "type": "function", "function": {
                "name": "synthetic_tool", "arguments": "{}",
            }}])
        else:
            assert body["messages"][-1]["tool_call_id"] == "one" and body["tool_choice"] == "none"
        return httpx.Response(200, headers={"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "1h"},
                              json={"choices": [{"finish_reason": "tool_calls" if len(sent) == 1 else "stop", "message": message}],
                                    "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                                              "prompt_tokens_details": {"cached_tokens": 80}}})
    async def run():
        async with httpx.AsyncClient(headers=CONFIG.headers, transport=httpx.MockTransport(handler)) as http:
            llm = YandexClient(http, MODEL)
            tools = [{"type": "function", "function": {"name": "synthetic_tool", "parameters": {"type": "object"}}}]
            first = await llm.complete([], tools)
            final = await llm.complete([first, {"role": "tool", "tool_call_id": "one", "content": "{}"}], tools, False)
            assert "reasoning" not in first and "reasoning" not in final
            assert llm.remaining_tokens is None and llm.reset_at == 0
            assert llm.usage_totals == {"requests": 2, "prompt_tokens": 200, "completion_tokens": 40, "missing_usage": 0}
    asyncio.run(run())
    assert len(sent) == 2


@pytest.mark.parametrize("status,kind", [(400, AgentError), (401, GroqConnectionError), (402, GroqConnectionError),
                                        (403, GroqConnectionError), (404, AgentError), (429, QuotaPauseError)])
def test_yandex_errors_are_safe_name_correct_provider_and_never_auto_retry(status, kind):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(status, headers={"retry-after": "20"}, json={"error": {
            "message": "synthetic-yandex-key private prompt",
        }})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            llm = YandexClient(http, MODEL)
            with pytest.raises(kind) as error:
                await llm.complete([], [])
            assert "Yandex AI Studio" in str(error.value) and str(status) in str(error.value)
            assert "Groq" not in str(error.value) and "private prompt" not in str(error.value)
            if status == 429:
                with pytest.raises(QuotaPauseError):
                    await llm.check_access()
    asyncio.run(run())
    assert len(requests) == 1


@pytest.mark.parametrize("reason,stage,expected", [
    ("length", False, "1600 токенов; finish_reason=length"),
    ("length", True, "750 токенов; finish_reason=length"),
    ("content_filter", False, "finish_reason=content_filter"),
    ("synthetic-private-detail", False, "Причина не установлена"),
])
def test_incomplete_generation_has_a_specific_reason_without_echoing_output(reason, stage, expected):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"finish_reason": reason,
            "message": {"role": "assistant", "content": "synthetic-private-output"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            llm = YandexClient(http, MODEL)
            with pytest.raises(AgentError) as error:
                await llm.complete([], [], max_tokens=750 if stage else 1600,
                                   response_format={"type": "json_object"} if stage else None)
            text = str(error.value)
            assert expected in text and ("план запроса" if stage else "ответ") in text
            assert "synthetic-private" not in text
            assert llm.usage_totals["completion_tokens"] == 20
    asyncio.run(run())
    assert len(requests) == 1
