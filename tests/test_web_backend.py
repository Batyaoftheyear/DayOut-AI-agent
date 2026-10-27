import asyncio
import json
import pickle
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from mcp.types import CallToolResult, Tool

import web_backend as web
from agent import AgentError, REVIEWED_MODEL

SOURCE = "https://www.google.com/maps/place/synthetic-web-fixture"
CONFIG = web.WebConfig("synthetic-groq-key", "synthetic-maps-key", REVIEWED_MODEL)


def install_network(monkeypatch, fail_once=False):
    """Actual agent/GroqClient, synthetic MCP and HTTP, no external connections."""
    requests, searches, clients, loops = [], [], [], []
    failed = False
    def handler(request):
        nonlocal failed
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": REVIEWED_MODEL}]})
        body = json.loads(request.content)
        requests.append(body)
        if body["messages"][-1]["content"] == "Technical connectivity check. Reply OK.":
            message = {"role": "assistant", "content": "OK"}
        elif body["messages"][-1]["role"] == "tool":
            if fail_once and not failed:
                failed = True
                raise httpx.ConnectError("synthetic-groq-key", request=request)
            message = {"role": "assistant", "content": "Синтетическая фикстура [S1]"}
        else:
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "web-call", "type": "function", "function": {
                    "name": "search_places", "arguments": '{"textQuery":"Синтетический поиск"}',
                },
            }]}
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": message}],
                                        "usage": {"prompt_tokens": 100, "completion_tokens": 10}})
    real_client = httpx.AsyncClient
    def client(**kwargs):
        value = real_client(**kwargs, transport=httpx.MockTransport(handler))
        clients.append(value)
        return value
    monkeypatch.setattr(web.httpx, "AsyncClient", client)
    @asynccontextmanager
    async def transport(*args, **kwargs):
        loops.append(asyncio.get_running_loop())
        yield None, None, None
    monkeypatch.setattr(web, "streamable_http_client", transport)
    class Session:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def initialize(self):
            pass
        async def list_tools(self, cursor=None):
            return SimpleNamespace(nextCursor=None, tools=[Tool(name="search_places", inputSchema={
                "type": "object", "properties": {"textQuery": {"type": "string"}}, "required": ["textQuery"],
            })])
        async def call_tool(self, name, arguments):
            searches.append(arguments)
            return CallToolResult(content=[], structuredContent={"summary": "Синтетические данные", "places": [{
                "attribution": {"title": "Синтетическая фикстура", "url": SOURCE},
            }]})
    monkeypatch.setattr(web, "ClientSession", Session)
    return requests, searches, clients, loops


def test_reruns_create_fresh_connections_keep_history_and_isolate_browsers(monkeypatch):
    requests, searches, clients, loops = install_network(monkeypatch)
    first, second, gate = web.WebMemory(), web.WebMemory(), web.RequestGate(cooldown=0)
    assert web.run_action(CONFIG, first, gate, "ask", "Бюджет 4000, Москва").reply
    assert web.run_action(CONFIG, first, gate, "ask", "Теперь в помещении").reply
    assert "Бюджет 4000" in json.dumps(requests[2], ensure_ascii=False)
    assert web.run_action(CONFIG, second, gate, "ask", "Независимый запрос").reply
    assert "Бюджет 4000" not in json.dumps(requests[-2], ensure_ascii=False)
    assert second.user_history == ["Независимый запрос"]
    assert first.user_history == ["Бюджет 4000, Москва", "Теперь в помещении"]
    assert len(searches) == 3
    assert len(set(loops)) == 3 and all(loop.is_closed() for loop in loops)
    assert all(client.is_closed for client in clients)
    pickle.dumps(first)  # Session memory contains no loops, clients or locks.


def test_web_check_and_retry_preserve_pending_mcp_without_repeating_search(monkeypatch):
    requests, searches, clients, _ = install_network(monkeypatch, fail_once=True)
    memory, gate = web.WebMemory(), web.RequestGate(cooldown=0)
    result = web.run_action(CONFIG, memory, gate, "ask", "Синтетический запрос")
    assert result.error and "synthetic-groq-key" not in result.error
    assert SOURCE in result.partial
    assert memory.paused and memory.pending and not memory.turns
    with pytest.raises(AgentError, match="Сначала проверьте"):
        web.run_action(CONFIG, memory, gate, "retry")
    assert web.run_action(CONFIG, memory, gate, "check").error is None
    assert not memory.paused
    reply = web.run_action(CONFIG, memory, gate, "retry").reply
    assert reply and SOURCE in reply.text
    assert len(searches) == 1 and len(requests) == 4
    assert memory.failed_query is None and memory.pending is None
    assert reply.usage == {"requests": 3, "prompt_tokens": 200, "completion_tokens": 20, "missing_usage": 1}
    assert all(client.is_closed for client in clients)


def test_gate_rejects_concurrent_or_fast_requests_and_releases_after_error(monkeypatch):
    gate = web.RequestGate()
    with gate.slot():
        with pytest.raises(web.BusyError, match="другой запрос"):
            with gate.slot():
                pass
    memory = web.WebMemory(failed_query="Сохранённый вопрос")
    with pytest.raises(web.BusyError, match="Подождите"):
        web.run_action(CONFIG, memory, gate, "ask", "Новый вопрос")
    assert memory.failed_query == "Сохранённый вопрос"
    assert not gate.lock.locked()
    gate.cooldown = 0
    with pytest.raises(RuntimeError):
        with gate.slot():
            raise RuntimeError("synthetic failure")
    assert not gate.lock.locked()


def test_reset_keeps_connection_pause_and_clears_only_this_browser():
    first = web.WebMemory(user_history=["Условие"], failed_query="Запрос", paused=True)
    second = web.WebMemory(user_history=["Другая история"])
    first.reset()
    assert first.paused and not first.user_history and first.failed_query is None
    assert second.user_history == ["Другая история"]


def test_cloud_secrets_override_dotenv_and_environment_has_priority(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text('GROQ_API_KEY=local\nGOOGLE_MAPS_API_KEY=local-maps\nGROQ_MODEL=' + REVIEWED_MODEL, encoding="utf-8")
    monkeypatch.setattr(web, "ENV_PATH", env)
    for name in web.CONFIG_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "environment")
    config = web.load_web_config({"GROQ_API_KEY": "cloud", "GOOGLE_MAPS_API_KEY": "cloud-maps"})
    assert config.groq_key == "environment" and config.maps_key == "cloud-maps"
    assert "environment" not in repr(config) and "cloud-maps" not in repr(config)
    assert config.redact("environment cloud-maps") == "[REDACTED] [REDACTED]"
    assert "GOOGLE_MAPS_API_KEY" not in __import__("os").environ


def test_missing_config_is_safe_and_does_not_attempt_network(monkeypatch, tmp_path):
    monkeypatch.setattr(web, "ENV_PATH", tmp_path / "missing.env")
    for name in web.CONFIG_NAMES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(AgentError, match="Не настроены"):
        web.load_web_config()


def test_progress_uses_plain_labels_instead_of_internal_tool_names():
    assert web.progress_label("MCP → search_places") == "Ищу подходящие места…"
    assert "лимита" in web.progress_label("Groq: ожидание обновления лимита, около 30 с")
