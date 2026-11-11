import asyncio
import json
import pickle
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from mcp.types import CallToolResult, Tool

import web_backend as web
from agent import AgentError
from conversation_context import empty_context

SOURCE = "https://www.google.com/maps/place/synthetic-web-fixture"
CONFIG = web.WebConfig("synthetic-yandex-key", "synthetic-maps-key",
                       "gpt://synthetic-folder/aliceai-llm-flash/latest", "yandex", "synthetic-folder")


def install_network(monkeypatch, fail_once=False):
    """Real agent and Yandex client with mocked MCP and HTTP connections."""
    requests, searches, clients, loops = [], [], [], []
    failed = 0
    def handler(request):
        nonlocal failed
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": CONFIG.model}]})
        body = json.loads(request.content)
        requests.append(body)
        if body.get('response_format',{}).get('json_schema',{}).get('name')=='dayout_context':
            message={'role':'assistant','content':json.dumps(empty_context(),ensure_ascii=False)}
        elif body["messages"][-1]["content"] == "Technical connectivity check. Reply OK.":
            message = {"role": "assistant", "content": "OK"}
        elif body["messages"][-1]["role"] == "tool":
            # Exhaust the client's one automatic retry before testing manual recovery.
            if fail_once and failed < 2:
                failed += 1
                raise httpx.ConnectError("synthetic-yandex-key", request=request)
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
        kwargs['transport'] = httpx.MockTransport(handler)
        value = real_client(**kwargs)
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
    assert "Бюджет 4000" in json.dumps(requests[3], ensure_ascii=False)
    assert web.run_action(CONFIG, second, gate, "ask", "Независимый запрос").reply
    assert "Бюджет 4000" not in json.dumps(requests[-3], ensure_ascii=False)
    assert second.user_history == ["Независимый запрос"]
    assert first.user_history == ["Бюджет 4000, Москва", "Теперь в помещении"]
    assert len(searches) == 3
    assert len(set(loops)) == 3 and all(loop.is_closed() for loop in loops)
    assert all(client.is_closed for client in clients)
    pickle.dumps(first)  # Session memory contains no loops, clients or locks.


def test_a_connection_failure_before_generation_is_retried_once_without_duplicate_llm_cost(monkeypatch):
    requests, searches, _, loops = install_network(monkeypatch)
    original = web.ClientSession.initialize
    attempts = []
    async def initialize(self):
        attempts.append(True)
        if len(attempts) == 1:
            raise httpx.ConnectError("synthetic-private-connection-detail")
        await original(self)
    monkeypatch.setattr(web.ClientSession, "initialize", initialize)
    monkeypatch.setattr(web, "sleep", lambda _: None)
    result = web.run_action(CONFIG, web.WebMemory(), web.RequestGate(cooldown=0), "ask", "Найди кафе")
    assert result.reply and len(attempts) == 2 and len(requests) == 3 and len(searches) == 1
    assert len(loops) == 2 and "повторяю подключение" in result.events[0]
    assert "synthetic-private" not in " ".join(result.events)


def test_web_check_and_retry_preserve_pending_mcp_without_repeating_search(monkeypatch):
    requests, searches, clients, _ = install_network(monkeypatch, fail_once=True)
    memory, gate = web.WebMemory(), web.RequestGate(cooldown=0)
    result = web.run_action(CONFIG, memory, gate, "ask", "Синтетический запрос")
    assert result.error and "synthetic-yandex-key" not in result.error
    assert SOURCE in result.partial
    assert memory.paused and memory.pending and not memory.turns
    with pytest.raises(AgentError, match="Сначала проверьте"):
        web.run_action(CONFIG, memory, gate, "retry")
    assert web.run_action(CONFIG, memory, gate, "check").error is None
    assert not memory.paused
    reply = web.run_action(CONFIG, memory, gate, "retry").reply
    assert reply and SOURCE in reply.text
    assert len(searches) == 1 and len(requests) == 6
    assert memory.failed_query is None and memory.pending is None
    assert reply.usage == {"requests": 5, "prompt_tokens": 300, "completion_tokens": 30, "missing_usage": 2}
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
    env.write_text('LLM_PROVIDER=yandex\nYANDEX_API_KEY=local\nYANDEX_FOLDER_ID=synthetic-folder\n'
                   'YANDEX_MODEL=aliceai-llm-flash\nGOOGLE_MAPS_API_KEY=local-maps', encoding="utf-8")
    monkeypatch.setattr(web, "ENV_PATH", env)
    for name in web.CONFIG_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("YANDEX_API_KEY", "environment")
    config = web.load_web_config({"YANDEX_API_KEY": "cloud", "GOOGLE_MAPS_API_KEY": "cloud-maps"})
    assert config.llm_key == "environment" and config.maps_key == "cloud-maps"
    assert "environment" not in repr(config) and "cloud-maps" not in repr(config)
    assert config.redact("environment cloud-maps") == "[REDACTED] [REDACTED]"
    assert "GOOGLE_MAPS_API_KEY" not in __import__("os").environ


def test_missing_config_is_safe_and_does_not_attempt_network(monkeypatch, tmp_path):
    monkeypatch.setattr(web, "ENV_PATH", tmp_path / "missing.env")
    for name in web.CONFIG_NAMES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(AgentError, match="Не настроены"):
        web.load_web_config()


def test_yandex_web_round_trip_searches_and_saves_context(monkeypatch):
    requests, searches, _, _ = install_network(monkeypatch)
    config = CONFIG
    gate = web.RequestGate(cooldown=0)
    result = web.run_action(config, web.WebMemory(), gate, "ask", "Найди кафе")
    assert result.reply and SOURCE in result.reply.text
    assert len(requests) == 3 and len(searches) == 1
    assert all(body["model"] == config.model and "max_tokens" in body
               and "reasoning_effort" not in body for body in requests)
    assert all({t['function']['name'] for t in body['tools']}=={'search_places'} for body in requests if body.get('tools'))
    assert requests[-1]['response_format']['json_schema']['name']=='dayout_context' and 'tools' not in requests[-1]
