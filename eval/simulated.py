"""Offline fault injection. Scripted LLM outputs assess handling, not AI quality."""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from mcp.types import CallToolResult, Tool

from agent import REVIEWED_MODEL
import web_backend as web


def run_simulated(scenario):
    searches, posts, seen_empty = [], [], []
    injected = False
    source = "https://example.test/synthetic-eval-place"
    def handler(request):
        nonlocal injected
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": REVIEWED_MODEL}]})
        body = json.loads(request.content)
        posts.append(body)
        if body["messages"][-1]["content"] == "Technical connectivity check. Reply OK.":
            message = {"role": "assistant", "content": "OK"}
        elif body["messages"][-1]["role"] == "tool":
            if scenario == "empty_results":
                seen_empty.append(json.loads(body["messages"][-1]["content"])["places"] == [])
                message = {"role": "assistant", "content": "Поиск не вернул мест. Уточните район или условия."}
            elif not injected:
                injected = True
                if scenario == "connection_error":
                    raise httpx.ConnectError("synthetic-secret", request=request)
                return httpx.Response(403 if scenario == "access_denied" else 429, text="synthetic-secret")
            else:
                message = {"role": "assistant", "content": "Синтетический результат [S1]"}
        else:
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "eval-call", "type": "function", "function": {
                    "name": "search_places", "arguments": '{"textQuery":"Синтетический тест"}',
                },
            }]}
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": message}]})
    real_client = httpx.AsyncClient
    def client(**kwargs):
        return real_client(**kwargs, transport=httpx.MockTransport(handler))
    @asynccontextmanager
    async def transport(*args, **kwargs):
        yield None, None, None
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
            searches.append(name)
            return CallToolResult(content=[], structuredContent={"summary": "Синтетические данные", "places": []
                if scenario == "empty_results" else [{"attribution": {"title": "Synthetic fixture", "url": source}}]})
    config = web.WebConfig("synthetic-groq", "synthetic-maps", REVIEWED_MODEL)
    memory, gate = web.WebMemory(), web.RequestGate(cooldown=0)
    with patch.object(web.httpx, "AsyncClient", client), patch.object(web, "streamable_http_client", transport), patch.object(web, "ClientSession", Session):
        first = web.run_action(config, memory, gate, "ask", "Синтетический тест")
        if scenario == "empty_results":
            checks = {"empty_payload_forwarded": seen_empty == [True], "completed": first.reply is not None,
                      "no_source_invented": bool(first.reply and "http" not in first.reply.text)}
        else:
            checks = {"failure_visible": first.error is not None and first.reply is None,
                      "secret_hidden": "synthetic-secret" not in (first.error or ""),
                      "pending_saved": memory.pending is not None,
                      "partial_sources_available": bool(first.partial)}
            if scenario == "access_denied":
                checks["paused"] = memory.paused
            else:
                if scenario == "connection_error":
                    checks["paused"] = memory.paused
                    checks["manual_check"] = web.run_action(config, memory, gate, "check").error is None
                else:
                    checks["not_network_pause"] = not memory.paused
                retried = web.run_action(config, memory, gate, "retry")
                checks["manual_resume_completed"] = retried.reply is not None
                checks["mcp_not_repeated"] = len(searches) == 1
    return {"controlled_checks": checks, "execution_status": "completed" if all(checks.values()) else "failed",
            "llm": "scripted_mock", "external_requests": 0, "mcp_calls": len(searches),
            "scripted_model_posts": len(posts)}
