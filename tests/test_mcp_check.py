import asyncio
import json

import httpx
import pytest

import mcp_check as check


def schema(query="text_query", language="language_code"):
    return {"type": "object", "properties": {
        query: {"type": "string"}, language: {"type": "string"},
    }, "required": [query], "additionalProperties": False}


@pytest.mark.parametrize("query,language", [("text_query", "language_code"), ("textQuery", "languageCode")])
def test_live_schema_aliases(query, language):
    assert check.search_arguments(schema(query, language), "Москва") == {query: "Москва", language: "ru"}


def test_redaction_at_cli_boundary(monkeypatch, capsys):
    monkeypatch.setattr(check, "load_key", lambda: "secret-value")
    monkeypatch.setattr("sys.argv", ["mcp_check.py"])
    async def run(*args):
        args[-1]("secret-value")
        raise check.CheckError("server echoed secret-value")
    monkeypatch.setattr(check, "run_check", run)
    assert check.main() == 1
    output = capsys.readouterr()
    assert "secret-value" not in output.out + output.err
    assert "[REDACTED]" in output.out + output.err


def test_sdk_streamable_http_lifecycle_without_network(monkeypatch):
    calls = []
    client_class = httpx.AsyncClient

    def handler(request):
        assert str(request.url) == check.MCP_URL
        assert request.headers["X-Goog-Api-Key"] == "synthetic-test-key"
        body = json.loads(request.content)
        method = body["method"]
        calls.append(method)
        if "id" not in body:
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": body["params"]["protocolVersion"],
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "synthetic-test-server", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{"name": "search_places", "inputSchema": schema()}]}
        elif method == "tools/call":
            assert body["params"]["arguments"]["text_query"] == "Москва"
            result = {"content": [{"type": "text", "text": "Synthetic fixture only"}]}
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    def client_factory(**kwargs):
        return client_class(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(check.httpx, "AsyncClient", client_factory)
    output = []
    asyncio.run(check.run_check("synthetic-test-key", "Москва", False, 5, output.append))
    assert calls == ["initialize", "notifications/initialized", "tools/list", "tools/call"]
    assert "Synthetic fixture only" in "\n".join(output)
