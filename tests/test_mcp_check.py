import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from mcp.types import CallToolResult, TextContent, Tool

import mcp_check as check


def schema(query="text_query", language="language_code"):
    return {"type": "object", "properties": {
        query: {"type": "string"}, language: {"type": "string"},
    }, "required": [query], "additionalProperties": False}


@pytest.mark.parametrize("query,language", [("text_query", "language_code"), ("textQuery", "languageCode")])
def test_live_schema_aliases(query, language):
    assert check.search_arguments(schema(query, language), "Москва") == {query: "Москва", language: "ru"}


def test_changed_required_fields_prevent_search():
    changed = schema()
    changed["required"].append("new_required_field")
    with pytest.raises(check.CheckError):
        check.search_arguments(changed, "Москва")


def test_missing_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    monkeypatch.setattr(check, "ENV_PATH", tmp_path / ".env")
    with pytest.raises(check.CheckError, match="Интеграция не проверена"):
        check.load_key()


def test_environment_has_priority_over_dotenv(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("GOOGLE_MAPS_API_KEY=local-value\n", encoding="utf-8")
    monkeypatch.setattr(check, "ENV_PATH", env_file)
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "environment-value")
    assert check.load_key() == "environment-value"


@pytest.mark.parametrize("status,expected", [(401, "отказ"), (403, "отказ"), (429, "квота"), (503, "503")])
def test_http_errors_do_not_echo_secrets(status, expected):
    error = httpx.HTTPStatusError(
        "secret-value", request=httpx.Request("POST", check.MCP_URL),
        response=httpx.Response(status),
    )
    message = check.error_message(ExceptionGroup("secret-value", [error]))
    assert expected in message
    assert "secret-value" not in message


@pytest.mark.parametrize("error,expected", [
    (TimeoutError("secret"), "таймаут"),
    (httpx.ConnectError("secret"), "сети"),
    (RuntimeError("secret"), "протокола"),
])
def test_transport_error_messages(error, expected):
    message = check.error_message(error)
    assert expected in message
    assert "secret" not in message


def fake_session(result=None):
    return SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(
            tools=[Tool(name="search_places", inputSchema=schema())], nextCursor=None,
        )),
        call_tool=AsyncMock(return_value=result or CallToolResult(
            content=[TextContent(type="text", text="Test response; not Google Maps data")],
        )),
    )


def test_schema_listing_does_not_search():
    session = fake_session()
    output = []
    asyncio.run(check.inspect_session(session, "Москва", True, output.append))
    session.call_tool.assert_not_awaited()
    assert '"inputSchema"' in "\n".join(output)


def test_one_search_preserves_response_and_sources():
    session = fake_session(CallToolResult(content=[], structuredContent={
        "summary": "Synthetic test fixture", "attribution": "Test attribution",
        "places": [{"googleMapsLinks": {"placeUrl": "https://example.com/test-source"}}],
    }))
    output = []
    asyncio.run(check.inspect_session(session, "Москва", False, output.append))
    session.call_tool.assert_awaited_once_with("search_places", arguments={"text_query": "Москва", "language_code": "ru"})
    assert "https://example.com/test-source" in "\n".join(output)
    assert "Test attribution" in "\n".join(output)


@pytest.mark.parametrize("result", [CallToolResult(content=[], isError=True), CallToolResult(content=[])])
def test_tool_error_or_empty_response_is_failure(result):
    with pytest.raises(check.CheckError):
        asyncio.run(check.inspect_session(fake_session(result), "Москва", False, lambda _: None))


def test_pagination_reads_next_page():
    session = fake_session()
    session.list_tools.side_effect = [
        SimpleNamespace(tools=[], nextCursor="next"),
        SimpleNamespace(tools=[Tool(name="search_places", inputSchema=schema())], nextCursor=None),
    ]
    asyncio.run(check.inspect_session(session, "Москва", False, lambda _: None))
    assert session.list_tools.await_args_list[1].kwargs == {"cursor": "next"}
    session.call_tool.assert_awaited_once()


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


@pytest.mark.parametrize("error,kind", [
    (httpx.ConnectError("synthetic-secret"), "ConnectError"),
    (httpx.ProxyError("synthetic-secret"), "ProxyError"),
    (httpx.ReadError("synthetic-secret"), "ReadError"),
    (httpx.RemoteProtocolError("synthetic-secret"), "RemoteProtocolError"),
    (httpx.ReadTimeout("synthetic-secret"), "ReadTimeout"),
])
def test_network_diagnostic_has_type_without_raw_exception(error, kind):
    message = check.error_message(error)
    assert kind in message
    assert "synthetic-secret" not in message


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
