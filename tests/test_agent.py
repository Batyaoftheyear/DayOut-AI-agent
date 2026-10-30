import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from mcp.types import CallToolResult, TextContent, Tool

from agent import (AgentError, AnswerValidationError, DayOutAgent, GroqClient, GroqConnectionError,
                   QuotaPauseError, collect_sources, compact_schema, discover_tools, extract_payload,
                   tool_evidence, validate_behavior)

SOURCE = "https://www.google.com/maps/place/test-fixture"
SCHEMA = {"type": "object", "properties": {"textQuery": {"type": "string"}},
          "required": ["textQuery"], "additionalProperties": False}


def tool_call(name="search_places", arguments=None, call_id="call-1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(
        {"textQuery": "Искусственный тест в Москве"} if arguments is None else arguments,
    )}}


def calling(*calls):
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def answer(text="Вымышленное место. Google Maps: " + SOURCE):
    return {"role": "assistant", "content": text}


def fixture_agent(responses, result=None, **kwargs):
    session = SimpleNamespace(call_tool=AsyncMock(return_value=result or CallToolResult(
        content=[], structuredContent={"summary": "Синтетическая фикстура", "places": [{
            "id": "fixture-id", "attribution": {"title": "Тест — Google Maps", "url": SOURCE},
            "googleMapsLinks": {"placeUrl": SOURCE},
        }]},
    )))
    llm = SimpleNamespace(complete=AsyncMock(side_effect=responses))
    tools = {"search_places": Tool(name="search_places", inputSchema=SCHEMA)}
    return DayOutAgent(session, llm, tools, **kwargs)


def test_followup_keeps_evidence_and_constraints_without_old_tool_protocol():
    snapshots = []
    responses = iter([calling(tool_call()), answer(), answer("Сохраняю условия; источник: " + SOURCE)])
    async def complete(messages, tools, allow_tools):
        snapshots.append(copy.deepcopy(messages))
        return next(responses)
    agent = fixture_agent([])
    agent.llm.complete = complete
    async def run():
        first = await agent.ask("На двоих в Москве, бюджет 4000")
        second = await agent.ask("Теперь сократи длительность")
        return first, second
    first, second = asyncio.run(run())
    assert first.tools == ["search_places"]
    assert second.tools == []
    assert not any(message.get("tool_call_id") for message in snapshots[-1])
    assert "fixture-id" in json.dumps(snapshots[-1])
    assert SOURCE not in json.dumps(snapshots[-1])
    assert "[S1]" in json.dumps(snapshots[-1])
    assert any("бюджет 4000" in (message.get("content") or "") for message in snapshots[-1])
    assert any(message["role"] == "tool" for message in snapshots[1])


@pytest.mark.parametrize("call", [tool_call(name="delete_file"), tool_call(arguments={"textQuery": 123})])
def test_unknown_tool_and_invalid_arguments_never_reach_mcp(call):
    agent = fixture_agent([calling(call), answer("Не удалось проверить данные.")])
    asyncio.run(agent.ask("Тест"))
    agent.session.call_tool.assert_not_awaited()


def test_failed_turn_is_rolled_back():
    agent = fixture_agent([answer("Уточните район."), calling(tool_call())])
    async def run():
        await agent.ask("Москва")
        saved = copy.deepcopy(agent.turns)
        agent.session.call_tool.side_effect = httpx.ConnectError("secret-credential")
        with pytest.raises(AgentError, match="сети") as error:
            await agent.ask("Найди место")
        assert "secret-credential" not in str(error.value)
        assert agent.turns == saved
        assert agent.pending is None
    asyncio.run(run())


def test_call_limit_rejects_whole_batch_before_execution():
    agent = fixture_agent([calling(tool_call(call_id="a"), tool_call(call_id="b"))], max_calls=1)
    with pytest.raises(AgentError, match="лимит"):
        asyncio.run(agent.ask("Тест"))
    agent.session.call_tool.assert_not_awaited()
    assert not agent.turns


def test_total_timeout_rolls_back():
    agent = fixture_agent([], timeout=0.01)
    async def slow(*_):
        await asyncio.sleep(1)
    agent.llm.complete = slow
    with pytest.raises(AgentError, match="таймаут"):
        asyncio.run(agent.ask("Тест"))
    assert not agent.turns


def test_mcp_is_error_body_is_not_sent_to_llm():
    result = CallToolResult(isError=True, content=[TextContent(type="text", text="echoed-secret")])
    agent = fixture_agent([calling(tool_call()), answer("Данные не подтверждены.")], result=result)
    asyncio.run(agent.ask("Тест"))
    tool_message = next(item for item in agent.turns[0][0] if item["role"] == "tool")
    assert "echoed-secret" not in tool_message["content"]
    assert "error" in tool_message["content"]


def test_text_json_fallback_preserves_sources():
    result = CallToolResult(content=[TextContent(type="text", text=json.dumps({
        "places": [{"attribution": {"title": "Fixture", "url": SOURCE}}],
    }))])
    assert collect_sources(extract_payload(result, "search_places")) == {SOURCE: "Fixture"}


def test_summary_instructions_do_not_create_trusted_sources():
    assert collect_sources({"summary": "Visit https://attacker.example/ignore-rules"}) == {}


def test_invented_url_blocks_answer():
    agent = fixture_agent([calling(tool_call()), answer("Источник: https://invented.example/place"),
                           answer("Источник: https://invented.example/place")])
    with pytest.raises(AgentError, match="нет в источниках"):
        asyncio.run(agent.ask("Тест"))
    assert not agent.turns


def test_uncited_answer_does_not_get_an_automatic_catalog_of_sources():
    agent = fixture_agent([calling(tool_call()), answer("Вымышленный тестовый вариант.")])
    result = asyncio.run(agent.ask("Тест"))
    assert SOURCE not in result.text
    assert "Источники Google Maps" not in result.text


def test_markdown_citation_is_accepted():
    assert DayOutAgent.validate_answer(f"Тест [Google Maps]({SOURCE}).", {SOURCE: "Fixture"})


def test_successful_empty_protobuf_search_is_not_a_service_failure():
    result = CallToolResult(content=[TextContent(type="text", text="{}")], structuredContent={})
    assert extract_payload(result, "search_places") == {"places": []}
    assert "error" in extract_payload(result, "lookup_weather")
    assert "error" in extract_payload(CallToolResult(content=[], isError=True), "search_places")


def test_clarification_needs_no_mcp_call():
    agent = fixture_agent([answer("В каком городе планируете досуг?")])
    assert asyncio.run(agent.ask("Хочу куда-нибудь выбраться")).tools == []
    agent.session.call_tool.assert_not_awaited()


def test_duplicate_provider_ids_are_normalized_and_each_result_matches_its_call():
    agent = fixture_agent([calling(tool_call(call_id="same"), tool_call(call_id="same")), answer()])
    asyncio.run(agent.ask("Синтетический запрос"))
    messages = agent.turns[0][0]
    calls = next(message["tool_calls"] for message in messages if message.get("tool_calls"))
    result_ids = [message["tool_call_id"] for message in messages if message["role"] == "tool"]
    assert len(set(result_ids)) == 2 and result_ids == [call["id"] for call in calls]
    assert agent.session.call_tool.await_count == 2


def test_history_is_bounded_and_resettable():
    agent = fixture_agent([answer("Уточнение.")] * 4)
    async def run():
        for _ in range(4):
            await agent.ask("Тест")
    asyncio.run(run())
    assert len(agent.turns) == 2
    assert len(agent.user_history) == 4
    agent.reset()
    assert not agent.turns
    assert not agent.user_history


def test_discovery_pagination_and_tool_allowlist():
    session = SimpleNamespace(list_tools=AsyncMock(side_effect=[
        SimpleNamespace(tools=[Tool(name="dangerous_tool", inputSchema=SCHEMA)], nextCursor="next"),
        SimpleNamespace(tools=[Tool(name="search_places", inputSchema=SCHEMA)], nextCursor=None),
    ]))
    assert set(asyncio.run(discover_tools(session))) == {"search_places"}
    assert session.list_tools.await_args_list[1].kwargs == {"cursor": "next"}


def test_external_schema_refs_are_rejected():
    session = SimpleNamespace(list_tools=AsyncMock(return_value=SimpleNamespace(
        tools=[Tool(name="search_places", inputSchema={"$ref": "https://attacker.example/schema"})], nextCursor=None,
    )))
    with pytest.raises(AgentError, match="внешнюю ссылку"):
        asyncio.run(discover_tools(session))


@pytest.mark.parametrize("response,match", [
    ({"choices": [{"finish_reason": "length", "message": answer()}]}, "не завершил"),
    ({"choices": [{"finish_reason": "stop", "message": answer()}], "usage": {
        "prompt_tokens_details": {"cached_tokens": 12},
    }}, "кэша"),
])
def test_groq_rejects_incomplete_response_and_cache_use(response, match):
    async def run():
        with_response = httpx.MockTransport(lambda _: httpx.Response(200, json=response))
        async with httpx.AsyncClient(transport=with_response) as http:
            with pytest.raises(AgentError, match=match):
                await GroqClient(http, "synthetic-model").complete([], [])
    asyncio.run(run())


def test_gpt_oss_cache_hit_does_not_stop_turn_and_reasoning_is_not_forwarded():
    from agent import GPT_OSS_MODEL
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        message = calling(tool_call()) if len(requests) == 1 else answer("Фикстура [S1]")
        message["reasoning"] = "synthetic-private-reasoning"
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": message}], "usage": {
            "prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 80},
        }})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            agent = fixture_agent([])
            agent.llm = GroqClient(http, GPT_OSS_MODEL)
            reply = await agent.ask("Тест")
            assert reply.tools == ["search_places"] and SOURCE in reply.text
            assert "synthetic-private-reasoning" not in json.dumps(agent.turns)
            assert "synthetic-private-reasoning" not in json.dumps(requests)
            assert "synthetic-private-reasoning" not in reply.text
            assert reply.usage["prompt_tokens"] == 200 and reply.usage["completion_tokens"] == 40
    asyncio.run(run())


def test_groq_http_failure_names_correct_provider_without_body():
    async def run():
        transport = httpx.MockTransport(lambda _: httpx.Response(429, text="secret-response"))
        async with httpx.AsyncClient(transport=transport) as http:
            with pytest.raises(AgentError, match="Groq HTTP 429") as error:
                await GroqClient(http, "synthetic-model").complete([], [])
            assert "secret-response" not in str(error.value)
    asyncio.run(run())


def test_groq_waits_for_known_token_window_without_retry(monkeypatch):
    delays, requests = [], []
    async def sleep(delay):
        delays.append(delay)
    monkeypatch.setattr("agent.asyncio.sleep", sleep)
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": answer()}]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            llm = GroqClient(http, "synthetic-model", max_quota_wait=65)
            llm.update_limits({"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "1m0.5s"})
            await llm.complete([], [])
    asyncio.run(run())
    assert len(delays) == 1 and 60 < delays[0] < 62
    assert len(requests) == 1


def test_manual_check_does_not_send_chat_or_maps_content():
    requests = []
    def handler(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "synthetic-model"}]})
        body = json.loads(request.content)
        assert "tools" not in body
        assert body["max_completion_tokens"] == 64
        assert body["messages"] == [{"role": "user", "content": "Technical connectivity check. Reply OK."}]
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": answer("OK")}]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            await GroqClient(http, "synthetic-model").check_access()
    asyncio.run(run())
    assert len(requests) == 2


def test_actual_usage_counts_input_and_output_across_llm_calls():
    sent = []
    def handler(request):
        sent.append(json.loads(request.content))
        if len(sent) == 1:
            message, usage = calling(tool_call()), {"prompt_tokens": 100, "completion_tokens": 20}
        else:
            message, usage = answer("Тестовое место. [S1]"), {"prompt_tokens": 140, "completion_tokens": 30}
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "tool_calls" if len(sent) == 1 else "stop", "message": message,
        }], "usage": usage})
    async def run():
        output = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            agent = fixture_agent([])
            agent.llm = GroqClient(http, "synthetic-model", output.append, show_usage=True)
            reply = await agent.ask("Тест")
            assert reply.usage == {"requests": 2, "prompt_tokens": 240, "completion_tokens": 50, "missing_usage": 0}
            assert any("вход 100, выход 20" in item for item in output)
            assert SOURCE in reply.text and "[S1]" not in reply.text
    asyncio.run(run())
    assert SOURCE not in json.dumps(sent[-1])
    assert sent[-1]["messages"][-1]["role"] == "tool"


def test_missing_usage_is_not_reported_as_measured_zero():
    async def run():
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": answer("Уточните город.")}],
        }))
        async with httpx.AsyncClient(transport=transport) as http:
            agent = fixture_agent([])
            agent.llm = GroqClient(http, "synthetic-model")
            reply = await agent.ask("Тест")
            assert reply.usage["requests"] == reply.usage["missing_usage"] == 1
    asyncio.run(run())


def test_user_conditions_survive_more_than_two_turns():
    sent = []
    async def complete(messages, *_):
        sent.append(copy.deepcopy(messages))
        return answer("Уточнение.")
    agent = fixture_agent([])
    agent.llm.complete = complete
    async def run():
        for text in ["Москва, на двоих, бюджет 4000", "Без улицы", "В центре", "Теперь музеи"]:
            await agent.ask(text)
    asyncio.run(run())
    history = json.dumps(sent[-1], ensure_ascii=False)
    assert "бюджет 4000" in history and "Без улицы" in history and "В центре" in history


def test_history_limit_rejects_new_turn_without_silent_condition_loss():
    agent = fixture_agent([answer("Уточнение.")] * 4)
    async def run():
        for _ in range(3):
            await agent.ask("а" * 4000)
        saved = copy.deepcopy(agent.user_history)
        with pytest.raises(AgentError, match="История условий заполнена"):
            await agent.ask("Новый запрос")
        assert agent.user_history == saved
    asyncio.run(run())


def test_sources_aliases_are_resolved_before_mcp_arguments_are_sent():
    calls = [calling(tool_call()), calling(tool_call("resolve_maps_urls", {"urls": ["[S1]"]}, "call-2")),
             answer("Ссылка: [S1]")]
    agent = fixture_agent(calls)
    agent.tools["resolve_maps_urls"] = Tool(name="resolve_maps_urls", inputSchema={
        "type": "object", "properties": {"urls": {"type": "array", "items": {"type": "string"}}},
        "required": ["urls"],
    })
    result = asyncio.run(agent.ask("Тест"))
    assert agent.session.call_tool.await_args_list[-1].kwargs == {"arguments": {"urls": [SOURCE]}}
    assert SOURCE in result.text


def test_unknown_source_placeholder_blocks_answer():
    agent = fixture_agent([calling(tool_call()), answer("Источник [S999]"), answer("Источник [S999]")])
    with pytest.raises(AgentError, match="неизвестную метку"):
        asyncio.run(agent.ask("Тест"))
    assert agent.llm.complete.await_count == 3
    assert not agent.turns


def test_fabricated_source_before_search_gets_one_repair_and_real_evidence():
    output = []
    agent = fixture_agent([answer("UNTRUSTED-FABRICATION [S999]"), calling(tool_call()),
                           answer("Подтверждённая фикстура [S1]")], emit=output.append)
    reply = asyncio.run(agent.ask("Тест"))
    assert reply.tools == ["search_places"]
    assert SOURCE in reply.text
    agent.session.call_tool.assert_awaited_once()
    first, repaired = agent.llm.complete.await_args_list[:2]
    assert "Доступные метки источников: нет." in first.args[0][0]["content"]
    assert "вызови search_places" in repaired.args[0][-1]["content"]
    assert "UNTRUSTED-FABRICATION" not in json.dumps(repaired.args[0])
    assert "UNTRUSTED-FABRICATION" not in json.dumps(agent.turns)
    assert len([text for text in output if "одна попытка" in text]) == 1


def test_citation_repair_uses_existing_data_without_repeating_search():
    agent = fixture_agent([calling(tool_call()), answer("Источник [S999]"), answer("Источник [S1]")])
    reply = asyncio.run(agent.ask("Тест"))
    assert SOURCE in reply.text
    agent.session.call_tool.assert_awaited_once()
    repaired = agent.llm.complete.await_args_list[-1].args[0]
    assert "Доступные метки источников: [S1]." in repaired[0]["content"]
    assert any(item["role"] == "tool" for item in repaired)


def test_local_summary_indices_are_bound_to_each_response_metadata_without_renumbering():
    second = SOURCE + "-second"
    payload = {"summary": "Первое [0], второе [2], вместе [0, 2], неизвестное [1].", "places": [
        {"googleMapsLinks": {"placeUrl": SOURCE}}, None,
        {"googleMapsLinks": {"placeUrl": second}},
    ]}
    result = extract_payload(CallToolResult(content=[], structuredContent=payload), "search_places")
    assert result["summary"] == f"Первое {SOURCE}, второе {second}, вместе {SOURCE} {second}, неизвестное [1]."
    assert payload["summary"].startswith("Первое [0]")  # Original provider response was not mutated.
    other = extract_payload(CallToolResult(content=[], structuredContent={
        "summary": "Другой поиск [0].", "places": [{"googleMapsLinks": {"placeUrl": second}}],
    }), "search_places")
    assert other["summary"] == f"Другой поиск {second}."


def test_numeric_citation_gets_one_repair_without_repeating_mcp_search():
    agent = fixture_agent([calling(tool_call()), answer("Фикстура [0]"), answer("Фикстура [S1]")])
    reply = asyncio.run(agent.ask("Тест"))
    assert SOURCE in reply.text and "[0]" not in reply.text
    agent.session.call_tool.assert_awaited_once()
    assert agent.llm.complete.await_count == 3


@pytest.mark.parametrize("explicit,expected", [(None, 3), (5, 5)])
def test_small_default_page_size_does_not_override_explicit_count(explicit, expected):
    args = {"textQuery": "Тест"}
    if explicit is not None:
        args["pageSize"] = explicit
    agent = fixture_agent([calling(tool_call(arguments=args)), answer()])
    schema = copy.deepcopy(SCHEMA)
    schema["properties"]["pageSize"] = {"type": "integer"}
    agent.tools["search_places"] = Tool(name="search_places", inputSchema=schema)
    asyncio.run(agent.ask("Тест"))
    assert agent.session.call_tool.await_args.kwargs["arguments"]["pageSize"] == expected


def test_schema_compaction_preserves_property_names_and_literal_values():
    schema = {"type": "object", "description": "Annotation to omit", "properties": {
        "description": {"type": "string", "description": "Field annotation"},
        "title": {"const": {"description": "Literal value must remain"}},
    }, "required": ["description", "title"]}
    compact = compact_schema(schema)
    assert "description" not in compact
    assert compact["properties"]["description"] == {"type": "string"}
    assert compact["properties"]["title"]["const"] == {"description": "Literal value must remain"}
    assert compact["required"] == schema["required"]
    assert schema["properties"]["description"]["description"] == "Field annotation"


def test_multiple_searches_keep_source_aliases_matched_to_returned_places():
    other = SOURCE + "-second"
    agent = fixture_agent([calling(tool_call(call_id="one")), calling(tool_call(call_id="two")),
                           answer("Второе [S2], первое [S1].")])
    def result(url):
        return CallToolResult(content=[], structuredContent={"places": [
            {"attribution": {"title": "Synthetic fixture", "url": url}},
        ]})
    agent.session.call_tool.side_effect = [result(SOURCE), result(other)]
    reply = asyncio.run(agent.ask("Тест"))
    assert f"Второе [Google Maps]({other}), первое [Google Maps]({SOURCE})" in reply.text


def test_manual_resume_keeps_mcp_protocol_and_usage_but_excludes_connectivity_check():
    sent = []
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "synthetic-model"}]})
        body = json.loads(request.content)
        sent.append(body)
        if len(sent) == 1:
            message, usage = calling(tool_call(call_id="one"), tool_call(call_id="two")), {
                "prompt_tokens": 100, "completion_tokens": 20,
            }
        elif len(sent) == 2:
            raise httpx.ConnectError("synthetic-secret", request=request)
        elif len(sent) == 3:  # Manual /check: fixed text, no conversation.
            assert "tools" not in body
            message, usage = answer("OK"), {"prompt_tokens": 5, "completion_tokens": 1}
        else:
            assert [item["tool_call_id"] for item in body["messages"] if item["role"] == "tool"] == ["one", "two"]
            assert body["tool_choice"] == "none"  # MCP budget survived the failure.
            assert SOURCE not in json.dumps(body)
            message, usage = answer("Фикстура [S1]"), {"prompt_tokens": 140, "completion_tokens": 30}
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": message}], "usage": usage})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            agent = fixture_agent([], max_calls=2)
            agent.llm = GroqClient(http, "synthetic-model")
            with pytest.raises(GroqConnectionError):
                await agent.ask("Тест")
            assert len(sent) == 2  # No automatic network retry.
            assert agent.pending is not None
            assert not agent.turns and not agent.user_history
            assert agent.session.call_tool.await_count == 2
            await agent.llm.check_access()
            reply = await agent.ask("Тест", resume=True)
            assert reply.usage == {"requests": 3, "prompt_tokens": 240, "completion_tokens": 50, "missing_usage": 1}
            assert reply.tools == ["search_places", "search_places"]
            assert SOURCE in reply.text
            assert agent.pending is None
            assert agent.user_history == ["Тест"]
            assert len(agent.turns) == 1
            assert agent.session.call_tool.await_count == 2
    asyncio.run(run())
    assert len(sent) == 4


@pytest.mark.parametrize("new_query", ["Тест", "Другой вопрос"])
def test_new_question_discards_pending_state_even_if_text_matches(new_query):
    agent = fixture_agent([calling(tool_call()), GroqConnectionError("ConnectError"),
                           answer("Уточните город.")])
    async def run():
        with pytest.raises(GroqConnectionError):
            await agent.ask("Тест")
        await agent.ask(new_query)
        assert agent.pending is None
        assert not any(item["role"] == "tool" for item in agent.llm.complete.await_args.args[0])
        assert agent.user_history == [new_query]
    asyncio.run(run())


def test_reset_discards_pending_results():
    agent = fixture_agent([calling(tool_call()), GroqConnectionError("ConnectError")])
    async def run():
        with pytest.raises(GroqConnectionError):
            await agent.ask("Тест")
        assert agent.pending is not None
        agent.reset()
        assert agent.pending is None
    asyncio.run(run())


def test_expired_pending_results_are_not_reused(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("agent.monotonic", lambda: now[0])
    output = []
    agent = fixture_agent([calling(tool_call()), GroqConnectionError("ConnectError"),
                           answer("Уточните город.")], emit=output.append)
    async def run():
        with pytest.raises(GroqConnectionError):
            await agent.ask("Тест")
        now[0] += 301
        await agent.ask("Тест", resume=True)
        assert not any(item["role"] == "tool" for item in agent.llm.complete.await_args.args[0])
        assert agent.pending is None
    asyncio.run(run())
    assert any("устарел" in text for text in output)


def test_resume_does_not_reset_source_repair_limit():
    agent = fixture_agent([answer("Источник [S999]"), GroqConnectionError("ConnectError"),
                           answer("Источник [S999]")])
    async def run():
        with pytest.raises(GroqConnectionError):
            await agent.ask("Тест")
        with pytest.raises(AgentError, match="неизвестную метку"):
            await agent.ask("Тест", resume=True)
        assert agent.pending is None
        assert not agent.turns
        assert agent.llm.complete.await_count == 3
        agent.session.call_tool.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("text", [
    "Проверяю часы работы.", "**Что я делаю сейчас:**\n1. Проверяю кафе.",
    "Сейчас найду кафе.", "Кафе закрыто к 22:00.", "Парк открыт круглосуточно.",
    "Часы работы: 10:00–23:00.", "Заведение работает до 23:00, но лучше проверьте.",
    "Дворец в 15 минутах от района.", "Пешком идти около 15 минут.",
])
def test_observable_unsupported_claims_are_rejected(text):
    with pytest.raises(AnswerValidationError):
        validate_behavior(text, [])


@pytest.mark.parametrize("text", [
    "Часы работы неизвестны; доступ в 22:00 не подтверждён.",
    "Открытая веранда, время работы неизвестно.", "Открытое пространство.",
    "Проверьте часы работы на 22:00 перед визитом.", "Время пути неизвестно.",
    "Если укажете город, смогу подобрать варианты.",
])
def test_unknowns_and_clarifications_are_allowed(text):
    validate_behavior(text, [])


def test_hours_and_route_evidence_can_support_corresponding_claims():
    validate_behavior("Кафе открыто до 23:00.", [("search_places", {"summary": "Кафе работает с 10:00 до 23:00."})])
    validate_behavior("Пешком идти 15 минут.", [("compute_routes", {"routes": [{"duration": "900s"}]})])


def test_unknown_hours_in_search_are_not_treated_as_confirmation():
    with pytest.raises(AnswerValidationError, match="Часы работы"):
        validate_behavior("Кафе открыто до 23:00.", [("search_places", {"summary": "Часы работы не указаны."})])


def test_previous_assistant_assertions_and_error_results_are_not_evidence():
    messages = [answer("Кафе работает до 23:00."), calling(tool_call("compute_routes")),
                {"role": "tool", "tool_call_id": "call-1", "content": '{"error":"unavailable"}'}]
    evidence = list(tool_evidence(messages))
    assert evidence == []
    with pytest.raises(AnswerValidationError):
        validate_behavior("Кафе работает до 23:00.", evidence)


def test_promise_is_repaired_inside_turn_and_not_committed():
    agent = fixture_agent([answer("Проверяю подходящие кафе."), calling(tool_call()), answer("Фикстура [S1]")])
    reply = asyncio.run(agent.ask("Тест"))
    assert SOURCE in reply.text
    assert "Проверяю подходящие" not in json.dumps(agent.turns, ensure_ascii=False)
    agent.session.call_tool.assert_awaited_once()


@pytest.mark.parametrize("field,explicit,expected", [
    ("languageCode", None, "ru"), ("language_code", None, "ru"), ("languageCode", "en", "en"),
])
def test_search_language_defaults_to_russian_only_when_schema_supports_it(field, explicit, expected):
    args = {"textQuery": "Синтетический тест"}
    if explicit is not None:
        args[field] = explicit
    agent = fixture_agent([calling(tool_call(arguments=args)), answer()])
    schema = copy.deepcopy(SCHEMA)
    schema["properties"][field] = {"type": "string"}
    agent.tools["search_places"] = Tool(name="search_places", inputSchema=schema)
    asyncio.run(agent.ask("Тест"))
    assert agent.session.call_tool.await_args.kwargs["arguments"][field] == expected


def test_search_keeps_place_citation_without_unused_direction_link():
    result = CallToolResult(content=[], structuredContent={"summary": "Синтетическая фикстура", "places": [{
        "attribution": {"title": "Fixture", "url": SOURCE},
        "googleMapsLinks": {"placeUrl": SOURCE, "directionsUrl": "https://example.test/dir"},
    }]})
    search = extract_payload(result, "search_places")
    assert collect_sources(search) == {SOURCE: "Fixture"}
    assert collect_sources(extract_payload(result, "compute_routes"))["https://example.test/dir"]


def test_timeout_at_llm_boundary_preserves_results_for_retry():
    output, calls = [], []
    async def complete(*args):
        calls.append(args)
        if len(calls) == 1:
            return calling(tool_call())
        if len(calls) == 2:
            await asyncio.sleep(1)
        return answer("Фикстура [S1]")
    agent = fixture_agent([], timeout=.01, emit=output.append)
    agent.llm.complete = complete
    async def run():
        with pytest.raises(AgentError, match="таймаут.*сохранён"):
            await agent.ask("Тест")
        assert agent.pending is not None and not agent.turns
        assert any("Поиск вернул кандидатов" in text and SOURCE in text for text in output)
        agent.timeout = 1
        reply = await agent.ask("Тест", resume=True)
        assert SOURCE in reply.text
        agent.session.call_tool.assert_awaited_once()
        assert agent.pending is None
    asyncio.run(run())


def test_timeout_during_mcp_does_not_keep_incomplete_tool_protocol():
    agent = fixture_agent([calling(tool_call())], timeout=.01)
    async def slow(*args, **kwargs):
        await asyncio.sleep(1)
    agent.session.call_tool.side_effect = slow
    with pytest.raises(AgentError, match="таймаут"):
        asyncio.run(agent.ask("Тест"))
    assert agent.pending is None and not agent.turns


def test_quota_wait_budget_is_cumulative_and_does_not_send_paused_request(monkeypatch):
    delays, requests = [], []
    monkeypatch.setattr("agent.monotonic", lambda: 1000.0)
    async def sleep(delay):
        delays.append(delay)
    monkeypatch.setattr("agent.asyncio.sleep", sleep)
    def handler(request):
        requests.append(request)
        return httpx.Response(200, headers={"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "20s"},
                              json={"choices": [{"finish_reason": "stop", "message": answer()}]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            llm = GroqClient(http, "synthetic-model")
            llm.update_limits({"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "20s"})
            await llm.complete([], [])
            await llm.complete([], [])
            with pytest.raises(QuotaPauseError):
                await llm.complete([], [])
            assert llm.usage_totals["requests"] == 2
            llm.start_attempt()
            await llm.complete([], [])
    asyncio.run(run())
    assert delays == [21, 21, 21] and len(requests) == 3


def test_long_quota_wait_preserves_mcp_and_counts_only_sent_requests():
    requests = []
    def handler(request):
        requests.append(request)
        message = calling(tool_call()) if len(requests) == 1 else answer("Фикстура [S1]")
        return httpx.Response(200, headers={"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "60s"},
                              json={"choices": [{"finish_reason": "stop", "message": message}],
                                    "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            agent = fixture_agent([])
            agent.llm = GroqClient(http, "synthetic-model")
            with pytest.raises(QuotaPauseError):
                await agent.ask("Тест")
            assert len(requests) == 1 and agent.pending is not None
            agent.llm.reset_at = 0  # The window has elapsed before manual retry.
            reply = await agent.ask("Тест", resume=True)
            assert reply.usage == {"requests": 2, "prompt_tokens": 200, "completion_tokens": 40, "missing_usage": 0}
            agent.session.call_tool.assert_awaited_once()
    asyncio.run(run())


def test_http_429_keeps_completed_mcp_results_for_manual_retry():
    requests = []
    def handler(request):
        requests.append(request)
        if len(requests) == 2:
            return httpx.Response(429, text="synthetic-secret")
        message = calling(tool_call()) if len(requests) == 1 else answer("Фикстура [S1]")
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": message}]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            agent = fixture_agent([])
            agent.llm = GroqClient(http, "synthetic-model")
            with pytest.raises(QuotaPauseError, match="429") as error:
                await agent.ask("Тест")
            assert "synthetic-secret" not in str(error.value)
            assert agent.pending is not None
            await agent.ask("Тест", resume=True)
            agent.session.call_tool.assert_awaited_once()
            assert len(requests) == 3
    asyncio.run(run())


def test_retry_after_blocks_completions_and_connectivity_checks_until_deadline(monkeypatch):
    clock, requests = [1000.0], []
    monkeypatch.setattr("agent.monotonic", lambda: clock[0])
    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, headers={"retry-after": "140"}, json={"error": {
                "message": "Rate limit on tokens per day (TPD). synthetic-secret",
            }})
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": answer("OK")}],
                                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            llm = GroqClient(http, "synthetic-model")
            with pytest.raises(QuotaPauseError, match="TPD"):
                await llm.complete([], [])
            llm.start_attempt()  # A new attempt does not reset a provider deadline.
            for action in (lambda: llm.complete([], []), llm.check_access):
                with pytest.raises(QuotaPauseError, match="140 с"):
                    await action()
            assert len(requests) == 1 and llm.usage_totals["requests"] == 1
            clock[0] += 140
            assert (await llm.complete([], []))["content"] == "OK"
            assert len(requests) == 2
    asyncio.run(run())


def test_invalid_tool_format_does_not_break_evidence_extraction():
    agent = fixture_agent([calling({"id": "invalid", "type": "function"}), answer("Не удалось подтвердить данные.")])
    asyncio.run(agent.ask("Тест"))
    assert list(tool_evidence(agent.turns[0][0])) == []


def test_access_check_quota_pause_directs_user_to_check_instead_of_retry():
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"data": [{"id": "synthetic-model"}]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            llm = GroqClient(http, "synthetic-model")
            llm.update_limits({"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "60s"})
            with pytest.raises(QuotaPauseError) as error:
                await llm.check_access()
            assert "/check" in str(error.value) and "/retry" not in str(error.value)
            assert llm.usage_totals["requests"] == 0
    asyncio.run(run())
    assert [request.method for request in requests] == ["GET"]
