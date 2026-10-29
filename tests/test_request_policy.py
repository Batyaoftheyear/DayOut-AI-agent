import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, Tool

from agent import AgentError, DayOutAgent
from request_policy import parse_plan

FIRST = "https://example.test/synthetic-cafe"
SECOND = "https://example.test/synthetic-bath"


def plan(task="search", amount=None, action="keep", strict=False):
    return {"task": task, "city": "Москва", "area": "Нагатино", "activity": "досуг", "purpose": "свидание",
            "other_constraints": "", "budget": {"action": action, "amount": amount, "currency": "RUB",
                "people": 2, "scope": "total"}, "strict_area": strict,
            "required_tools": [], "reference_kind": "none", "price_requested": False}


def planned(value):
    return {"role": "assistant", "content": json.dumps(value, ensure_ascii=False)}


def call(query, ident="one"):
    return {"role": "assistant", "content": None, "tool_calls": [{"id": ident, "type": "function", "function": {
        "name": "search_places", "arguments": json.dumps({"textQuery": query}, ensure_ascii=False),
    }}]}


def answer(text):
    return {"role": "assistant", "content": text}


def result(url=FIRST, empty=False, prices=True):
    return CallToolResult(content=[], structuredContent={
        "summary": "Цена входа 500 руб. на человека." if prices else "Цены не указаны.",
        "places": [] if empty else [{"attribution": {"title": "Synthetic fixture", "url": url}}],
    })


def agent(responses, results):
    session = SimpleNamespace(call_tool=AsyncMock(side_effect=results))
    llm = SimpleNamespace(planning_enabled=True, complete=AsyncMock(side_effect=responses))
    tool = Tool(name="search_places", inputSchema={"type": "object", "properties": {
        "textQuery": {"type": "string"}}, "required": ["textQuery"], "additionalProperties": False})
    return DayOutAgent(session, llm, {"search_places": tool})


def test_budget_keep_and_clear_are_merged_by_code_not_overwritten_by_model_guess():
    previous = parse_plan(planned(plan(amount=4000, action="set")))
    kept = parse_plan(planned(plan(amount=999999, action="keep")), previous)
    assert kept["budget"] == previous["budget"]
    assert previous["budget"]["amount"] == 4000
    assert parse_plan(planned(plan(action="clear")), previous)["budget"]["amount"] is None


def test_new_category_cannot_be_refused_without_search_and_budget_reaches_both_queries():
    first, second = plan(amount=4000, action="set"), plan()
    first["activity"], second["activity"] = "кафе", "бани"
    a = agent([planned(first), call("кафе"), answer("Лимит 4000 руб.: кафе [S1]."),
               planned(second), answer("Нет данных о банях."), call("бани", "two"),
               answer("Лимит 4000 руб.: баня [S2].")], [result(), result(SECOND)])
    async def run():
        await a.ask("Найди кафе в Москве, Нагатино, на двоих, бюджет 4000 рублей.")
        return await a.ask("А теперь бани, остальные условия те же.")
    reply = asyncio.run(run())
    assert SECOND in reply.text and FIRST not in reply.text
    assert a.session.call_tool.await_count == 2
    for invoked in a.session.call_tool.await_args_list:
        query = invoked.kwargs["arguments"]["textQuery"]
        assert "4000 руб. на всех" in query and "число людей: 2" in query
    assert "бани" in a.session.call_tool.await_args.kwargs["arguments"]["textQuery"]
    assert a.last_plan["budget"]["amount"] == 4000


def test_general_question_after_search_has_no_old_place_sources_or_new_search():
    a = agent([planned(plan()), call("кафе"), answer("Вариант [S1]."),
               planned(plan("chat")), answer("MCP связывает приложение с инструментами.")], [result()])
    async def run():
        await a.ask("Найди кафе в Нагатино, Москва.")
        return await a.ask("Что такое MCP?")
    reply = asyncio.run(run())
    assert "http" not in reply.text and "Источники" not in reply.text
    assert not reply.tools and a.session.call_tool.await_count == 1


def test_unrequested_budget_boilerplate_is_repaired_without_another_search():
    a = agent([planned(plan()), call("кафе"), answer("Вариант [S1]. Попадание в бюджет не подтверждено."),
               answer("Вариант для встречи [S1].")], [result()])
    reply = asyncio.run(a.ask("Найди место для свидания в Москве."))
    assert "бюджет" not in reply.text and FIRST in reply.text
    a.session.call_tool.assert_awaited_once()


def test_empty_search_gets_one_nearby_attempt_and_keeps_geography_explicit():
    a = agent([planned(plan()), call("в Нагатино"), answer("Нет данных."), call("рядом с Нагатино", "two"),
               answer("Рядом с Нагатино, за исходным районом: вариант [S1].")], [result(empty=True), result()])
    reply = asyncio.run(a.ask("Найди бани в Нагатино, Москва."))
    assert a.session.call_tool.await_count == 2 and "за исходным районом" in reply.text
    assert [item["status"] for item in reply.observations["tool_outcomes"]] == ["empty", "success"]


def test_strict_area_does_not_force_expanding_an_empty_search():
    a = agent([planned(plan(strict=True)), call("строго в Нагатино"),
               answer("В указанном районе поиск не вернул вариантов.")], [result(empty=True)])
    reply = asyncio.run(a.ask("Только в Нагатино, Москва. Другие районы не подходят."))
    assert "http" not in reply.text
    a.session.call_tool.assert_awaited_once()


def test_budget_and_price_request_reach_search_without_blocking_an_honest_unknown():
    p = plan(amount=1500, action="set")
    p["budget"]["scope"] = "per_person"
    a = agent([planned(p), call("бани"),
               answer("Лимит 1500 руб. на человека: цены не найдены, попадание в бюджет не подтверждено [S1].")],
              [result(prices=False)])
    reply = asyncio.run(a.ask("Бани в Москве на двоих, 1500 рублей на человека."))
    assert a.session.call_tool.await_count == 1 and "1500" in reply.text
    query = a.session.call_tool.await_args.kwargs["arguments"]["textQuery"]
    assert "1500 руб. на человека" in query and "нужны цены с единицами" in query


def test_failed_search_can_be_reported_without_forcing_budget_prose_repairs():
    a = agent([planned(plan(amount=4000, action="set")), call("кафе"),
               answer("Поиск временно не выполнился. Данные не получены.")],
              [CallToolResult(content=[], isError=True)])
    reply = asyncio.run(a.ask("Найди кафе в Москве на двоих, бюджет 4000 рублей."))
    assert reply.observations["tool_outcomes"] == [{"tool": "search_places", "status": "error"}]
    assert a.llm.complete.await_count == 3


def test_a_failed_price_attempt_does_not_force_a_third_search():
    a = agent([planned(plan(amount=4000, action="set")), call("кафе"),
               call("цены выбранного кафе", "two"),
               answer("Вариант [S1]. Лимит 4000 руб.: цены не получены, бюджет не подтверждён.")],
              [result(prices=False), CallToolResult(content=[], isError=True)])
    reply = asyncio.run(a.ask("Кафе в Москве на двоих до 4000 рублей."))
    assert a.session.call_tool.await_count == 2 and FIRST in reply.text


def test_unknown_prices_cannot_be_combined_with_a_budget_guarantee():
    a = agent([planned(plan(amount=4000, action="set")), call("кафе"),
               answer("Вариант в рамках бюджета 4000 рублей [S1]. Стоимость неизвестна; попадание в бюджет не подтверждено."),
               answer("Кандидат при ориентире 4000 рублей [S1]. Стоимость неизвестна; попадание в бюджет не подтверждено.")],
              [result(prices=False)])
    reply = asyncio.run(a.ask("Кафе в Москве до 4000 рублей."))
    assert "в рамках бюджета" not in reply.text and "4000" in reply.text
    a.session.call_tool.assert_awaited_once()
    assert json.loads(a.turns[0][0][1]["tool_calls"][0]["function"]["arguments"])["textQuery"] == a.session.call_tool.await_args.kwargs["arguments"]["textQuery"]


def test_bad_plan_is_not_echoed_or_committed_and_does_not_reach_mcp():
    a = agent([answer("synthetic-private-input")], [])
    with pytest.raises(AgentError, match="некорректный план") as error:
        asyncio.run(a.ask("Найди кафе."))
    assert "synthetic-private-input" not in str(error.value) and not a.turns
    a.session.call_tool.assert_not_awaited()


@pytest.mark.parametrize("prefix", ["", " ", "\n\t"])
def test_planner_accepts_whitespace_around_a_markdown_json_block(prefix):
    message = answer(prefix + "```json\n" + json.dumps(plan()) + "\n```\n ")
    assert parse_plan(message)["task"] == "search"


def test_reset_discards_previous_structured_constraints():
    a = agent([planned(plan(amount=4000, action="set")), call("кафе"), answer("Лимит 4000 руб., вариант [S1].")], [result()])
    asyncio.run(a.ask("Кафе в Москве на двоих, бюджет 4000."))
    assert a.last_plan["budget"]["amount"] == 4000
    a.reset()
    assert a.last_plan is None and not a.user_history


def test_weather_oneof_and_calendar_errors_are_caught_before_network():
    from agent import validate_tool_semantics
    valid = {"location": {"address": "Москва, Россия"}, "date": {"year": 2026, "month": 10, "day": 30}}
    validate_tool_semantics("lookup_weather", valid)
    invalid = [dict(valid, hour=24), dict(valid, location={"address": "Москва", "placeId": "synthetic"}),
               dict(valid, date={"year": 2026, "month": 2, "day": 30}),
               {"location": valid["location"], "hour": 22}]
    for value in invalid:
        with pytest.raises(ValueError):
            validate_tool_semantics("lookup_weather", value)


def test_weather_uses_api_default_language_and_preserves_requested_date():
    a = agent([], [CallToolResult(content=[], structuredContent={"weatherCondition": {"type": "CLOUDY"}})])
    a.tools["lookup_weather"] = Tool(name="lookup_weather", inputSchema={"type": "object", "properties": {
        "location": {"type": "object"}, "date": {"type": "object"}, "languageCode": {"type": "string"}},
        "required": ["location"]})
    args = {"location": {"address": "Москва, Россия"}, "date": {"year": 2026, "month": 10, "day": 30}, "languageCode": "ru"}
    invocation = {"type": "function", "function": {"name": "lookup_weather", "arguments": json.dumps(args)}}
    payload, _ = asyncio.run(a.execute(invocation))
    sent = a.session.call_tool.await_args.kwargs["arguments"]
    assert sent["date"] == args["date"] and sent["location"] == args["location"]
    assert "languageCode" not in sent and "error" not in payload
