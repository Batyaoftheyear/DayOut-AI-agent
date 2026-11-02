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
            "required_tools": [], "reference_kind": "none", "price_requested": False,
            "weather_date": None, "weather_hour": None, "geo_anchor": None, "walk_limit_minutes": None}


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
        assert "Пожелания:" not in query
    main_wires = [invocation.args[0][0]["content"] for invocation in a.llm.complete.await_args_list
                  if not invocation.args[0][0]["content"].startswith("Разбери текущую просьбу")]
    assert all('"amount":4000' in wire and '"people":2' in wire and '"scope":"total"' in wire for wire in main_wires)
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
    assert query == "бани" and a.last_plan["budget"]["scope"] == "per_person"


@pytest.mark.parametrize("wording", ["Цены неизвестны.", "Ориентир — три тысячи рублей, стоимость не подтверждена."])
def test_budget_is_rendered_from_memory_without_a_repair_for_missing_digits(wording):
    a = agent([planned(plan(amount=3000, action="set")), call("кафе"),
               answer("Кандидат [S1]. " + wording)], [result(prices=False)])
    reply = asyncio.run(a.ask("Кафе в Москве на двоих, общий бюджет 3000 рублей."))
    assert "Бюджет запроса: 3000 ₽ на всех; участников: 2." in reply.text
    assert a.llm.complete.await_count == 3 and a.session.call_tool.await_count == 1
    assert a.last_plan["budget"]["amount"] == 3000


def test_a_proximity_plan_requires_route_evidence_before_a_recommendation():
    p = plan()
    p["area"] = "рядом с метро Синтетическая"
    p["required_tools"] = ["search_places", "compute_routes"]
    route_call = {"role": "assistant", "content": None, "tool_calls": [{"id": "route", "type": "function",
        "function": {"name": "compute_routes", "arguments": json.dumps({
            "origin": {"address": "метро Синтетическая, Москва"},
            "destination": {"placeId": "synthetic-place"}, "travelMode": "WALK"})}}]}
    a = agent([planned(p), call("кафе рядом с метро Синтетическая"), answer("Кафе рядом [S1]."),
               route_call, answer("Кандидат [S1], пеший маршрут подтверждён [S2].")],
              [result(), CallToolResult(content=[], structuredContent={
                  "attribution": {"title": "Synthetic walking route", "url": "https://example.test/walking"}})])
    a.tools["compute_routes"] = Tool(name="compute_routes", inputSchema={"type": "object"})
    reply = asyncio.run(a.ask("Найди кафе рядом с метро Синтетическая, Москва."))
    assert reply.tools == ["search_places", "compute_routes"]
    sent = a.session.call_tool.await_args.kwargs["arguments"]
    assert sent["travelMode"] == "WALK" and sent["origin"]["address"] == "метро Синтетическая, Москва"


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


def test_missing_city_becomes_one_question_without_search_or_another_generation():
    p = plan()
    p["city"] = ""
    a = agent([planned(p)], [])
    reply = asyncio.run(a.ask("Найди кафе; город я ещё не указал."))
    assert reply.text == "В каком городе искать места?" and not reply.tools
    assert a.llm.complete.await_count == 1


def test_people_change_keeps_amount_and_unit_but_updates_participants():
    p = plan(amount=1500, action="set")
    p["budget"]["scope"] = "per_person"
    previous = parse_plan(planned(p))
    changed = plan(action="set")
    changed["budget"].update(people=3, currency="unknown", scope="unknown")
    merged = parse_plan(planned(changed), previous)["budget"]
    assert merged["people"] == 3 and merged["amount"] == 1500 and merged["scope"] == "per_person"


def test_empty_search_does_not_force_a_route_between_invented_endpoints():
    p = plan()
    p["required_tools"] = ["search_places", "compute_routes"]
    a = agent([planned(p), call("кафе"), call("кафе поблизости", "two"),
               answer("Поиск не вернул кандидатов; маршрут построить не из чего.")],
              [result(empty=True), result(empty=True)])
    reply = asyncio.run(a.ask("Кафе рядом с ориентиром в Москве."))
    assert reply.tools == ["search_places", "search_places"]


def test_refinement_clears_unnecessary_tools_and_recovers_place_reference():
    previous=parse_plan(planned(plan()))
    p=plan('refine')
    p['required_tools']=['search_places']
    refined=parse_plan(planned(p),previous)
    assert refined['required_tools'] == [] and refined['reference_kind'] == 'places'


def test_place_sources_survive_two_unrelated_questions():
    search=plan()
    chat=plan('chat')
    refine=plan('refine')
    refine['reference_kind']='places'
    a=agent([planned(search),call('кафе'),answer('Кандидат [S1].'),
             planned(chat),answer('Первый общий ответ.'),planned(chat),answer('Второй общий ответ.'),
             planned(refine),answer('Прежний кандидат [S1].')],[result()])
    async def run():
        await a.ask('Кафе в Москве')
        await a.ask('Что такое MCP?')
        await a.ask('Для чего нужен API?')
        return await a.ask('Повтори найденный вариант без нового поиска')
    reply=asyncio.run(run())
    assert FIRST in reply.text and a.session.call_tool.await_count == 1


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
    args = {"location": {"address": "Москва, Россия"}, "date": {"year": 2026, "month": 10, "day": 30}, "languageCode": "ru", "capabilities": ["AIR_QUALITY"]}
    invocation = {"type": "function", "function": {"name": "lookup_weather", "arguments": json.dumps(args)}}
    payload, _ = asyncio.run(a.execute(invocation))
    sent = a.session.call_tool.await_args.kwargs["arguments"]
    assert sent["date"] == args["date"] and sent["location"] == args["location"]
    assert "languageCode" not in sent and "capabilities" not in sent and "error" not in payload
    p = plan("weather")
    p.update(weather_date="2026-10-31", weather_hour=None)
    p = parse_plan(planned(p))
    a.session.call_tool.side_effect = [CallToolResult(content=[], structuredContent={"weatherCondition": {"type": "CLOUDY"}})]
    wrong = {"type": "function", "function": {"name": "lookup_weather", "arguments": json.dumps(dict(args, hour=22))}}
    asyncio.run(a.execute(wrong, p))
    sent = a.session.call_tool.await_args.kwargs["arguments"]
    assert sent["date"] == {"year": 2026, "month": 10, "day": 31} and "hour" not in sent


def test_explicit_moscow_in_area_does_not_trigger_a_city_question():
    p=plan('clarify')
    p.update(city='',area='юг Москвы',activity='прогулка')
    a=agent([planned(p),call('парки на юге Москвы'),answer('Кандидат для прогулки [S1].')],[result()])
    reply=asyncio.run(a.ask('Найди красивые места для прогулки вдвоём на юге Москвы.'))
    assert a.last_plan['city']=='Москва' and a.last_plan['task']=='search'
    assert reply.tools==['search_places'] and FIRST in reply.text
    assert a.llm.complete.await_args_list[1].kwargs['tool_choice']['function']['name']=='search_places'


def test_new_search_request_overrides_a_previous_temporary_no_search_command():
    previous=parse_plan(planned(plan('chat',amount=3000,action='set')))
    p=plan('refine')
    parsed=parse_plan(planned(p),previous,'Предложи два тихих кафе в центре Москвы.')
    assert parsed['task']=='search' and parsed['required_tools']==['search_places']
    assert parsed['budget']['amount']==3000
    assert parse_plan(planned(p),previous,'Перечисли условия, новых мест не ищи.')['task']=='refine'


@pytest.mark.parametrize('query,activity',[
    ('Удобно завтра в 22:00. Проверь часы работы этих вариантов.','кафе'),
    ('А теперь лучше бани, район и бюджет те же.','бани'),
])
def test_hours_and_new_category_require_search_even_when_planner_calls_it_refinement(query,activity):
    previous=plan();previous['activity']='кафе'
    p=plan('refine');p['activity']='кафе'
    current=parse_plan(planned(p),previous,query)
    assert current['task']=='search' and current['activity']==activity and current['required_tools']==['search_places']


def test_dropping_the_area_restarts_search_without_the_old_metro_radius():
    previous=plan();previous.update(area='Алтуфьево',geo_anchor='метро Алтуфьево',walk_limit_minutes=15)
    p=dict(previous,task='refine')
    current=parse_plan(planned(p),previous,'А если не ограничиваться Алтуфьево?')
    assert current['task']=='search' and current['area']=='' and current['geo_anchor'] is None


def test_coordinates_from_retained_place_evidence_work_after_unrelated_turns():
    center={'latitude':55.7,'longitude':37.6}
    a=agent([], [result(SECOND)])
    previous=result().structuredContent
    previous['places'][0]['location']=center
    a.last_place_turn=([call('кафе'),{'role':'tool','tool_call_id':'one','content':json.dumps(previous)}],{FIRST:'Fixture'})
    a.tools['search_places'].inputSchema['properties']['locationBias']={'type':'object'}
    invocation=call('рестораны')['tool_calls'][0]
    invocation['function']['arguments']=json.dumps({'textQuery':'рестораны','locationBias':{'circle':{'center':center,'radiusMeters':1200}}})
    payload,_=asyncio.run(a.execute(invocation))
    assert 'error' not in payload
    assert a.session.call_tool.await_args.kwargs['arguments']['locationBias']['circle']['center']==center


def test_first_district_search_drops_unverified_coordinates_and_uses_named_location():
    a=agent([], [result()])
    a.tools['search_places'].inputSchema['properties']['locationBias']={'type':'object'}
    invocation=call('кафе')['tool_calls'][0]
    invocation['function']['arguments']=json.dumps({'textQuery':'кафе','locationBias':{'circle':{
        'center':{'latitude':55.7,'longitude':37.6},'radiusMeters':1000}}})
    payload,_=asyncio.run(a.execute(invocation,plan()))
    sent=a.session.call_tool.await_args.kwargs['arguments']
    assert 'error' not in payload and 'locationBias' not in sent
    assert 'Москва' in sent['textQuery'] and 'Нагатино' in sent['textQuery']


def test_missing_city_is_still_a_question_without_a_search():
    p=plan('search');p.update(city='',area='')
    a=agent([planned(p)],[])
    reply=asyncio.run(a.ask('Найди красивые места для прогулки вдвоём.'))
    assert reply.text=='В каком городе искать места?'
    a.session.call_tool.assert_not_awaited()


def test_printed_tool_syntax_does_not_count_as_a_real_result():
    a=agent([planned(plan()),call('парки Москва'),answer('[search_places: парк Москва]'),
             answer('Кандидат [S1].')],[result()])
    reply=asyncio.run(a.ask('Найди парк в Москве'))
    assert '[search_places:' not in reply.text and FIRST in reply.text
    assert a.session.call_tool.await_count==1


def test_restaurant_category_is_not_replaced_by_the_purpose_of_a_date():
    p=plan();p['activity']='свидание'
    parsed=parse_plan(planned(p),query='Найди два тихих ресторана для свидания рядом с метро Бутырская в Москве.')
    assert parsed['activity']=='рестораны'


def test_near_station_search_resolves_anchor_before_using_coordinates():
    p=plan();p.update(geo_anchor='метро Синтетическая',activity='кафе')
    center={'latitude':55.7,'longitude':37.6}
    anchor=result();anchor.structuredContent['places'][0]['location']=center
    found=result(SECOND);found.structuredContent['places'][0]['location']=dict(center,latitude=55.7001)
    a=agent([planned(p),call('свидание'),call('кафе','two'),answer('Можно рассмотреть [S2].')],[anchor,found])
    a.tools['search_places'].inputSchema['properties'].update({'pageSize':{'type':'integer'},'locationBias':{'type':'object'}})
    reply=asyncio.run(a.ask('Найди кафе рядом с метро Синтетическая, Москва'))
    first,second=[c.kwargs['arguments'] for c in a.session.call_tool.await_args_list]
    assert 'Синтетическая' in first['textQuery'] and 'locationBias' not in first
    assert second['textQuery']=='кафе' and second['locationBias']['circle']['center']==center
    assert SECOND in reply.text


def test_place_link_cannot_hide_a_route_over_the_requested_walking_limit():
    from agent import TurnState, AnswerValidationError
    p=plan();p['walk_limit_minutes']=15
    state=TurnState('test',[{'role':'tool','tool_call_id':'route','content':json.dumps({'app_route_check':[{
        'candidate_urls':[FIRST], 'duration_seconds':1296,'within_walk_limit':False}]})}],0,{},[],{})
    state.plan=p
    a=agent([],[])
    with pytest.raises(AnswerValidationError,match='21.6'):
        a.validate_geography(state,f'Подходящее кафе: [Google Maps]({FIRST}), 15 минут пешком.')
    a.validate_geography(state,f'Не подходит: [Google Maps]({FIRST}), путь превышает 15 минут.')


def test_nearby_fallback_uses_only_verified_route_and_cannot_include_far_candidates():
    from agent import TurnState
    p=plan();p.update(geo_anchor='метро Синтетическая',walk_limit_minutes=15)
    messages=[]
    for index,(name,payload) in enumerate([
        ('search_places',result('https://example.test/anchor').structuredContent),
        ('search_places',{'places':[{'attribution':{'title':'Fixture','url':FIRST}},
                                   {'attribution':{'title':'Far fixture','url':SECOND}}]}),
        ('compute_routes',{'app_route_check':[{'candidate_urls':[FIRST],'duration_seconds':840,'within_walk_limit':True},
                                              {'candidate_urls':[SECOND],'duration_seconds':1800,'within_walk_limit':False}]})]):
        messages.extend([{'role':'assistant','tool_calls':[{'id':str(index),'function':{'name':name}}]},
                         {'role':'tool','tool_call_id':str(index),'content':json.dumps(payload)}])
    state=TurnState('test',messages,0,{},[],{});state.plan=p
    text=agent([],[]).nearby_fallback(state)
    assert FIRST in text and SECOND not in text and '14 мин' in text


def test_grouped_source_markers_still_reject_unknown_sources():
    from agent import render_sources,SourceValidationError
    text=render_sources('Кандидаты [S1, S2].',{'[S1]':FIRST,'[S2]':SECOND})
    assert FIRST in text and SECOND in text
    with pytest.raises(SourceValidationError):
        render_sources('Кандидаты [S1, S99].',{'[S1]':FIRST})


def test_search_fallback_cannot_reuse_a_place_from_a_different_category():
    from agent import TurnState
    payload={'places':[{'attribution':{'title':'Synthetic bath - Google Maps','url':SECOND}}],
             'summary':f'**Synthetic bath** {SECOND} — в описании указаны парная и душ.'}
    messages=[call('бани'),{'role':'tool','tool_call_id':'one','content':json.dumps(payload)}]
    state=TurnState('test',messages,0,{},['search_places'],{});state.plan=plan()
    a=agent([],[]);a.last_place_turn=([{'role':'tool','tool_call_id':'old','content':json.dumps(result().structuredContent)}],{FIRST:'Old cafe'})
    text=a.search_fallback(state)
    assert SECOND in text and FIRST not in text and 'парная и душ' in text
