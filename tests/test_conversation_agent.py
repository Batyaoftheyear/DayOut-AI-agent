import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, Tool

from agent import ALLOWED_TOOLS, AnswerValidationError, DayOutAgent
from conversation_context import CONTEXT_TOOL_NAME, empty_context, update_context
from dayout import safe_error

ANCHOR='https://example.test/anchor'
NEAR='https://example.test/near'
FAR='https://example.test/far'
ROUTE='https://example.test/route'
POINT={'latitude':34.69,'longitude':135.5}


def call(name,args,ident='native'):
    return {'role':'assistant','content':None,'tool_calls':[{'id':ident,'type':'function',
        'function':{'name':name,'arguments':json.dumps(args,ensure_ascii=False)}}]}


def answer(text):
    return {'role':'assistant','content':text}


def reviewed(snapshot,action='answer',missing=(),next_step='',recommended_sources=()):
    return snapshot|{'response_check':{'action':action,'missing':list(missing),'next_step':next_step,
                                     'recommended_sources':list(recommended_sources)}}


def place(ident,url,title=None):
    title=title or ('Synthetic anchor' if ident=='anchor' else 'Synthetic place')
    return CallToolResult(content=[],structuredContent={'summary':title,'places':[
        {'id':ident,'place':'places/'+ident,'location':POINT,
         'attribution':{'title':title+' - Google Maps','url':url}}]})


def route(duration,url=ROUTE):
    return CallToolResult(content=[],structuredContent={'routes':[{'duration':str(duration)+'s',
        'attribution':{'title':'Synthetic route','url':url}}]})


def agent(responses,results):
    schemas={
        'search_places':{'type':'object','properties':{'textQuery':{'type':'string'},'pageSize':{'type':'integer'},
            'locationBias':{'type':'object'}},'required':['textQuery'],'additionalProperties':False},
        'compute_routes':{'type':'object','properties':{'origin':{'type':'object'},'destination':{'type':'object'},
            'travelMode':{'enum':['WALK','DRIVE']}},'required':['origin','destination','travelMode']},
        'lookup_weather':{'type':'object'},'resolve_names':{'type':'object'},'resolve_maps_urls':{'type':'object'}}
    tools={name:Tool(name=name,inputSchema=schema) for name,schema in schemas.items()}
    values=iter(responses)
    async def complete(*args,**kwargs):
        message=next(values)
        if isinstance(message,BaseException):raise message
        if (kwargs.get('response_format') or {}).get('json_schema',{}).get('name')=='dayout_response_review':
            calls=message.get('tool_calls') or []
            if calls and calls[0]['function']['name']==CONTEXT_TOOL_NAME:
                return answer(calls[0]['function']['arguments'])
        return message
    llm=SimpleNamespace(complete=AsyncMock(side_effect=complete))
    return DayOutAgent(SimpleNamespace(call_tool=AsyncMock(side_effect=results)),llm,tools)


def search(radius=None):
    args={'textQuery':'synthetic places'}
    if radius:
        args['locationBias']={'circle':{'center':POINT,'radiusMeters':radius}}
    return call('search_places',args)


def walking(destination):
    return call('compute_routes',{'origin':{'address':'Synthetic anchor'},'destination':destination,'travelMode':'WALK'})


def test_named_landmark_can_be_searched_without_a_city_or_preliminary_plan():
    a=agent([call('search_places',{'textQuery':'Osaka Castle'}),answer('Ориентир найден [S1].')],[place('anchor',ANCHOR)])
    reply=asyncio.run(a.ask('Куда сходить рядом с Osaka Castle?'))
    assert reply.tools==['search_places'] and a.llm.complete.await_count==2
    assert not any('response_format' in c.kwargs for c in a.llm.complete.await_args_list)
    assert a.session.call_tool.await_args.kwargs['arguments']['textQuery']=='Osaka Castle'


@pytest.mark.parametrize('fake',[
    'tool_calls: [{"tool_name":"search_places","parameters":{"location":"Osaka"}}]',
    '```json\n{"tool_calls":[{"name":"search_places"}]}\n```',
    '{"tool_name":"search_places","parameters":{"type":"park"}}',
    '[search_places: park Osaka]',
    '`search_places({"textQuery":"ресторан азиатской кухни ВДНХ"})`',
    'После получения данных подскажу места.\n```python\nsearch_places({"textQuery":"кафе"})\n```',
    'search_places: {"textQuery":"азиатские рестораны ВДНХ"}',
    'После получения данных подскажу конкретные павильоны.',
])
def test_city_reply_resumes_actual_dialogue_and_protocol_text_is_repaired(fake):
    a=agent([answer('В каком городе искать места?'),answer(fake),
        call('search_places',{'textQuery':'parks near Osaka Castle'}),answer('Кандидат [S1].')],[place('near',NEAR)])
    async def run():
        await a.ask('Куда сходить рядом с замком?')
        return await a.ask('Osaka')
    reply=asyncio.run(run())
    assert reply.tools==['search_places'] and NEAR in reply.text and fake not in reply.text
    for invocation in a.llm.complete.await_args_list:
        assert invocation.args[2] is True
        assert {t['function']['name'] for t in invocation.args[1]}==ALLOWED_TOOLS|{CONTEXT_TOOL_NAME}
    messages=a.llm.complete.await_args_list[1].args[0]
    assert any(m['role']=='assistant' and m.get('content')=='В каком городе искать места?' for m in messages)
    assert any(m['role']=='user' and 'рядом с замком' in m.get('content','') for m in messages)


def test_ambiguous_landmark_can_produce_one_model_clarification_with_sources():
    results=CallToolResult(content=[],structuredContent={'summary':'Two different fictional cities.',
        'places':[{'attribution':{'title':'First landmark','url':ANCHOR}},
                  {'attribution':{'title':'Second landmark','url':NEAR}}]})
    a=agent([call('search_places',{'textQuery':'Temple with ambiguous name'}),
        answer('Есть два разных ориентира: [S1] и [S2]. Какой имеется в виду?')],[results])
    reply=asyncio.run(a.ask('Найди кафе рядом с храмом с неоднозначным названием.'))
    assert '?' in reply.text and reply.tools==['search_places']
    assert a.llm.complete.await_count==2


def test_soft_fifteen_minute_target_expands_instead_of_ending_with_refusal():
    a=agent([search(),search(1200),walking({'placeId':'near'}),answer('В пределах 15 минут не удалось найти вариант.'),
        search(2800),walking({'placeId':'far'}),answer('Чуть дальше: кандидат [S4], 18 минут пешком [S5].')],
        [place('anchor',ANCHOR),place('near',NEAR),route(1200),place('far',FAR),route(1080,ROUTE+'-far')])
    reply=asyncio.run(a.ask('Найди рамен рядом с этим ориентиром.'))
    searches=[c.kwargs['arguments'] for c in a.session.call_tool.await_args_list if c.args[0]=='search_places']
    assert searches[-1]['locationBias']['circle']['radiusMeters']==2800
    assert FAR in reply.text and '18 минут' in reply.text and len(reply.tools)==5


def test_explicit_strict_limit_rejects_a_long_route_and_does_not_force_expansion():
    a=agent([call(CONTEXT_TOOL_NAME,{'proximity':{'minutes':15,'strict':True}}),
        search(),search(1200),walking({'placeId':'near'}),answer('Подойдёт кандидат [S2], 20 минут [S3].'),
        answer('Этот кандидат не подходит под предел: 20 минут [S2] [S3].')],
        [place('anchor',ANCHOR),place('near',NEAR),route(1200)])
    reply=asyncio.run(a.ask('Только до 15 минут пешком, дальше не подходит.'))
    assert 'не подходит' in reply.text and len(reply.tools)==3
    assert a.context['proximity']=={'minutes':15,'strict':True}


def test_route_by_exact_candidate_name_is_linked_and_unlinked_route_does_not_cause_global_refusal():
    a=agent([search(),walking({'address':'Synthetic place'}),answer('Кандидат [S1], две минуты пешком [S2].')],
        [place('near',NEAR),route(120)])
    reply=asyncio.run(a.ask('Проверь путь к найденному месту.'))
    check=next(p['app_route_check'][0] for _,p in __import__('agent').tool_evidence(a.turns[-1][0]) if 'app_route_check' in p)
    assert check['association_confirmed'] and NEAR in check['candidate_urls'] and NEAR in reply.text


def test_local_memory_preserves_budget_and_people_without_controlling_tool_access():
    a=agent([call(CONTEXT_TOOL_NAME,{'city':'Osaka','preferences':['можно с собакой'],
        'budget':{'action':'set','amount':3000,'currency':'RUB','people':2,'scope':'total'}}),answer('Условия сохранены.'),
        call(CONTEXT_TOOL_NAME,{'budget':{'action':'clear'}}),answer('Денежное ограничение отменено.')],[])
    async def run():
        first=await a.ask('Запомни: двое, 3000 рублей, можно с собакой. Пока не ищи.')
        assert a.context['budget']['amount']==3000 and first.tools==[]
        return await a.ask('Отменяем бюджет, остальное прежнее.')
    reply=asyncio.run(run())
    assert not reply.tools and a.session.call_tool.await_count==0
    assert a.context['budget']['amount'] is None and a.context['budget']['people']==2
    assert a.context['preferences']==['можно с собакой']


def test_context_updates_are_rolled_back_when_the_external_request_fails():
    a=agent([call(CONTEXT_TOOL_NAME,{'preferences':['веранда']}),search()],[RuntimeError('synthetic-private-detail')])
    with pytest.raises(Exception):
        asyncio.run(a.ask('Найди место с верандой.'))
    assert a.context==empty_context() and not a.turns


def test_repeated_protocol_failure_has_a_plain_user_error():
    a=agent([answer('tool_calls: []'),answer('tool_calls: []')],[])
    with pytest.raises(AnswerValidationError) as captured:
        asyncio.run(a.ask('Найди места в Osaka.'))
    message=safe_error(captured.value)
    assert 'tool_calls' not in message and 'JSON' not in message and not a.turns


def test_local_memory_rejects_unknown_fields_and_preserves_other_state():
    context=empty_context()
    with pytest.raises(Exception):
        update_context(context,{'arbitrary_command':'invalid'})
    assert context==empty_context()


def test_context_is_synchronized_after_search_and_cannot_route_or_block_it():
    snapshot=empty_context()
    snapshot.update(city='Osaka',preferences=['можно с собакой'])
    snapshot['budget'].update(action='set',amount=3000,currency='RUB',scope='total',people=2)
    a=agent([search(),answer('Кандидат [S1].'),call(CONTEXT_TOOL_NAME,reviewed(snapshot))],[place('near',NEAR)])
    a.llm.context_sync_required=True
    reply=asyncio.run(a.ask('Найди кафе в Осаке на двоих до 3000 рублей, можно с собакой.'))
    assert a.session.call_tool.await_count==1 and reply.tools==['search_places']
    assert a.llm.complete.await_count==3 and NEAR in reply.text
    assert '3000' in reply.text and a.context['city']=='Osaka' and a.context['budget']['people']==2
    assert 'tool_choice' not in a.llm.complete.await_args_list[0].kwargs
    assert a.llm.complete.await_args_list[-1].kwargs['response_format']['json_schema']['name']=='dayout_response_review'


def test_early_optional_memory_does_not_skip_final_snapshot():
    first=empty_context()
    final=empty_context()
    final['proximity']['minutes']=15
    final['proximity']['strict']=True
    a=agent([call(CONTEXT_TOOL_NAME,first),answer('Условия сохранены.'),
        call(CONTEXT_TOOL_NAME,reviewed(final))],[])
    a.llm.context_sync_required=True
    reply=asyncio.run(a.ask('Только до 15 минут пешком, больший путь не подходит. Пока не ищи.'))
    assert not reply.tools and a.context['proximity']['strict']
    assert a.llm.complete.await_count==3
    assert a.llm.complete.await_args_list[-1].kwargs['response_format']['json_schema']['name']=='dayout_response_review'


def test_resolver_without_coordinates_gives_actionable_recovery_instead_of_repeating_circle():
    resolved=CallToolResult(content=[],structuredContent={'places':[{'id':'anchor','place':'places/anchor'}]})
    a=agent([call('resolve_names',{'queries':['Synthetic anchor']}),search(1200),
        search(),search(1200),answer('Кандидат [S1].')],[resolved,place('anchor',ANCHOR),place('near',NEAR)])
    asyncio.run(a.ask('Найди место рядом с ориентиром.'))
    payloads=[json.loads(m['content']) for m in a.turns[-1][0] if m.get('role')=='tool']
    errors=[p for p in payloads if p.get('error')]
    assert len(errors)==1 and 'БЕЗ locationBias' in errors[0]['error']
    assert a.session.call_tool.await_count==3


def test_strict_expansion_is_not_sent_to_maps_and_honest_refusal_is_allowed():
    snapshot=empty_context()
    snapshot['proximity']['minutes']=15
    snapshot['proximity']['strict']=True
    a=agent([search(),search(1200),walking({'placeId':'near'}),call(CONTEXT_TOOL_NAME,snapshot),
        search(3000),answer('В пределах 15 минут подходящих мест не найдено. Маршрут 20 минут не подходит [S2] [S3].')],
        [place('anchor',ANCHOR),place('near',NEAR),route(1200)])
    reply=asyncio.run(a.ask('Только до 15 минут пешком.'))
    sent=[c.args[0] for c in a.session.call_tool.await_args_list]
    assert sent==['search_places','search_places','compute_routes']
    assert 'не подходит' in reply.text and '20' in reply.text


def test_unfinished_itinerary_is_reviewed_against_the_actual_draft_and_resumes_tools():
    snapshot=empty_context();snapshot.update(city='Москва',area='ВДНХ',preferences=['азиатская кухня'])
    draft='Погуляйте и посетите павильоны. Какую кухню предпочитаете?'
    a=agent([answer(draft),call(CONTEXT_TOOL_NAME,reviewed(snapshot,'continue',
        ['Нет конкретных павильонов и ресторанов'],'Найди конкретные павильоны и азиатский ресторан на ВДНХ.')),
        call('search_places',{'textQuery':'павильоны ВДНХ Москва'}),
        call('search_places',{'textQuery':'азиатский ресторан ВДНХ Москва'}),
        answer('Павильон «Синтетический космос» [S1], затем ресторан «Синтетическая Азия» [S2].'),
        call(CONTEXT_TOOL_NAME,reviewed(snapshot))],[place('pavilion',ANCHOR),place('asian',NEAR)])
    a.llm.context_sync_required=True
    reply=asyncio.run(a.ask('Сценарий вечера на ВДНХ: конкретные павильоны и ресторан, азиатская кухня.'))
    checkpoint=a.llm.complete.await_args_list[1].args[0][-1]['content']
    assert json.loads(checkpoint)['draft']==draft and a.session.call_tool.await_count==2
    assert ANCHOR in reply.text and NEAR in reply.text and '?' not in reply.text
    assert a.context['preferences']==['азиатская кухня']
    assert all(not message.get('app_internal') for request in a.llm.complete.await_args_list for message in request.args[0])
    assert all({d['function']['name'] for d in c.args[1]}==ALLOWED_TOOLS
               for c in a.llm.complete.await_args_list if c.args[1])


def test_specific_price_question_resumes_search_and_does_not_append_unknown_price_template():
    before=empty_context();before.update(city='Москва',area='Люблино')
    priced=empty_context();priced.update(city='Москва',area='Люблино')
    priced['budget'].update(action='set',amount=5000,currency='RUB',scope='total',people=2)
    price=place('lounge',NEAR,'Диапазон расходов 3000–8000 рублей на человека.')
    a=agent([search(),answer('Лаундж [S1].'),call(CONTEXT_TOOL_NAME,reviewed(before)),
        answer('Стоимость неизвестна.'),call(CONTEXT_TOOL_NAME,reviewed(priced,'continue',
            ['Цена не проверена целевым запросом'],'Проверь расходы на человека у найденного лаунджа.')),
        call('search_places',{'textQuery':'Лаундж Люблино Москва ценовой диапазон на человека'}),
        answer('Диапазон 3000–8000 рублей на человека [S1]. Для двоих ориентир 6000–16000 рублей; бюджет 5000 ниже него.'),
        call(CONTEXT_TOOL_NAME,reviewed(priced))],[place('lounge',NEAR),price])
    a.llm.context_sync_required=True
    async def run():
        await a.ask('Найди лаундж в Люблино.')
        return await a.ask('Бюджет 5000 рублей на двоих, укладываемся ли?')
    reply=asyncio.run(run())
    assert a.session.call_tool.await_count==2 and 'ценовой' in a.session.call_tool.await_args.kwargs['arguments']['textQuery']
    assert '6000' in reply.text and '5000' in reply.text and 'Стоимость неизвестна' not in reply.text
    assert 'Полная стоимость посещения не подтверждена' not in reply.text
    assert a.context['budget']['people']==2


def test_strict_ten_minutes_rejects_632_seconds_and_reports_ceil_minutes():
    a=agent([call(CONTEXT_TOOL_NAME,{'proximity':{'minutes':10,'strict':True}}),
        search(),search(1200),walking({'placeId':'near'}),answer('Подойдёт кафе [S2], до 10 минут пешком [S3].'),
        answer('Подходящий вариант не подтверждён. Этот кандидат не подходит: 10 минут 32 секунды [S2] [S3].')],
        [place('anchor',ANCHOR),place('near',NEAR),route(632)])
    reply=asyncio.run(a.ask('Не больше 10 минут от метро.'))
    checks=[p['app_route_check'][0] for _,p in __import__('agent').tool_evidence(a.turns[-1][0]) if p.get('app_route_check')]
    assert checks[0]['duration_minutes_ceil']==11 and checks[0]['origin_urls']==[ANCHOR]
    assert 'не подходит' in reply.text


def test_unverified_candidate_under_strict_limit_requires_a_route_instead_of_a_caveat():
    a=agent([call(CONTEXT_TOOL_NAME,{'proximity':{'minutes':10,'strict':True}}),search(),
        answer('Вариант кафе [S1]. Время до метро неизвестно, уточните, важно ли это.'),
        walking({'placeId':'near'}),answer('Кафе [S1], 8 минут пешком [S2].')],
        [place('near',NEAR),route(480)])
    reply=asyncio.run(a.ask('Кафе, строго не больше 10 минут от метро.'))
    assert reply.tools==['search_places','compute_routes'] and '8 минут' in reply.text


def test_last_place_evidence_survives_an_intermediate_weather_turn():
    weather=CallToolResult(content=[],structuredContent={'summary':'Облачно','fixture_weather':True})
    a=agent([search(),answer('Место [S1].'),call('lookup_weather',{'location':{'address':'Москва'}}),
        answer('Облачно.')],[place('near',NEAR),weather])
    async def run():
        await a.ask('Найди кафе в Москве.')
        await a.ask('Какая погода?')
    asyncio.run(run())
    history=json.dumps(a.build_history(),ensure_ascii=False)
    assert 'fixture_weather' in history and 'Synthetic place' in history and NEAR in history


def test_repaired_answer_gets_a_fresh_memory_and_completion_check():
    first=empty_context();final=empty_context();final['city']='Москва'
    a=agent([search(),answer('Найден кандидат.'),call(CONTEXT_TOOL_NAME,reviewed(first)),
        answer('Кандидат [S1].'),call(CONTEXT_TOOL_NAME,reviewed(final))],[place('near',NEAR)])
    a.llm.context_sync_required=True
    reply=asyncio.run(a.ask('Найди место в Москве.'))
    assert NEAR in reply.text and a.context['city']=='Москва'
    assert a.llm.complete.await_count==5


def test_unspecified_walking_time_is_not_saved_as_a_user_constraint():
    context=empty_context()
    assert context['proximity']=={'minutes':None,'strict':False}
    with pytest.raises(ValueError):
        update_context(context,{'proximity':{'minutes':None,'strict':True}})


def test_reviewed_strict_recommendations_use_route_facts_and_allow_declined_mentions():
    snapshot=empty_context();snapshot['proximity']={'minutes':10,'strict':True}
    a=agent([search(),search(1200),walking({'placeId':'near'}),
        answer('Кафе [S2] укладывается в 10 минут [S3].'),
        call(CONTEXT_TOOL_NAME,reviewed(snapshot,recommended_sources=['[S2]'])),
        answer('Кафе [S2] не укладывается в 10 минут: 10 минут 32 секунды [S3].'),
        call(CONTEXT_TOOL_NAME,reviewed(snapshot))],
        [place('anchor',ANCHOR),place('near',NEAR),route(632)])
    a.llm.context_sync_required=True
    reply=asyncio.run(a.ask('Кафе, не больше 10 минут пешком от метро.'))
    assert 'не укладывается' in reply.text and a.session.call_tool.await_count==3
    repair_request=a.llm.complete.await_args_list[-2].args[0]
    assert 'within_limit' in repair_request[1]['content'] and 'false' in repair_request[1]['content']


def test_new_independent_place_task_can_clear_an_old_anchor_bound_limit():
    old=empty_context();old.update(city='Москва',area='Коломенская')
    old['proximity']={'minutes':10,'strict':True}
    new=empty_context();new.update(city='Москва',area='ВДНХ',preferences=['азиатская кухня'])
    a=agent([answer('Условия сохранены.'),call(CONTEXT_TOOL_NAME,reviewed(old)),
        search(),answer('Павильон и ресторан [S1].'),call(CONTEXT_TOOL_NAME,reviewed(new,recommended_sources=['[S1]']))],
        [place('pavilion',NEAR)])
    a.llm.context_sync_required=True
    async def run():
        await a.ask('Запомни Коломенскую и не больше 10 минут от метро, пока не ищи.')
        return await a.ask('Теперь новая задача: сценарий вечера на ВДНХ, азиатская кухня.')
    reply=asyncio.run(run())
    assert NEAR in reply.text and a.context['proximity']['minutes'] is None
    assert a.context['preferences']==['азиатская кухня'] and a.session.call_tool.await_count==1


def test_explicit_currency_and_walking_bound_survive_a_wrong_model_snapshot():
    wrong=empty_context();wrong['budget'].update(amount=5000,people=2,scope='total')
    a=agent([search(),walking({'placeId':'near'}),answer('Кафе [S1], 8 минут пешком [S2]. Цены неизвестны.'),
        call(CONTEXT_TOOL_NAME,reviewed(wrong,recommended_sources=['[S1]']))],[place('near',NEAR),route(480)])
    a.llm.context_sync_required=True
    reply=asyncio.run(a.ask('Найди кафе, бюджет 5000 рублей на двоих, не больше 10 минут пешком от метро.'))
    assert a.context['budget']['currency']=='RUB' and a.context['proximity']=={'minutes':10,'strict':True}
    assert '5000 ₽' in reply.text and 'валюта не уточнена' not in reply.text


def test_literal_binding_is_conservative_for_ambiguous_currency_and_other_durations():
    from literal_constraints import bind_literals
    context=empty_context();context['budget']['amount']=5000
    actual=bind_literals(context,'Бюджет указан в RUB или USD? После прогулки пешком нужен фильм не больше 120 минут.')
    assert actual['budget']['currency']=='unknown' and actual['proximity']['minutes'] is None


def test_failed_user_refinement_survives_short_followup_without_leaking_rejected_output():
    a=agent([answer('search_places({"textQuery":"UNTRUSTED_COMMAND"})'),
        answer('search_places({"textQuery":"UNTRUSTED_COMMAND"})'),
        search(),answer('Синтетический павильон и азиатский ресторан [S1].'),
        answer('Азиатская кухня сохранена.')],[place('near',NEAR)])
    async def run():
        with pytest.raises(AnswerValidationError):
            await a.ask('Подбери конкретные павильоны и рестораны на ВДНХ, азиатская кухня.')
        assert a.unanswered_queries and not a.turns
        await a.ask('Так и?')
        return await a.ask('Какая кухня у нас выбрана?')
    reply=asyncio.run(run())
    third_request=json.dumps(a.llm.complete.await_args_list[-1].args[0],ensure_ascii=False)
    assert 'азиатская кухня' in third_request and 'ВДНХ' in third_request and 'Так и?' in third_request
    assert 'UNTRUSTED_COMMAND' not in third_request and not a.unanswered_queries
    assert len(a.user_history)==3 and 'Азиатская' in reply.text


def test_route_citation_in_review_is_mapped_to_its_confirmed_candidate():
    snapshot=empty_context();snapshot['proximity']={'minutes':10,'strict':True}
    a=agent([search(),walking({'placeId':'near'}),answer('Кафе [S1], 8 минут пешком [S2].'),
        call(CONTEXT_TOOL_NAME,reviewed(snapshot,recommended_sources=['[S2]']))],[place('near',NEAR),route(480)])
    a.llm.context_sync_required=True
    reply=asyncio.run(a.ask('Кафе, не больше 10 минут пешком.'))
    assert NEAR in reply.text and ROUTE in reply.text and a.session.call_tool.await_count==2


def test_invalid_route_id_does_not_consume_call_budget_and_returned_url_is_resolved():
    a=agent([search(),walking({'placeId':'invented-id'}),walking({'placeId':NEAR}),
        answer('Кафе [S1], 8 минут пешком [S2].')],[place('near',NEAR),route(480)])
    a.max_calls=2
    reply=asyncio.run(a.ask('Кафе, не больше 10 минут пешком.'))
    assert a.session.call_tool.await_count==2 and reply.tools==['search_places','compute_routes']
    assert a.session.call_tool.await_args.kwargs['arguments']['destination']=={'placeId':'near'}
    assert any(outcome['status']=='invalid_arguments' for outcome in reply.observations['tool_outcomes'])


def test_context_discards_large_visualization_tokens_without_losing_price_evidence():
    from agent import extract_payload
    result=CallToolResult(content=[],structuredContent={'contentToken':'opaque'*6000,
        'summary':'Диапазон расходов 3000–8000 рублей на человека.',
        'places':[{'id':'near','location':POINT,'attribution':{'title':'Synthetic place','url':NEAR}}]})
    payload=extract_payload(result,'search_places')
    assert 'contentToken' not in payload and '3000–8000' in payload['summary']
    assert payload['places'][0]['id']=='near' and 'error' not in payload


def test_strict_limit_checks_other_returned_candidates_without_more_model_steps():
    choices=CallToolResult(content=[],structuredContent={'summary':'Two synthetic cafes.',
        'places':place('near',NEAR).structuredContent['places']+place('far',FAR).structuredContent['places']})
    a=agent([search(),walking({'placeId':'near'}),answer('Кафе [S2], 8 минут пешком [S4].')],
        [choices,route(632),route(480,ROUTE+'-second')])
    reply=asyncio.run(a.ask('Кафе, не больше 10 минут пешком.'))
    assert a.session.call_tool.await_count==3 and a.llm.complete.await_count==3
    assert reply.tools==['search_places','compute_routes','compute_routes']
    facts=[p for _,p in __import__('agent').tool_evidence(a.turns[-1][0]) if p.get('app_walking_limit')]
    assert facts[0]['app_walking_limit']['passing_candidate_urls']==[FAR]
    assert FAR in reply.text and '8 минут' in reply.text


def test_fallible_completion_critic_cannot_force_endless_unrequested_checks():
    snapshot=empty_context()
    a=agent([answer('Кандидат [S1].'),call(CONTEXT_TOOL_NAME,reviewed(snapshot,'continue',
        ['Нужны дополнительные сведения'],'Проверь описание.')),
        search(),answer('Проверенный кандидат [S1].'),call(CONTEXT_TOOL_NAME,reviewed(snapshot,'continue',
        ['Хотелось бы ещё сведений'],'Ищи ещё.'))],[place('near',NEAR)])
    a.llm.context_sync_required=True
    reply=asyncio.run(a.ask('Предложи место для прогулки.'))
    assert 'Проверенный кандидат' in reply.text and a.session.call_tool.await_count==1
    assert reply.observations['semantic_review_unresolved'] is True


def test_synthetic_memory_checks_are_not_advertised_or_replayed_to_the_actor():
    snapshot=empty_context()
    a=agent([answer('Условия понял.'),call(CONTEXT_TOOL_NAME,reviewed(snapshot,'continue',
        ['Нужен поиск'],'Найди место.')),search(),answer('Место [S1].'),
        call(CONTEXT_TOOL_NAME,reviewed(snapshot))],[place('near',NEAR)])
    a.llm.context_sync_required=True
    asyncio.run(a.ask('Найди место.'))
    for request in a.llm.complete.await_args_list:
        if not request.args[1]:continue
        assert not any(call['function']['name']==CONTEXT_TOOL_NAME
                       for message in request.args[0] for call in message.get('tool_calls') or [])


def test_returned_route_ids_constrain_the_next_model_request_and_duplicates_are_reused():
    a=agent([search(),walking({'placeId':'near'}),walking({'placeId':'near'}),
        answer('Кафе [S1], 8 минут пешком [S2].'),
        call(CONTEXT_TOOL_NAME,reviewed(empty_context()))],[place('near',NEAR),route(480)])
    a.llm.context_sync_required=True
    reply=asyncio.run(a.ask('Проверь путь до кафе.'))
    definitions=a.llm.complete.await_args_list[1].args[1]
    schema=next(d['function']['parameters'] for d in definitions if d['function']['name']=='compute_routes')
    assert schema['properties']['destination']['properties']['placeId']['enum']==['near']
    assert a.session.call_tool.await_count==2 and reply.tools==['search_places','compute_routes']
