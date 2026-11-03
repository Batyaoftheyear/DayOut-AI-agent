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


def place(ident,url,title='Synthetic place'):
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
    llm=SimpleNamespace(planning_enabled=True,complete=AsyncMock(side_effect=responses))
    return DayOutAgent(SimpleNamespace(call_tool=AsyncMock(side_effect=results)),llm,tools)


def search(radius=None):
    args={'textQuery':'synthetic places'}
    if radius:
        args['locationBias']={'circle':{'center':POINT,'radiusMeters':radius}}
    return call('search_places',args)


def walking(destination):
    return call('compute_routes',{'origin':{'placeId':'anchor'},'destination':destination,'travelMode':'WALK'})


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
