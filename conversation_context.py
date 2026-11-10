"""Optional conversational memory, updated through a native local tool.

This state never selects or disables MCP tools. The model always sees the
dialogue and the discovered tools, and chooses its next action itself.
"""
import copy
import math

from jsonschema import validate
from response_review import RESPONSE_CHECK_SCHEMA

CONTEXT_TOOL_NAME='remember_context'
CONTEXT_SCHEMA={
    'type':'object','additionalProperties':False,
    'properties':{
        'city':{'type':'string','maxLength':160},
        'area':{'type':'string','maxLength':160},
        'activity':{'type':'string','maxLength':160},
        'preferences':{'type':'array','maxItems':12,'items':{'type':'string','maxLength':160}},
        'budget':{'type':'object','additionalProperties':False,'required':['action'],
            'properties':{
                'action':{'enum':['keep','set','clear']},
                'amount':{'type':['number','null'],'minimum':0},
                'currency':{'enum':['RUB','USD','EUR','unknown']},
                'people':{'type':['integer','null'],'minimum':1,'maximum':100},
                'scope':{'enum':['total','per_person','per_hour','unknown']}}},
        'proximity':{'type':'object','additionalProperties':False,'required':['minutes','strict'],
            'properties':{'minutes':{'type':'integer','minimum':1,'maximum':120,
                'description':'Исходный ориентир или предел пользователя, по умолчанию 15. Не подменяй его временем найденного маршрута.'},'strict':{'type':'boolean'}}}
    }
}
CONTEXT_TOOL={'type':'function','function':{'name':CONTEXT_TOOL_NAME,
    'description':'Сохранить изменённые условия пользователя. Это локальная память, не поиск и не данные Maps. '
        'Передай полный текущий снимок условий. preferences — полный список любых пожеланий; отменённые удали. '
        'Бюджет clear отменяет денежное ограничение. proximity.strict=true только для явно жёсткого предела пользователя. '
        'Сохрани прежние условия, если пользователь их не менял. Неизвестный город — пустая строка; сначала можно определить ориентир через Maps.',
    'parameters':CONTEXT_SCHEMA}}

# A complete snapshot is a final state check, not a preliminary action plan.
CONTEXT_TOOL['function']['parameters']=copy.deepcopy(CONTEXT_SCHEMA)
CONTEXT_TOOL['function']['parameters']['required']=list(CONTEXT_SCHEMA['properties'])
CONTEXT_TOOL['function']['parameters']['properties']['budget']['required']=list(CONTEXT_SCHEMA['properties']['budget']['properties'])
CONTEXT_TOOL['function']['parameters']['properties']['response_check']=RESPONSE_CHECK_SCHEMA
CONTEXT_TOOL['function']['description']+=' При проверке готового черновика заполни response_check: answer или continue с недостающими данными и следующим действием.'


def empty_context():
    return {'city':'','area':'','activity':'','preferences':[],
        'budget':{'action':'keep','amount':None,'currency':'unknown','people':None,'scope':'unknown'},
        'proximity':{'minutes':15,'strict':False}}


def update_context(current, patch):
    validate(patch,CONTEXT_SCHEMA)
    result=copy.deepcopy(current)
    for key in ('city','area','activity','preferences','proximity'):
        if key in patch:
            result[key]=copy.deepcopy(patch[key])
    if 'budget' in patch:
        change=patch['budget']
        budget=result['budget']
        if change.get('people') is not None:
            budget['people']=change['people']
        if change['action']=='clear':
            budget.update(action='clear',amount=None,currency='unknown',scope='unknown')
        else:
            for key in ('amount','currency','people','scope'):
                if key in change and change[key] is not None:
                    budget[key]=change[key]
            budget['action']=change['action']
        if budget['amount'] is not None and not math.isfinite(budget['amount']):
            raise ValueError('non-finite amount')
    return result
