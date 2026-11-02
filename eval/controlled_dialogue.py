"""Paid real-LLM dialogue Eval with fictional, deterministic Maps fixtures.

This measures dialogue state and tool use, NOT real Maps availability or venue
quality. It never replaces the production MCP integration.
"""
import argparse
import json
import sys
import time
from contextlib import asynccontextmanager
from datetime import date,datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mcp.types import CallToolResult,Tool
import web_backend as web
from agent import MOSCOW_TZ
from eval.reviewer import ModelReviewer
from eval_runner import ROOT,fingerprint,load_cases,run_live,summary


class FictionalMaps:
    calls=[]
    def __init__(self,*args,**kwargs):pass
    async def __aenter__(self):return self
    async def __aexit__(self,*args):pass
    async def initialize(self):pass
    async def list_tools(self,cursor=None):
        schemas={
            'search_places':{'textQuery':{'type':'string'},'languageCode':{'type':'string'},'pageSize':{'type':'integer'},
                'regionCode':{'type':'string'},'includeExtendedDetails':{'type':'boolean'},'locationBias':{'type':'object'}},
            'lookup_weather':{'location':{'type':'object'},'date':{'type':'object'},'hour':{'type':'integer'},'unitsSystem':{'type':'string'}},
            'compute_routes':{'origin':{'type':'object'},'destination':{'type':'object'},'travelMode':{'enum':['WALK','DRIVE']}}}
        return SimpleNamespace(nextCursor=None,tools=[Tool(name=name,inputSchema={'type':'object','properties':properties,
            'required':{'search_places':['textQuery'],'lookup_weather':['location'],'compute_routes':['origin','destination','travelMode']}[name]})
            for name,properties in schemas.items()])
    async def call_tool(self,name,arguments):
        self.calls.append(name)
        n=len(self.calls)
        if name=='search_places':
            query=arguments.get('textQuery','').lower()
            category='ресторан' if 'ресторан' in query else 'кафе'
            area='Нагатино' if 'нагат' in query else 'юг Москвы' if 'юг' in query else 'центр Москвы'
            places=[];paragraphs=[]
            for i in (1,2):
                url=f'https://example.test/fixture-place-{n}-{i}'
                title=f'Учебный {category} {n}-{i}'
                places.append({'id':f'fixture-{n}-{i}','place':f'places/fixture-{n}-{i}',
                    'location':{'latitude':55.68+i*.0001,'longitude':37.66},'attribution':{'title':title,'url':url}})
                paragraphs.append(f'**{title}** [{i-1}] — вымышленное место для проверки: {category}, {area}, Москва. В учебном описании указаны тихий зал и столики для двоих. Цены неизвестны.')
            payload={'places':places,'summary':'\n\n'.join(paragraphs)}
        elif name=='lookup_weather':
            payload={'summary':'Учебный дневной прогноз: минимум +3 °C, максимум +8 °C, облачно. Данные искусственные.',
                     'date':arguments.get('date'),'attribution':{'title':'Synthetic weather fixture','url':'https://example.test/weather'}}
        elif name=='compute_routes':
            payload={'routes':[{'distanceMeters':1495,'duration':'1231s',
                'attribution':{'title':'Synthetic route fixture','url':'https://example.test/route'}}]}
        else:
            raise ValueError('Tool outside controlled dialogue fixture')
        return CallToolResult(content=[],structuredContent=payload)


@asynccontextmanager
async def fictional_transport(*args,**kwargs):
    yield None,None,None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visit-date',type=date.fromisoformat,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output filename')
    config=web.load_web_config()
    case=next(c for c in load_cases(ROOT/'eval/extended_cases.json') if c['id']=='L01')
    print('REAL paid Yandex LLM; FICTIONAL Maps fixtures; zero external Maps requests.',flush=True)
    report={'metadata':{'started_at':datetime.now(MOSCOW_TZ).isoformat(timespec='seconds'),
        'scope':'real_llm_synthetic_maps_dialogue','release':'2026-11-02-r12','code_sha256':fingerprint(),
        'maps_data':'fictional_fixtures','external_maps_requests':0,'raw_maps_content_saved':False,
        'provider':'yandex','review_method':'llm_assisted_same_model','prices_rub_per_1k':[.1,.2],
        'visit_date':str(args.visit_date),'not_a_live_maps_integration_test':True},'results':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    FictionalMaps.calls=[]
    start=time.monotonic()
    with patch.object(web,'ClientSession',FictionalMaps),patch.object(web,'streamable_http_client',fictional_transport):
        result=run_live(case,config,web.RequestGate(cooldown=0),str(args.visit_date),2,[None],reviewer=ModelReviewer(config))
    result.update(case_id='L01',mode='live',repeat=1,elapsed_seconds=round(time.monotonic()-start,3))
    report['results'].append(result)
    report['fixture_calls']=FictionalMaps.calls
    report['summary']=summary(report)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'summary':report['summary'],'checks':result['auto_checks'],'scores':result['manual_scores']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
