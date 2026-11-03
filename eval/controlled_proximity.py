"""Real LLM, fictional Maps: soft expansion and explicit strict limit.

No external Maps calls. Never used by the production app.
"""
import argparse
import hashlib
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from mcp.types import CallToolResult,Tool
import web_backend as web
from agent import MOSCOW_TZ
from eval.reviewer import ModelReviewer
from eval_runner import fingerprint,run_live,summary

POINT={'latitude':34.69,'longitude':135.5}


class FictionalMaps:
    radii=[]
    durations=[]
    def __init__(self,*args,**kwargs):pass
    async def __aenter__(self):return self
    async def __aexit__(self,*args):pass
    async def initialize(self):pass
    async def list_tools(self,cursor=None):
        source=Path(__file__).with_name('fixtures')/'maps-input-schemas-2026-11-03.json'
        schemas=json.loads(source.read_text(encoding='utf-8'))['schemas']
        return SimpleNamespace(nextCursor=None,tools=[Tool(name=name,inputSchema=schema) for name,schema in schemas.items()])
    async def call_tool(self,name,arguments):
        if name=='search_places':
            radius=((arguments.get('locationBias') or {}).get('circle') or {}).get('radiusMeters')
            if radius is not None:self.radii.append(radius)
            kind='anchor' if radius is None else 'far' if radius>1200 else 'near'
            names={'anchor':'Парк Встреча','near':'Раменная Рядом','far':'Раменная Дальше'}
            point=dict(POINT)
            if kind=='near':point['latitude']+=.001
            if kind=='far':point['latitude']+=.01
            payload={'places':[{'id':'fixture-'+kind,'place':'places/fixture-'+kind,'location':point,
                'attribution':{'title':names[kind]+' - Google Maps','url':'https://example.test/proximity-'+kind}}],
                'summary':names[kind]+(' — парк в Осаке, Япония.' if kind=='anchor' else
                    ' — ресторан рамена, Осака. Цены неизвестны. Маршрут проверяется отдельно.')}
        elif name=='compute_routes':
            dest=arguments['destination']
            far=('far' in json.dumps(dest).lower() or 'дальше' in json.dumps(dest,ensure_ascii=False).lower()
                 or (dest.get('latLng') or {}).get('latitude',0)>=34.699)
            seconds=1080 if far else 1200
            self.durations.append(seconds)
            payload={'routes':[{'duration':f'{seconds}s','distanceMeters':1400 if far else 1100,
                'attribution':{'title':'Synthetic route','url':'https://example.test/proximity-route-'+('far' if far else 'near')}}]}
        else:raise AssertionError('Unexpected fixture tool')
        return CallToolResult(content=[],structuredContent=payload)


@asynccontextmanager
async def local_transport(*args,**kwargs):yield None,None,None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new result path.')
    cases=[
        {'id':'SOFT','mode':'live','turns':['Найди рамен рядом с парком Встреча в Осаке. Начни с ориентировочных 15 минут пешком; если не найдётся, ищи немного дальше и явно сообщи время.'],
         'expected':'Начальный кандидат: 20 минут. Нужно расширить поиск за 1200 м и предложить кандидата Дальше с подтверждёнными 18 минутами. Расширение явно обозначено, нет обещания предела 15 минут.',
         'criteria':['relevance','constraints','grounding','honesty','tool_choice'],'required_tools':['search_places','compute_routes']},
        {'id':'STRICT','mode':'live','turns':['Найди рамен рядом с парком Встреча в Осаке. Пожалуйста, только до 15 минут пешком. Больший путь не подходит, другие варианты не предлагай.'],
         'expected':'Кандидат с 20 минутами не назван подходящим. Строгий предел 15 минут сохранён, расширение вопреки ему не выполнено.',
         'criteria':['relevance','constraints','grounding','honesty','tool_choice'],'required_tools':['search_places','compute_routes']}]
    report={'metadata':{'scope':'real_llm_fictional_proximity','started_at':datetime.now(MOSCOW_TZ).isoformat(),
        'code_sha256':fingerprint(),'fixture_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'external_maps_requests':0,'raw_content_saved':False,'prices_rub_per_1k':[.1,.2],
        'review_method':'llm_assisted_same_model'},'results':[]}
    config=web.load_web_config()
    reviewer=ModelReviewer(config)
    with patch.object(web,'ClientSession',FictionalMaps),patch.object(web,'streamable_http_client',local_transport):
        for case in cases:
            FictionalMaps.radii=[];FictionalMaps.durations=[]
            start=time.monotonic()
            row=run_live(case,config,web.RequestGate(cooldown=0),'2026-11-04',0,[None],reviewer=reviewer)
            expanded=any(r>1200 for r in FictionalMaps.radii)
            row['auto_checks']['expansion_policy']=expanded if case['id']=='SOFT' else not expanded
            row.update(case_id=case['id'],mode='live',repeat=1,elapsed_seconds=round(time.monotonic()-start,3),
                fixture_search_radii=list(FictionalMaps.radii),fixture_route_seconds=list(FictionalMaps.durations))
            report['results'].append(row);report['summary']=summary(report)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print(json.dumps({'case':case['id'],'execution':row['execution_status'],'expanded':expanded,
                'review':row.get('model_review',{}).get('codes',[])}),flush=True)
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':main()
