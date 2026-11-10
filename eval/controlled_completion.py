"""Real LLM, fictional evidence: missing properties and incomplete drafts.

No external Maps calls; never a production fallback.
"""
import argparse,asyncio,hashlib,json
from pathlib import Path
from datetime import datetime
import httpx
from agent import DayOutAgent,TurnState,MOSCOW_TZ
from conversation_context import empty_context
from eval_runner import fingerprint
from web_backend import load_web_config
from yandex_client import make_client


async def run(output):
    config=load_web_config()
    cases=[
        ('price_missing','Бюджет 5000 рублей на двоих. Проверь цены выбранного лаунджа.',
         'Стоимость неизвестна.',[],
         {'summary':'Синтетический лаундж. Цены в этой общей выдаче не запрашивались.'}),
        ('pavilions_missing','Подбери конкретные павильоны на ВДНХ и азиатский ресторан.',
         'Погуляйте у синтетической арки [S1], затем азиатский ресторан [S2].',[],
         {'summary':'Синтетическая арка [0] — архитектурный объект. Синтетический ресторан [1] — азиатская кухня.'}),
        ('cuisine_followup','Так и? Дай конкретный сценарий.',
         'Какую кухню вы предпочитаете?',
         ['Найди конкретные павильоны и рестораны на ВДНХ, азиатская кухня.'],
         {'summary':'Общая выдача; конкретные павильоны и рестораны ещё не проверены.'})
    ]
    report={'metadata':{'scope':'real_llm_fictional_completion_review','code_sha256':fingerprint(),
        'fixture_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'started_at':datetime.now(MOSCOW_TZ).isoformat(),'external_maps_requests':0,'raw_maps_content_saved':False},'results':[]}
    async with httpx.AsyncClient(headers=config.headers,timeout=60) as client:
        llm=make_client(client,config)
        for ident,query,draft,previous,payload in cases:
            payload=dict(payload,places=[{'id':f'fixture-{i}','location':{'latitude':55.8+i*.001,'longitude':37.6},
                'attribution':{'title':'Synthetic fixture','url':f'https://example.test/review-{i}'}} for i in range(2)])
            messages=[{'role':'user','content':q} for q in previous]+[{'role':'user','content':query},
                {'role':'assistant','content':None,'tool_calls':[{'id':'evidence','type':'function','function':{
                    'name':'search_places','arguments':'{"textQuery":"synthetic general search"}'}}]},
                {'role':'tool','tool_call_id':'evidence','content':json.dumps(payload,ensure_ascii=False)}]
            sources={f'https://example.test/review-{i}':'Synthetic fixture' for i in range(2)}
            state=TurnState(query,messages,0,sources,[],{url:f'[S{i+1}]' for i,url in enumerate(sources)})
            state.proposed_text=draft
            a=DayOutAgent(None,llm,{})
            result=json.loads(await a.review_draft(state))
            review=result['response_check']
            row={'case':ident,'action':review['action'],'missing_count':len(review['missing']),
                 'continues_incomplete_draft':review['action']=='continue' and bool(review['missing'])}
            report['results'].append(row)
            report['usage']=dict(llm.usage_totals)
            output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            print(json.dumps(row),flush=True)
    return 0 if all(r['continues_incomplete_draft'] for r in report['results']) else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new result path.')
    raise SystemExit(asyncio.run(run(args.output)))


if __name__=='__main__':main()
