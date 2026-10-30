"""Run all dialogue turns after graceful tool errors, just as a user can.

No raw answers or Maps data are persisted. Fatal/model errors still stop the run.
"""
import argparse
import copy
import json
import sys
import time
from datetime import date,datetime
from pathlib import Path

from agent import AGENT_RELEASE, MOSCOW_TZ
from eval.reviewer import ModelReviewer
from eval_runner import ROOT,fingerprint,estimate_cost,load_cases,failure_category
from web_backend import WebMemory,RequestGate,load_web_config,run_action


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visit-date',type=date.fromisoformat,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Отчёт уже существует')
    case=next(c for c in load_cases(ROOT/'eval/extended_cases.json') if c['id']=='L01')
    config=load_web_config()
    memory,gate=WebMemory(),RequestGate(cooldown=0)
    report={'metadata':{'scope':'dialogue_stress','release':AGENT_RELEASE,'code_sha256':fingerprint(),
        'started_at':datetime.now(MOSCOW_TZ).isoformat(timespec='seconds'),'review_method':'llm_assisted_same_model',
        'raw_content_saved':False,'prices_rub_per_1k':[.1,.2]},'turns':[]}
    records=[]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save()
    for index,query in enumerate(case['turns'],1):
        print(f'Dialogue {index}/15',flush=True)
        start=time.monotonic()
        result=run_action(config,memory,gate,'ask',query.format(visit_date=str(args.visit_date)))
        tools=[e.removeprefix('MCP → ') for e in result.events if e.startswith('MCP → ')]
        observed=result.reply.observations if result.reply else result.observations
        plan=memory.last_plan or {}
        report['turns'].append({'index':index,'completed':bool(result.reply),'seconds':round(time.monotonic()-start,3),
            'tools':tools,'failure_category':failure_category(result.error),
            'tool_outcomes':observed.get('tool_outcomes',[]),'budget':copy.deepcopy(plan.get('budget')),
            'task':plan.get('task'),'city':plan.get('city'),'area':plan.get('area'),'activity':plan.get('activity')})
        records.append({'query':query.format(visit_date=str(args.visit_date)),'answer':result.reply.text if result.reply else None,
                        'messages':result.evidence_messages,'outcomes':observed.get('tool_outcomes',[])})
        report['agent_usage']=dict(memory.usage_totals)
        save()
        if result.error:
            print('Stopped: '+result.error,flush=True)
            break
        if index<15:time.sleep(2)
    turns=report['turns']
    complete=len(turns)==15 and all(t['completed'] for t in turns)
    def budget_at(index):return turns[index-1].get('budget') or {} if len(turns)>=index else {}
    report['checks']={'all_15_answers':complete,
        'initial_budget_preserved':budget_at(2).get('amount')==3000,
        'updated_budget_preserved':budget_at(7).get('amount')==2000,
        'cancelled_budget_stays_cancelled':len(turns)>=15 and all((t.get('budget') or {}).get('amount') is None for t in turns[8:]),
        'no_tools_for_requested_restatement':all(not turns[i-1]['tools'] for i in (1,3,8,10,13,15) if len(turns)>=i),
        'weather_and_route_used':len(turns)>=12 and 'lookup_weather' in turns[10]['tools'] and 'compute_routes' in turns[11]['tools']}
    review=ModelReviewer(config)(case,records) if complete else {'status':'not_run','usage':{}}
    report['model_review']=review
    total={k:memory.usage_totals[k]+review.get('usage',{}).get(k,0) for k in memory.usage_totals}
    report['summary']={'turn_attempts':len(turns),'turn_completed':sum(t['completed'] for t in turns),
        'graceful_tool_errors':sum(o['status']=='error' for t in turns for o in t['tool_outcomes']),
        'checks_passed':sum(report['checks'].values()),'checks_total':len(report['checks']),
        'total_usage':total,'estimated_rub':estimate_cost(total,[.1,.2])}
    save()
    print(json.dumps({'checks':report['checks'],'summary':report['summary'],'review':review.get('codes')},ensure_ascii=False,indent=2))
    return 0 if complete else 1


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
