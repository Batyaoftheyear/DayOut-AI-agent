"""Rebuild the final release summary from immutable live reports, without API calls."""
import argparse
import hashlib
import json
from pathlib import Path

from eval_runner import ROOT, load_cases, summary


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def subset(report, rows):
    return summary({'metadata':report['metadata'],'results':rows})


def build(main_path, additional_path, require_complete=False, regression_path=None, final_checks_path=None, controlled_path=None):
    main,extra=read(main_path),read(additional_path)
    if main['metadata']['code_sha256']!=extra['metadata']['code_sha256']:
        raise ValueError('Cannot combine releases with different runtime fingerprints')
    cases={c['id']:c for c in load_cases() if c['mode']=='live'}
    extended={c['id']:c for c in load_cases(ROOT/'eval/extended_cases.json') if c['mode']=='live'}
    expected={(1,key) for key in cases}
    actual={(r['repeat'],r['case_id']) for r in main['results']}
    covered=actual==expected and len(main['results'])==len(expected)
    extra_covered={r['case_id'] for r in extra['results']}==set(extended) and len(extra['results'])==len(extended)
    if require_complete and not (covered and extra_covered):
        raise ValueError('Planned scenario coverage is incomplete')
    sources=[{'file':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in (main_path,additional_path)]
    groups={group:subset(main,[r for r in main['results'] if cases[r['case_id']]['group']==group])
            for group in sorted({c['group'] for c in cases.values()})}
    splits={split:subset(main,[r for r in main['results'] if cases[r['case_id']].get('split','development')==split])
            for split in ('development','holdout')}
    usage={k:main['summary']['total_usage'][k]+extra['summary']['total_usage'][k]
           for k in ('requests','prompt_tokens','completion_tokens','missing_usage')}
    report={'metadata':{'scope':'benchmark_and_targeted_verification','date':'2026-11-02','benchmark_release':'2026-11-02-r10',
            'source_commit':'bab605a','source_tag':'eval-r10-2026-11-02',
            'code_sha256':main['metadata']['code_sha256'],'review_method':'llm_assisted_same_model',
            'sources':sources,'raw_maps_content_saved':False,'main_coverage_complete':covered,
            'additional_coverage_complete':extra_covered,'planned_main_attempts':30,'planned_additional_attempts':len(extended)},
        'main':main['summary'],'additional':extra['summary'],'splits':splits,'groups':groups,
        'cases':[{'id':key,'group':cases[key]['group'],**subset(main,[r for r in main['results'] if r['case_id']==key])}
                 for key in cases],
        'additional_cases':[{'id':r['case_id'],'turns_attempted':len(r['turns']),
             'turns_completed':sum(t['completed'] for t in r['turns']),'execution_status':r['execution_status'],
             'auto_checks':r['auto_checks'],'scores':r['manual_scores'],'review_codes':r.get('model_review',{}).get('codes',[])}
             for r in extra['results']],
        'cost':{'usage':usage,'prices_rub_per_1k':[.1,.2],
            'known_tokens_rub':(usage['prompt_tokens']*.1+usage['completion_tokens']*.2)/1000,
            'complete_usage':not usage['missing_usage'],'not_invoice':True}}
    if regression_path:
        regression=read(regression_path)
        intended={c['id'] for c in load_cases(ROOT/'eval/regression_cases.json')}
        covered={r['case_id'] for r in regression['results']}==intended and len(regression['results'])==len(intended)
        if require_complete and not covered:
            raise ValueError('Targeted verification is not complete')
        report['metadata'].update(implementation_release='2026-11-02-r11',implementation_commit='cd95fb5',
            implementation_tag='eval-r11-2026-11-02',implementation_code_sha256=regression['metadata']['code_sha256'],
            targeted_coverage_complete=covered,quality_rates_are_not_pooled_across_versions=True)
        report['metadata']['sources'].append({'file':regression_path.relative_to(ROOT).as_posix(),
            'sha256':hashlib.sha256(regression_path.read_bytes()).hexdigest()})
        report['regression']=regression['summary']
        report['regression_cases']=[{'id':r['case_id'],'turns_attempted':len(r['turns']),
             'turns_completed':sum(t['completed'] for t in r['turns']),'execution_status':r['execution_status'],
             'auto_checks':r['auto_checks'],'scores':r['manual_scores'],'review_codes':r.get('model_review',{}).get('codes',[])}
             for r in regression['results']]
        for key in usage:
            usage[key]+=regression['summary']['total_usage'][key]
        report['cost'].update(usage=usage,known_tokens_rub=(usage['prompt_tokens']*.1+usage['completion_tokens']*.2)/1000,
                              complete_usage=not usage['missing_usage'])
    if final_checks_path:
        final=read(final_checks_path)
        intended={c['id'] for c in load_cases(ROOT/'eval/final_checks.json')}
        covered={r['case_id'] for r in final['results']}==intended and len(final['results'])==len(intended)
        if require_complete and not covered:
            raise ValueError('Final release checks are not complete')
        report['metadata'].update(implementation_release='2026-11-02-r12',implementation_commit='2e48184',
            implementation_tag='eval-r12-2026-11-02',implementation_code_sha256=final['metadata']['code_sha256'],
            final_checks_coverage_complete=covered,benchmark_code_sha256=main['metadata']['code_sha256'])
        report['metadata']['sources'].append({'file':final_checks_path.relative_to(ROOT).as_posix(),
            'sha256':hashlib.sha256(final_checks_path.read_bytes()).hexdigest()})
        report['final_checks']=final['summary']
        report['final_check_cases']=[{'id':r['case_id'],'turns_attempted':len(r['turns']),
             'turns_completed':sum(t['completed'] for t in r['turns']),'execution_status':r['execution_status'],
             'auto_checks':r['auto_checks'],'scores':r['manual_scores'],'review_codes':r.get('model_review',{}).get('codes',[])}
             for r in final['results']]
        for key in usage:
            usage[key]+=final['summary']['total_usage'][key]
        report['cost'].update(usage=usage,known_tokens_rub=(usage['prompt_tokens']*.1+usage['completion_tokens']*.2)/1000,
                              complete_usage=not usage['missing_usage'])
    if controlled_path and controlled_path.exists():
        controlled=read(controlled_path)
        report['controlled_dialogue']=controlled
        report['metadata']['sources'].append({'file':controlled_path.relative_to(ROOT).as_posix(),
            'sha256':hashlib.sha256(controlled_path.read_bytes()).hexdigest()})
        if controlled.get('summary'):
            for key in usage:
                usage[key]+=controlled['summary']['total_usage'][key]
            report['cost'].update(usage=usage,known_tokens_rub=(usage['prompt_tokens']*.1+usage['completion_tokens']*.2)/1000,
                                  complete_usage=not usage['missing_usage'])
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main',type=Path,default=ROOT/'eval/results/release-r10-2026-11-02.json')
    parser.add_argument('--additional',type=Path,default=ROOT/'eval/results/additional-complete-r10-2026-11-02.json')
    parser.add_argument('--regression',type=Path,default=ROOT/'eval/results/regression-r11-2026-11-02.json')
    parser.add_argument('--final-checks',type=Path,default=ROOT/'eval/results/final-checks-r12-2026-11-02.json')
    parser.add_argument('--controlled',type=Path,default=ROOT/'eval/results/controlled-dialogue-r12-2026-11-02.json')
    parser.add_argument('--output',type=Path,default=ROOT/'eval/results/defense-evaluation-2026-11-02.json')
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    report=build(args.main.resolve(),args.additional.resolve(),args.require_complete,args.regression.resolve(),args.final_checks.resolve(),args.controlled.resolve())
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'main':report['main'],'additional':report['additional'],'cost':report['cost']},indent=2))


if __name__=='__main__':main()
