"""Aggregate the two frozen r6 core reports; do not mix development versions."""
import hashlib
import json
from pathlib import Path

from eval_runner import ROOT, summary


def build():
    directory = ROOT / 'eval/results'
    names = ['release-development-2026-10-30.json', 'release-holdout-2026-10-30.json']
    reports = [json.loads((directory / name).read_text(encoding='utf-8')) for name in names]
    hashes = {r['metadata']['code_sha256'] for r in reports}
    if len(hashes) != 1:
        raise ValueError('Cannot combine different runtime versions')
    rows = [row for report in reports for row in report['results']]
    combined = {'metadata': reports[0]['metadata'], 'results': rows}
    development_names = ['preflight', 'development-r4', 'diagnostics-r4', 'corrections-r5', 'extended-r5']
    all_names = [n + '-2026-10-30.json' for n in development_names] + names + [
        'release-extended-2026-10-30.json', 'dialogue-stress-2026-10-30.json']
    cost_rows = [{'file': n, 'estimated_rub': json.loads((directory / n).read_text(encoding='utf-8'))['summary']['estimated_rub']} for n in all_names]
    result = {'metadata': {'scope': 'partial_core_live_eval', 'date': '2026-10-30',
        'release': '2026-10-30-r6', 'source_commit': '90aef8f', 'source_tag': 'eval-r6-2026-10-30',
        'code_sha256': next(iter(hashes)), 'review_method': 'llm_assisted_same_model',
        'raw_maps_content_saved': False, 'planned_jobs': 90, 'unattempted_jobs': 90-len(rows),
        'unique_core_cases_attempted': len({r['case_id'] for r in rows}),
        'reason_incomplete': 'Google Maps Demo Key daily HTTP 429; continuation deferred by user',
        'sources': [{'file': n, 'sha256': hashlib.sha256((directory/n).read_bytes()).hexdigest()} for n in names]},
        'summary': summary(combined),
        'splits': {r['metadata']['split']: r['summary'] for r in reports},
        'supplementary': json.loads((directory/'release-extended-2026-10-30.json').read_text(encoding='utf-8'))['summary'],
        'stress': json.loads((directory/'dialogue-stress-2026-10-30.json').read_text(encoding='utf-8'))['summary'],
        'all_today_cost': {'reports': cost_rows, 'estimated_rub': sum(x['estimated_rub'] for x in cost_rows),
                           'not_invoice': True, 'prices_rub_per_1k': [.1, .2]},
        'failed_core_jobs': [{'case_id': r['case_id'], 'repeat': r['repeat'],
            'failed_auto_checks': [k for k, v in r['auto_checks'].items() if not v],
            'failed_review_criteria': [k for k, v in r['manual_scores'].items() if v == 'fail'],
            'failure_categories': [t['failure_category'] for t in r['turns'] if t.get('failure_category')]}
            for r in rows if r['execution_status'] != 'completed' or not all(r['auto_checks'].values()) or 'fail' in r['manual_scores'].values()]}
    return result


if __name__ == '__main__':
    report = build()
    target = ROOT / 'eval/results/final-summary-2026-10-30.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'core': report['summary'], 'all_today_estimated_rub': report['all_today_cost']['estimated_rub']}, indent=2))
