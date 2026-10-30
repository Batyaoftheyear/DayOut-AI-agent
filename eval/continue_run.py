"""Continue only unattempted jobs from a frozen live Eval, preserving failures.

The new report includes its parent rows. Do not add its cost to its parent's.
Use --dry-run to inspect the queue without contacting any API.
"""
import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from datetime import date, datetime
from pathlib import Path

from agent import MOSCOW_TZ
from eval.reviewer import ModelReviewer
from eval_runner import (CASE_PATH, fingerprint, limit_reason, load_cases,
                         run_live, summary, write_report, write_review)
from web_backend import RequestGate, load_web_config


def pending_jobs(cases, results, repeats, split):
    selected = [c for c in cases if c['mode'] == 'live'
                and (split == 'all' or c.get('split', 'development') == split)]
    done = {(r['repeat'], r['case_id']) for r in results if r['mode'] == 'live'}
    if len(done) != sum(r['mode'] == 'live' for r in results):
        raise ValueError('Duplicate jobs in parent report')
    return [(repeat, c) for repeat in range(1, repeats + 1) for c in selected
            if (repeat, c['id']) not in done]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cases', type=Path, default=CASE_PATH)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--visit-date', type=date.fromisoformat)
    parser.add_argument('--max-new-turns', type=int, default=80)
    parser.add_argument('--max-new-rub', type=float, default=100)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    parent = json.loads(args.parent.read_text(encoding='utf-8'))
    meta = parent['metadata']
    if args.repeat < 1 or args.max_new_turns < 1 or not 0 < args.max_new_rub <= 1000:
        parser.error('Invalid repeat, turn or cost limit')
    if meta['code_sha256'] != fingerprint():
        parser.error('Runtime fingerprint changed: create a fresh Eval instead')
    if meta['dataset_sha256'] != hashlib.sha256(args.cases.read_bytes()).hexdigest():
        parser.error('Dataset changed: create a fresh Eval instead')
    if meta.get('review_method') != 'llm_assisted_same_model':
        parser.error('This continuation requires the original LLM review method')
    jobs = pending_jobs(load_cases(args.cases), parent['results'], args.repeat, meta['split'])
    print(json.dumps({'new_jobs': len(jobs), 'new_turns': sum(len(c['turns']) for _, c in jobs),
                      'jobs': [[r, c['id']] for r, c in jobs]}, ensure_ascii=False), flush=True)
    if args.dry_run or not jobs:
        return 0
    if not args.output or not args.visit_date:
        parser.error('Live continuation needs --output and current --visit-date')
    if args.output.exists() or args.output.with_suffix('.review.csv').exists():
        parser.error('Output already exists; choose a new filename')
    config = load_web_config()
    if config.provider != meta['provider'] or config.model.split('/')[-2:] != meta['model'].split('/')[-2:]:
        parser.error('Provider/model differs from parent')
    report = copy.deepcopy(parent)
    report['metadata'] = {**meta, 'started_at': datetime.now(MOSCOW_TZ).isoformat(timespec='seconds'),
        'python': platform.python_version(), 'visit_date': str(args.visit_date),
        'parent_report': args.parent.name,
        'parent_sha256': hashlib.sha256(args.parent.read_bytes()).hexdigest(),
        'inherited_rows': len(parent['results']), 'planned_repeats': args.repeat,
        'parent_visit_date': meta.get('visit_date'), 'continuation_dates_may_differ': True,
        'cost_includes_parent': True, 'new_turn_limit': args.max_new_turns,
        'new_cost_limit_rub': args.max_new_rub}
    report['metadata'].pop('stopped_reason', None)
    prices = meta['prices_rub_per_1k']
    limits = {'turns': 0, 'max_turns': args.max_new_turns, 'prices': prices,
              'max_rub': args.max_new_rub,
              'usage': dict.fromkeys(('requests', 'prompt_tokens', 'completion_tokens', 'missing_usage'), 0)}
    gate, pace = RequestGate(cooldown=0), [None]
    reviewer = ModelReviewer(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write('{}\n')
    write_report(args.output, report)
    try:
        for repeat, case in jobs:
            if reason := limit_reason(limits):
                report['metadata']['stopped_reason'] = reason
                break
            print(f"Continue {case['id']} repeat {repeat}", flush=True)
            start = time.monotonic()
            row = run_live(case, config, gate, str(args.visit_date), meta['interval_seconds'],
                           pace, limits=limits, reviewer=reviewer)
            row.update(case_id=case['id'], mode='live', repeat=repeat,
                       visit_date=str(args.visit_date), elapsed_seconds=round(time.monotonic()-start, 3))
            report['results'].append(row)
            write_report(args.output, report)
            print(row['execution_status'], flush=True)
            if any(t.get('failure_category') in {'quota', 'authentication', 'access_denied', 'billing'} for t in row['turns']):
                report['metadata']['stopped_reason'] = 'provider_access_or_quota'
                break
    except KeyboardInterrupt:
        report['metadata']['stopped_reason'] = 'interrupted'
    finally:
        report['metadata']['new_usage'] = limits['usage']
        write_report(args.output, report)
        write_review(args.output.with_suffix('.review.csv'), report)
    print(json.dumps(summary(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
