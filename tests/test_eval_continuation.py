import pytest

from eval.continue_run import pending_jobs


def test_continuation_preserves_failed_attempt_and_repeat_number():
    cases = [{'id': 'A', 'mode': 'live', 'split': 'development'},
             {'id': 'B', 'mode': 'live', 'split': 'development'},
             {'id': 'H', 'mode': 'live', 'split': 'holdout'}]
    rows = [{'case_id': 'A', 'mode': 'live', 'repeat': 1, 'execution_status': 'failed'}]
    assert [(r, c['id']) for r, c in pending_jobs(cases, rows, 2, 'development')] == [(1, 'B'), (2, 'A'), (2, 'B')]


def test_duplicate_parent_jobs_cannot_inflate_results():
    row = {'case_id': 'A', 'mode': 'live', 'repeat': 1}
    with pytest.raises(ValueError, match='Duplicate'):
        pending_jobs([], [row, row], 2, 'all')
