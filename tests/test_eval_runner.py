import copy
import csv
import json

import pytest

import eval_runner as evaluation
from eval.simulated import run_simulated


def live_row(scores=None):
    return {"case_id": "B01", "repeat": 1, "mode": "live", "execution_status": "completed",
            "auto_checks": {"completed": True, "required_tools": True},
            "manual_scores": scores if scores is not None else {"grounding": "pending", "honesty": "pending"},
            "turns": [{"seconds": 2.0}]}


def test_dataset_is_fixed_unique_and_splits_quality_from_fault_injection():
    cases = evaluation.load_cases()
    assert len(cases) == 34
    assert sum(case["mode"] == "live" for case in cases) == 30
    assert sum(case["mode"] == "simulated" for case in cases) == 4
    assert all(case["expected"] and case["criteria"] for case in cases)
    assert len(next(case for case in cases if case["id"] == "D01")["turns"]) == 2
    live = [case for case in cases if case["mode"] == "live"]
    assert sum(case["split"] == "holdout" for case in live) == 10
    covered = {tool for case in live for tool in case["required_tools"]}
    assert covered == {"search_places", "lookup_weather", "compute_routes", "resolve_names", "resolve_maps_urls"}


@pytest.mark.parametrize("scores", [{"grounding": "pending"}, {}])
def test_completed_execution_does_not_mean_quality_pass(scores):
    result = evaluation.summary({"results": [live_row(scores)]})
    assert result["live_completed"] == 1
    assert result["live_quality_pass"] == 0 and result["live_quality_pending"] == 1
    assert result["live_quality_pass_rate"] is None


def test_tool_failure_is_not_hidden_by_positive_manual_review():
    row = live_row({"grounding": "pass", "honesty": "pass"})
    row["auto_checks"]["required_tools"] = False
    assert evaluation.summary({"results": [row]})["live_quality_fail"] == 1


def test_simulated_passes_are_not_counted_in_live_quality_percentage():
    result = evaluation.summary({"results": [live_row(), {"mode": "simulated", "controlled_checks": {"safe": True}}]})
    assert result["simulated_pass"] == 1
    assert result["live_quality_pass_rate"] is None


@pytest.mark.parametrize("scenario", ["empty_results", "connection_error", "access_denied", "rate_limit"])
def test_controlled_fault_scenarios_are_offline_and_recover_when_expected(scenario):
    result = run_simulated(scenario)
    assert result["llm"] == "scripted_mock" and result["external_requests"] == 0
    assert result["mcp_calls"] == 1
    assert all(result["controlled_checks"].values())


def test_review_rejects_duplicate_rows_instead_of_silently_changing_grade(tmp_path):
    report = {"results": [live_row()]}
    path = tmp_path / "review.csv"
    evaluation.write_review(path, report)
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(["B01", 1, "grounding", "pass", ""])
    with pytest.raises(ValueError):
        evaluation.apply_review(report, path)
    assert report["results"][0]["manual_scores"]["grounding"] == "pending"


def test_grading_csv_is_applied_without_storing_freeform_notes_in_report(tmp_path):
    report = {"results": [live_row()]}
    path = tmp_path / "review.csv"
    path.write_text("case_id,repeat,criterion,verdict,note\nB01,1,grounding,pass,synthetic private note\nB01,1,honesty,pass,\n", encoding="utf-8")
    evaluation.apply_review(report, path)
    assert evaluation.summary(report)["live_quality_pass_rate"] == 1
    assert "synthetic private note" not in json.dumps(report)


def test_report_contains_only_aggregate_observations(tmp_path):
    path = tmp_path / "result.json"
    report = {"metadata": {"raw_content_saved": False}, "results": [live_row()]}
    evaluation.write_report(path, report)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["summary"]["live_quality_pass_rate"] is None
    assert "http" not in path.read_text(encoding="utf-8")


def test_infrastructure_failure_is_not_hidden_or_graded_as_model_quality():
    row = live_row({"grounding": "pending"})
    row["execution_status"] = "failed"
    row["auto_checks"]["completed"] = False
    row["turns"][0]["failure_category"] = "quota"
    result = evaluation.summary({"results": [row]})
    assert result["live_infrastructure_failures"] == 1 and result["live_quality_fail"] == 0
    assert result["live_quality_evaluable"] == 0 and result["live_quality_pass_rate"] is None
    assert result["live_end_to_end_pass_rate"] == 0


def test_stopped_pilot_is_not_reported_as_a_full_failed_quality_run():
    row = live_row()
    row["execution_status"] = "stopped"
    result = evaluation.summary({"results": [row]})
    assert result["live_stopped"] == 1 and result["live_quality_fail"] == 0
    assert result["live_end_to_end_pass_rate"] is None


def test_cost_limits_use_usage_and_stop_when_the_cost_is_unknown():
    limits = {"turns": 1, "max_turns": 5, "max_rub": 1, "prices": [.1, .2],
              "usage": {"prompt_tokens": 10000, "completion_tokens": 100, "missing_usage": 0}}
    assert evaluation.estimate_cost(limits["usage"], limits["prices"]) == pytest.approx(1.02)
    assert evaluation.limit_reason(limits) == "estimated_cost_limit"
    limits["usage"]["missing_usage"] = 1
    assert evaluation.estimate_cost(limits["usage"], limits["prices"]) is None
    assert evaluation.limit_reason(limits) == "unknown_usage"


def test_final_turn_and_source_checks_use_the_current_answer_not_old_history(monkeypatch):
    from types import SimpleNamespace
    from agent import Reply
    from web_backend import WebResult
    replies = iter([WebResult(reply=Reply("Synthetic fixture", ["search_places"], .1)),
                    WebResult(reply=Reply("MCP is a protocol.", [], .1))])
    monkeypatch.setattr(evaluation, "run_action", lambda *args, **kwargs: next(replies))
    case = {"turns": ["Find a fixture", "Explain MCP"], "required_tools": [],
            "forbidden_last_tools": ["search_places"], "last_source_links": 0, "criteria": ["dialogue"]}
    result = evaluation.run_live(case, SimpleNamespace(), None, "2026-10-30", 0, [None])
    assert all(result["auto_checks"].values())
    assert result["turns"][-1]["source_link_count"] == 0
