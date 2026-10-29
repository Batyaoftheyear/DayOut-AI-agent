"""Reproducible Eval: aggregate metrics only; live quality requires human review."""

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

from agent import MOSCOW_TZ, ALLOWED_TOOLS, collect_sources, tool_evidence
from eval.simulated import run_simulated
from web_backend import RequestGate, WebMemory, load_web_config, run_action

ROOT = Path(__file__).resolve().parent
CASE_PATH = ROOT / "eval" / "cases.json"
CRITERIA = {"relevance", "constraints", "grounding", "honesty", "tool_choice", "dialogue", "clarification", "controlled_behavior"}
INFRASTRUCTURE_FAILURES = {"quota", "access_denied", "authentication", "billing", "timeout", "network", "tool_error", "tool_request", "adapter_error"}


def load_cases(path=CASE_PATH):
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"]
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Повторяющийся ID сценария")
    for case in cases:
        if case["mode"] not in {"live", "simulated"} or not set(case["criteria"]) <= CRITERIA:
            raise ValueError("Неизвестный режим или критерий")
        if not case["criteria"] or not case.get("expected"):
            raise ValueError("Нужны критерии и ожидаемое поведение")
        if case["mode"] == "live" and not case.get("turns"):
            raise ValueError("Отсутствуют ходы live-сценария")
        if case.get("split", "development") not in {"development", "holdout"}:
            raise ValueError("Неизвестная часть набора")
        for key in ("required_tools", "forbidden_tools", "required_last_turn_tools", "forbidden_last_tools", "required_success_tools"):
            if not set(case.get(key, [])) <= ALLOWED_TOOLS:
                raise ValueError("Неизвестный инструмент в критериях сценария")
    return cases


def fingerprint():
    digest = hashlib.sha256()
    for name in ("app.py", "agent.py", "request_policy.py", "web_backend.py", "dayout.py", "mcp_check.py", "groq_check.py",
                 "provider_config.py", "yandex_client.py", "eval_runner.py", "eval/simulated.py", "requirements.txt"):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def failure_category(error):
    if not error:
        return None
    for marker, category in (("429", "quota"), ("403", "access_denied"), ("401", "authentication"),
                             ("402", "billing"), ("google maps: http 400", "tool_request"),
                             ("сбой запуска адаптера", "adapter_error"),
                             ("квот", "quota"), ("бюджет ожидания", "quota"), ("таймаут", "timeout"),
                             ("сети", "network"), ("метку", "source_validation"), ("источник", "answer_validation")):
        if marker in error.lower():
            return category
    return "other"


def paced_wait(seconds):
    # Keep output responsive; never do a single uninterruptible multi-minute wait.
    remaining = seconds
    while remaining > 0:
        delay = min(remaining, 20)
        print(f"Пауза между ходами: осталось ~{remaining:.0f} с.", flush=True)
        time.sleep(delay)
        remaining -= delay


def weather_parameters_match(requests, case, visit_date):
    mode = case.get("weather_mode")
    if not mode:
        return True
    if not requests:
        return False
    day = date.fromisoformat(visit_date)
    expected_date = {"year": day.year, "month": day.month, "day": day.day}
    for request in requests:
        if mode == "current":
            if request.get("date") is not None or request.get("hour") is not None:
                return False
        elif request.get("date") != expected_date or request.get("hour") != (case.get("weather_hour") if mode == "hourly" else None):
            return False
    return True


def run_live(case, config, gate, visit_date, interval, pace, inspect=False, limits=None):
    memory, turns, all_tools = WebMemory(), [], []
    successful_tools, texts, observations = [], [], []
    stop_reason = None
    for query in case["turns"]:
        if limits:
            stop_reason = limit_reason(limits)
            if stop_reason:
                break
        if pace[0] is not None:
            paced_wait(max(0, interval - (time.monotonic() - pace[0])))
        before = dict(memory.usage_totals)
        start = time.monotonic()
        try:
            result = run_action(config, memory, gate, "ask", query.format(visit_date=visit_date))
        except Exception:
            # Local/setup errors are recorded without raw exceptions or keys.
            from web_backend import WebResult
            result = WebResult(error="Сбой запуска адаптера; сырые детали скрыты.")
        pace[0] = time.monotonic()  # End-to-start spacing, not start-to-start.
        tools = [item.removeprefix("MCP → ") for item in result.events if item.startswith("MCP → ")]
        all_tools.extend(tools)
        usage = {key: memory.usage_totals[key] - value for key, value in before.items()}
        observed = (result.reply.observations or {}) if result.reply else getattr(result, "observations", {})
        outcomes = observed.get("tool_outcomes", [])
        successful_tools.extend(item["tool"] for item in outcomes if item["status"] in {"success", "empty"})
        category = failure_category(result.error)
        if any(item["status"] == "error" for item in outcomes) and category not in INFRASTRUCTURE_FAILURES:
            category = "tool_error"
        if not category and any(item["status"] == "invalid_arguments" for item in outcomes):
            category = "tool_arguments"
        texts.append(result.reply.text if result.reply else "")
        observations.append(observed)
        sources = {}
        for messages, _ in memory.turns:
            for _, payload in tool_evidence(messages):
                sources.update(collect_sources(payload))
        returned_links = sum(url in result.reply.text for url in sources) if result.reply else 0
        turns.append({"completed": result.reply is not None, "seconds": round(time.monotonic() - start, 3),
                      "tools": tools, "usage": usage, "source_link_count": returned_links,
                      "failure_category": category, "partial_available": bool(result.partial),
                      "tool_outcomes": outcomes, "task": observed.get("task"),
                      "weather_requests": observed.get("weather_requests", []),
                      "budget_requested": observed.get("budget_requested")})
        if limits:
            limits["turns"] += 1
            for key, value in usage.items():
                limits["usage"][key] += value
        if inspect:
            # Transient console inspection only; report files contain no answers,
            # URLs, names, summaries, tool payloads, or raw provider errors.
            print(config.redact(result.reply.text if result.reply else result.error or "Нет ответа"))
            for messages, _ in memory.turns:
                for message in messages:
                    if message.get("role") == "assistant":
                        for call in message.get("tool_calls") or []:
                            print(config.redact(json.dumps({"tool_request": call.get("function", {})}, ensure_ascii=False)))
                for name, payload in tool_evidence(messages):
                    print(config.redact(json.dumps({"tool": name, "evidence": payload}, ensure_ascii=False)))
        if result.error or category in INFRASTRUCTURE_FAILURES:
            break  # Preserve the failed first attempt; never replace it with a successful retry.
    last_tools = turns[-1]["tools"] if turns else []
    last_text = texts[-1] if texts else ""
    auto = {
        "all_turns_completed": len(turns) == len(case["turns"]) and all(turn["completed"] for turn in turns),
        "required_tools_called": set(case.get("required_tools", [])) <= set(all_tools),
        "forbidden_tools_absent": not set(case.get("forbidden_tools", [])) & set(all_tools),
        "last_turn_tools_called": set(case.get("required_last_turn_tools", [])) <= set(last_tools),
        "last_turn_forbidden_tools_absent": not set(case.get("forbidden_last_tools", [])) & set(last_tools),
        "required_tools_succeeded": set(case.get("required_success_tools", [])) <= set(successful_tools),
        "required_text_present": all(fragment.lower() in "\n".join(texts).lower() for fragment in case.get("required_text", [])),
        "forbidden_text_absent": all(fragment.lower() not in "\n".join(texts).lower() for fragment in case.get("forbidden_text", [])),
        "required_last_text_present": all(fragment.lower() in last_text.lower() for fragment in case.get("required_last_text", [])),
        "forbidden_last_text_absent": all(fragment.lower() not in last_text.lower() for fragment in case.get("forbidden_last_text", [])),
        "last_source_count_matches": "last_source_links" not in case or bool(turns and turns[-1]["source_link_count"] == case["last_source_links"]),
        "no_invalid_tool_arguments": not any(item["status"] == "invalid_arguments" for observed in observations for item in observed.get("tool_outcomes", [])),
        "weather_parameters_match": weather_parameters_match([request for observed in observations for request in observed.get("weather_requests", [])], case, visit_date),
    }
    infrastructure_failed = any(turn["failure_category"] in INFRASTRUCTURE_FAILURES for turn in turns)
    return {"execution_status": "stopped" if stop_reason else "completed" if auto["all_turns_completed"] and not infrastructure_failed else "failed",
            "stop_reason": stop_reason,
            "auto_checks": auto, "turns": turns,
            "manual_scores": {criterion: "pending" for criterion in case["criteria"]}}


def estimate_cost(usage, prices):
    if prices is None or usage.get("missing_usage", 0):
        return None
    return (usage.get("prompt_tokens", 0) * prices[0] + usage.get("completion_tokens", 0) * prices[1]) / 1000


def limit_reason(limits):
    if limits["turns"] >= limits["max_turns"]:
        return "turn_limit"
    if limits.get("max_rub") is not None:
        cost = estimate_cost(limits["usage"], limits.get("prices"))
        if cost is None:
            return "unknown_usage"
        if cost >= limits["max_rub"]:
            return "estimated_cost_limit"
    return None


def summary(report):
    live = [row for row in report["results"] if row["mode"] == "live"]
    simulated = [row for row in report["results"] if row["mode"] == "simulated"]
    passed = failed = pending = infrastructure = stopped = 0
    for row in live:
        if row["execution_status"] == "stopped":
            stopped += 1
            continue
        if any(turn.get("failure_category") in INFRASTRUCTURE_FAILURES for turn in row["turns"]):
            infrastructure += 1
            continue
        scores = row["manual_scores"].values()
        if not all(row["auto_checks"].values()) or "fail" in scores:
            failed += 1
        elif row["manual_scores"] and all(score == "pass" for score in scores):
            passed += 1
        else:
            pending += 1
    durations = [sum(turn["seconds"] for turn in row["turns"]) for row in live]
    ordered = sorted(durations)
    evaluable = len(live) - infrastructure - stopped
    usage = {key: sum(turn.get("usage", {}).get(key, 0) for row in live for turn in row["turns"])
             for key in ("requests", "prompt_tokens", "completion_tokens", "missing_usage")}
    return {"live_attempts": len(live), "live_completed": sum(row["execution_status"] == "completed" for row in live),
            "live_infrastructure_failures": infrastructure, "live_stopped": stopped, "live_quality_evaluable": evaluable,
            "live_quality_pass": passed, "live_quality_fail": failed, "live_quality_pending": pending,
            "live_quality_pass_rate": passed / evaluable if evaluable and not pending else None,
            "live_end_to_end_pass_rate": passed / (len(live)-stopped) if len(live)>stopped and not pending else None,
            "usage": usage,
            "estimated_rub": estimate_cost(usage, report.get("metadata", {}).get("prices_rub_per_1k")),
            "live_median_seconds": (ordered[(len(ordered)-1)//2] + ordered[len(ordered)//2]) / 2 if ordered else None,
            "live_p95_seconds": ordered[max(0, math.ceil(len(ordered)*.95)-1)] if ordered else None,
            "simulated_attempts": len(simulated),
            "simulated_pass": sum(all(row["controlled_checks"].values()) for row in simulated)}


def write_report(path, report):
    report["summary"] = summary(report)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_review(path, report):
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "repeat", "criterion", "verdict", "note"])
        for row in report["results"]:
            if row["mode"] == "live":
                for criterion, value in row["manual_scores"].items():
                    writer.writerow([row["case_id"], row["repeat"], criterion, value, ""])


def apply_review(report, path):
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    valid = {(r["case_id"], str(r["repeat"]), c) for r in report["results"] if r["mode"] == "live" for c in r["manual_scores"]}
    seen = set()
    for row in rows:
        key = (row["case_id"], row["repeat"], row["criterion"])
        if key not in valid or key in seen or row["verdict"] not in {"pass", "fail", "pending"}:
            raise ValueError("Неизвестная/повторная строка или недопустимая оценка в CSV")
        seen.add(key)
    for result in report["results"]:
        for row in rows:
            if row["case_id"] == result["case_id"] and int(row["repeat"]) == result["repeat"]:
                result["manual_scores"][row["criterion"]] = row["verdict"]


def interactive_review(case, result):
    print("Ожидаемое поведение: " + case["expected"])
    criteria = list(result["manual_scores"])
    print("Критерии по порядку: " + ", ".join(criteria))
    while True:
        value = input("Оценки через пробел: 1=pass, 0=fail, ?=pending; Enter=все pending: ").strip()
        if not value:
            return
        scores = value.split()
        if len(scores) == len(criteria) and all(score in {"1", "0", "?"} for score in scores):
            result["manual_scores"] = dict(zip(criteria, ({"1": "pass", "0": "fail", "?": "pending"}[score] for score in scores)))
            return
        print("Нужно по одной оценке на каждый перечисленный критерий.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--mode", choices=["live", "simulated"])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--visit-date", type=date.fromisoformat)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--interval", type=float, help="Пауза от конца до начала следующего live-хода; по умолчанию Яндекс 5 с, Groq 65 с")
    parser.add_argument("--split", choices=["development", "holdout", "all"], default="development")
    parser.add_argument("--variant", default="candidate", help="Метка версии; идентичность кода проверяется по fingerprint, не по метке")
    parser.add_argument("--max-live-turns", type=int, default=5, help="Остановка между ходами; по умолчанию только небольшой пилот")
    parser.add_argument("--input-rub-per-1k", type=float)
    parser.add_argument("--output-rub-per-1k", type=float)
    parser.add_argument("--max-estimated-rub", type=float, help="Остановка по расчётной стоимости МЕЖДУ ходами; один ход может превысить порог")
    parser.add_argument("--interactive-review", action="store_true", help="Показать ответ/доказательства и записать ручные оценки сразу, без сохранения Maps Content")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspect", action="store_true", help="Показать ответ и доказательства в терминале, без сохранения")
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    if args.summarize:
        report = json.loads(args.summarize.read_text(encoding="utf-8"))
        if args.review:
            apply_review(report, args.review)
        print(json.dumps(summary(report), ensure_ascii=False, indent=2))
        return 0
    cases = load_cases()
    if args.list:
        for case in cases:
            print(f"{case['id']}  {case['mode']:9} {case.get('split','development'):11} {case['title']}")
        return 0
    if not args.mode or not args.output:
        parser.error("Для запуска нужны --mode и --output; список сценариев: --list")
    if args.repeat < 1 or args.max_live_turns < 1 or (args.interval is not None and (not math.isfinite(args.interval) or args.interval < 0)):
        parser.error("Нужны repeat/max-live-turns >= 1 и конечный interval >= 0")
    prices = None
    if args.input_rub_per_1k is not None or args.output_rub_per_1k is not None:
        if any(value is None or not math.isfinite(value) or value < 0 for value in (args.input_rub_per_1k, args.output_rub_per_1k)):
            parser.error("Нужны обе конечные неотрицательные цены")
        prices = [args.input_rub_per_1k, args.output_rub_per_1k]
    if args.max_estimated_rub is not None and (prices is None or not math.isfinite(args.max_estimated_rub) or args.max_estimated_rub <= 0):
        parser.error("Для max-estimated-rub нужны цены и конечный положительный порог")
    selected = [case for case in cases if case["mode"] == args.mode and (not args.case or case["id"] in args.case)
                and (args.split == "all" or case.get("split", "development") == args.split)]
    if not selected or set(args.case) - {case["id"] for case in selected}:
        parser.error("Неизвестный сценарий или ID из другого режима")
    if args.mode == "live" and args.visit_date is None:
        parser.error("Для live явно зафиксируйте --visit-date YYYY-MM-DD")
    config = load_web_config() if args.mode == "live" else None
    if args.interval is None:
        args.interval = 5 if config and config.provider == "yandex" else 65
    review_path = args.output.with_suffix(".review.csv")
    if args.output.exists() or (config and review_path.exists()):
        parser.error("Отчёт или таблица оценок уже существуют; выберите новое имя")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write("{}\n")  # Refuse overwriting an earlier run.
    report = {"metadata": {"started_at": datetime.now(MOSCOW_TZ).isoformat(timespec="seconds"),
              "target": "local_web_adapter" if args.mode == "live" else "offline_fault_injection",
              "provider": config.provider if config else "scripted_mock",
              "model": config.model.replace("gpt://" + config.folder_id + "/", "") if config and config.folder_id else config.model if config else "scripted_mock",
              "variant": args.variant, "split": args.split, "max_live_turns": args.max_live_turns,
              "prices_rub_per_1k": prices, "max_estimated_rub": args.max_estimated_rub,
              "cost_is_estimate_not_invoice": True, "python": platform.python_version(),
              "code_sha256": fingerprint(), "dataset_sha256": hashlib.sha256(CASE_PATH.read_bytes()).hexdigest(),
              "visit_date": str(args.visit_date) if args.visit_date else None, "interval_seconds": args.interval,
              "raw_content_saved": False}, "results": []}
    gate, pace = RequestGate(cooldown=0), [None]
    limits = {"turns": 0, "max_turns": args.max_live_turns, "prices": prices, "max_rub": args.max_estimated_rub,
              "usage": {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "missing_usage": 0}}
    write_report(args.output, report)
    exit_code = 0
    try:
        for repeat in range(1, args.repeat + 1):
            for case in selected:
                if config and (reason := limit_reason(limits)):
                    report["metadata"]["stopped_reason"] = reason
                    break
                print(f"Eval {case['id']} · повтор {repeat}", flush=True)
                start = time.monotonic()
                result = run_live(case, config, gate, str(args.visit_date), args.interval, pace,
                                  args.inspect or args.interactive_review, limits) if config else run_simulated(case["scenario"])
                result.update(case_id=case["id"], mode=case["mode"], repeat=repeat, elapsed_seconds=round(time.monotonic()-start, 3))
                report["results"].append(result)
                write_report(args.output, report)
                if config and args.interactive_review and result["execution_status"] != "stopped":
                    interactive_review(case, result)
                    write_report(args.output, report)
                print(f"  {result['execution_status']} · {result['elapsed_seconds']} с", flush=True)
                if config and any(turn.get("failure_category") in {"quota", "authentication", "access_denied", "billing"} for turn in result["turns"]):
                    report["metadata"]["stopped_reason"] = "provider_access_or_quota"
                    break
            if report["metadata"].get("stopped_reason"):
                break
    except KeyboardInterrupt:
        print("Прогон прерван; завершённые сценарии сохранены.")
        report["metadata"]["stopped_reason"] = "interrupted"
        exit_code = 130
    finally:
        write_report(args.output, report)
        if config:
            write_review(review_path, report)
    if config:
        print("Таблица ручной оценки: " + str(review_path))
    print(json.dumps(summary(report), ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        raise SystemExit(main())
    except Exception:
        print("Eval не завершён. Проверьте настройки, аргументы и путь вывода; существующий отчёт не перезаписывается.", file=sys.stderr)
        raise SystemExit(1) from None
