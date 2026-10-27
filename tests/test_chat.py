import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from agent import AgentError, GroqConnectionError, QuotaPauseError, Reply
from dayout import chat, is_transcript_line
from groq_check import error_message


def run_chat(lines, failures=None, check_failure=None):
    incoming = iter(lines)
    async def read_line():
        try:
            return next(incoming)
        except StopIteration:
            raise EOFError
    agent = SimpleNamespace(ask=AsyncMock(side_effect=failures or [Reply("Готово", [], 0)]), reset=Mock())
    llm = SimpleNamespace(check_access=AsyncMock(side_effect=check_failure))
    output = []
    asyncio.run(chat(agent, llm, [], output.append, read_line))
    return agent, llm, output


def test_empty_lines_and_log_lines_are_never_sent_to_model():
    agent, _, output = run_chat([
        "", "  ", "DayOut подключён. Модель: test", "Вы: скопированный запрос",
        "MCP → search_places", "Groq HTTP 403: forbidden", "Groq: ожидание",
        "Запрос остановлен через 1 с", "Вызовов инструментов: 1", "Настоящий вопрос", "/exit",
    ])
    agent.ask.assert_awaited_once_with("Настоящий вопрос")
    assert any("строку лога" in item for item in output)


def test_network_failure_pauses_new_queries_until_successful_manual_check():
    agent, llm, output = run_chat([
        "Первый вопрос", "случайная вставка", "/retry", "/check", "/retry", "/exit",
    ], failures=[GroqConnectionError("Ошибка сети Groq (ConnectError)"), Reply("Ответ", [], 1)])
    assert [call.args[0] for call in agent.ask.await_args_list] == ["Первый вопрос", "Первый вопрос"]
    assert agent.ask.await_args_list[0].kwargs == {}
    assert agent.ask.await_args_list[1].kwargs == {"resume": True}
    llm.check_access.assert_awaited_once()
    assert any("приостановлена" in item for item in output)


def test_failed_access_check_keeps_chat_paused():
    agent, llm, _ = run_chat([
        "Запрос", "/check", "/retry", "Другой запрос", "/exit",
    ], failures=[GroqConnectionError("Groq HTTP 403")], check_failure=GroqConnectionError("Groq HTTP 403"))
    agent.ask.assert_awaited_once_with("Запрос")
    llm.check_access.assert_awaited_once()


def test_reset_clears_retry_but_does_not_remove_network_pause():
    agent, _, output = run_chat([
        "Запрос", "/reset", "Другой запрос", "/exit",
    ], failures=[GroqConnectionError("Groq HTTP 403")])
    agent.ask.assert_awaited_once()
    agent.reset.assert_called_once()
    assert any("/check" in item for item in output)


def test_retry_is_explicit_for_non_network_failure():
    agent, _, _ = run_chat(["Запрос", "/retry", "/retry", "/exit"], failures=[
        AgentError("Лимит шага"), Reply("Ответ", [], 1),
    ])
    assert [call.args[0] for call in agent.ask.await_args_list] == ["Запрос", "Запрос"]


def test_batch_error_does_not_silently_report_success():
    agent = SimpleNamespace(ask=AsyncMock(side_effect=GroqConnectionError("Groq HTTP 403")))
    with pytest.raises(GroqConnectionError):
        asyncio.run(chat(agent, None, ["Запрос"], lambda _: None))


def test_quota_pause_allows_manual_retry_without_connectivity_check():
    agent, llm, _ = run_chat(["Запрос", "/retry", "/exit"], failures=[
        QuotaPauseError("Ожидание квоты"), Reply("Ответ", [], 1),
    ])
    assert agent.ask.await_count == 2
    assert agent.ask.await_args.kwargs == {"resume": True}
    llm.check_access.assert_not_awaited()


def test_quota_during_check_does_not_report_network_failure_or_remove_pause():
    agent, llm, output = run_chat(["Запрос", "/check", "/retry", "/exit"], failures=[
        GroqConnectionError("ConnectError"),
    ], check_failure=QuotaPauseError("Ожидание квоты"))
    agent.ask.assert_awaited_once()
    assert any("не завершена из-за квоты" in text for text in output)


def test_multiline_transcript_is_recognized():
    assert is_transcript_line("мой вопрос\nMCP → search_places\nDayOut: ответ")
    assert not is_transcript_line("Найди место без кафе")


@pytest.mark.parametrize("error,label", [
    (httpx.ConnectError("secret-key"), "ConnectError"),
    (httpx.ProxyError("secret-key"), "ProxyError"),
    (httpx.ReadError("secret-key"), "ReadError"),
    (httpx.ReadTimeout("secret-key"), "ReadTimeout"),
    (httpx.RemoteProtocolError("secret-key"), "RemoteProtocolError"),
])
def test_network_diagnostics_do_not_expose_exception_body(error, label):
    message = error_message(error)
    assert label in message
    assert "secret-key" not in message


def test_html_403_reports_possible_network_filter_without_printing_body():
    response = httpx.Response(403, headers={"content-type": "text/html"},
                              text="secret-response", request=httpx.Request("GET", "https://api.groq.com"))
    error = httpx.HTTPStatusError("secret-key", request=response.request, response=response)
    message = error_message(error)
    assert "HTML" in message
    assert "secret" not in message
