import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agent import AgentError, GroqConnectionError, Reply
from dayout import chat


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
