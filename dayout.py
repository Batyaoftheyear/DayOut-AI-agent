"""DayOut console chat: Groq chooses tools, Python executes Google Maps MCP."""

import argparse
import asyncio
import logging
import sys
from datetime import timedelta
from time import monotonic

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agent import AgentError, DayOutAgent, GroqClient, GroqConnectionError, QuotaPauseError, SUPPORTED_MODELS, discover_tools
from groq_check import load_config, error_message as groq_error_message
from mcp_check import MCP_URL, load_key, error_message as mcp_error_message


def is_transcript_line(text: str) -> bool:
    prefixes = ("Вы:", "DayOut:", "DayOut подключён", "MCP →", "Groq:", "Groq HTTP ",
                "Запрос остановлен", "Вызовов инструментов:", "Ошибка сети", "Traceback ", "PS C:\\")
    return any(line.strip().startswith(prefixes) for line in text.splitlines())


async def read_query():
    return await asyncio.to_thread(input, "\nВы: ")


async def chat(agent, llm, queries, emit, read_line=None):
    pending = iter(queries) if queries else None
    read_line = read_line or read_query
    paused, failed_query = False, None
    emit("Команды: /check — проверка Groq, /retry — повтор неудачного запроса, /reset — очистка, /exit — выход.")
    while True:
        if pending is not None:
            query = next(pending, None)
            if query is None:
                break
            emit("Вы: " + query)
        else:
            try:
                query = await read_line()
            except EOFError:
                break
        query = query.strip()
        if not query:
            continue
        command = query.lower()
        if command in {"/exit", "/quit"}:
            break
        if command == "/reset":
            agent.reset()
            failed_query = None
            emit("Диалог очищен.")
            continue
        if command == "/check":
            try:
                await llm.check_access()
            except QuotaPauseError as error:
                emit(str(error))
                emit("Проверка Groq не завершена из-за квоты. Подождите, затем повторите /check.")
            except AgentError as error:
                paused = True
                emit(str(error))
                emit("Отправка запросов приостановлена. Проверьте сеть/доступ и снова введите /check.")
            else:
                paused = False
                emit("Groq: список моделей и генерация доступны. Можно ввести запрос или /retry.")
            continue
        if is_transcript_line(query):
            emit("Похоже на строку лога. Введите только свой вопрос, без «Вы:», «DayOut:» и сообщений программы.")
            continue
        if command.startswith("/") and command != "/retry":
            emit("Неизвестная команда. Доступны /check, /retry, /reset, /exit.")
            continue
        if paused:
            emit("Отправка запросов приостановлена после сбоя Groq. Введите /check для проверки связи.")
            continue
        if command == "/retry":
            if failed_query is None:
                emit("Нет неудачного запроса для повтора.")
                continue
            query = failed_query
            emit("Повторяю последний неудачный запрос.")
        start = monotonic()
        try:
            reply = await agent.ask(query, resume=True) if command == "/retry" else await agent.ask(query)
        except AgentError as error:
            failed_query = query
            if pending is not None:
                raise
            emit(str(error))
            emit(f"Запрос остановлен через {monotonic() - start:.1f} с; диалог сохранён до этого запроса.")
            if isinstance(error, GroqConnectionError):
                paused = True
                emit("Отправка приостановлена. После проверки сети введите /check, затем /retry.")
            continue
        failed_query = None
        emit("\nDayOut: " + reply.text)
        emit(f"\nВызовов инструментов: {len(reply.tools)}; время: {reply.seconds:.1f} с.")
        if reply.usage:
            usage = reply.usage
            if usage["missing_usage"] == usage["requests"]:
                emit(f"LLM: запросов {usage['requests']}; API не предоставил данные о расходе токенов.")
            else:
                suffix = f"; неполный учёт, без usage: {usage['missing_usage']}" if usage["missing_usage"] else ""
                emit(f"LLM: запросов {usage['requests']}; токены API — вход {usage['prompt_tokens']}, "
                     f"выход {usage['completion_tokens']}{suffix}.")


async def run(groq_key: str, maps_key: str, model: str, queries: list[str], emit, show_usage=False):
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {groq_key}"}, timeout=45) as groq:
        try:
            response = await groq.get("https://api.groq.com/openai/v1/models")
            response.raise_for_status()
            if model not in {item["id"] for item in response.json()["data"]}:
                raise AgentError("GROQ_MODEL не найдена в API. Проверьте groq_check.py --list-models.")
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            raise AgentError(groq_error_message(error)) from None
        async with httpx.AsyncClient(headers={"X-Goog-Api-Key": maps_key}, timeout=45) as maps:
            async with streamable_http_client(MCP_URL, http_client=maps) as (read, write, _):
                async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=45)) as session:
                    async with asyncio.timeout(60):
                        await session.initialize()
                        tools = await discover_tools(session)
                    emit("DayOut подключён. Модель: " + model)
                    llm = GroqClient(groq, model, emit=emit, show_usage=show_usage)
                    agent = DayOutAgent(session, llm, tools, emit=emit)
                    await chat(agent, llm, queries, emit)


def safe_error(error: BaseException) -> str:
    if isinstance(error, BaseExceptionGroup):
        return " / ".join(dict.fromkeys(safe_error(item) for item in error.exceptions))
    if isinstance(error, AgentError):
        return str(error)
    return "Google Maps: " + mcp_error_message(error)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", help="Один запрос; повторите параметр для уточнения в том же диалоге")
    parser.add_argument("--usage", action="store_true", help="Показывать токены каждого запроса LLM, дополнительно к итогу хода")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    secrets = []
    def redact(text):
        for key in secrets:
            text = text.replace(key, "[REDACTED]")
        return text
    try:
        groq_key, model = load_config()
        maps_key = load_key()
        secrets = [groq_key, maps_key]
        if model not in SUPPORTED_MODELS:
            raise AgentError("Неподдерживаемая GROQ_MODEL. Допустимы: " + ", ".join(sorted(SUPPORTED_MODELS)))
        asyncio.run(run(groq_key, maps_key, model, args.query or [], lambda text: print(redact(text), flush=True), args.usage))
    except KeyboardInterrupt:
        print("DayOut остановлен.", file=sys.stderr)
        return 130
    except Exception as error:
        print(redact(safe_error(error)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
