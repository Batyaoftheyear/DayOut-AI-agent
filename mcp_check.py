"""Read live Google Maps MCP schemas and optionally perform one place search."""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv
from jsonschema import SchemaError, ValidationError, validate
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "https://mapstools.googleapis.com/mcp"
ENV_PATH = Path(__file__).resolve().with_name(".env")


class CheckError(Exception):
    """An actionable diagnostic that is safe to show to the user."""


def load_key() -> str:
    load_dotenv(ENV_PATH, override=False)
    key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if not key:
        raise CheckError(
            "Не задан GOOGLE_MAPS_API_KEY. Скопируйте .env.example в .env "
            "и добавьте Google Maps Demo Key. Интеграция не проверена."
        )
    return key


def search_arguments(schema: dict, query: str) -> dict:
    """Use only fields advertised by the live tool, accepting protobuf aliases."""
    properties = schema.get("properties", {})
    query_field = next((name for name in ("text_query", "textQuery") if name in properties), None)
    if query_field is None:
        raise CheckError("В схеме search_places нет известного поля запроса. Проверьте выведенную схему.")
    arguments = {query_field: query}
    language_field = next((name for name in ("language_code", "languageCode") if name in properties), None)
    if language_field:
        arguments[language_field] = "ru"
    try:
        validate(arguments, schema)
    except (ValidationError, SchemaError):
        raise CheckError(
            "Аргументы не соответствуют реальной схеме search_places. "
            "Поиск не выполнен; проверьте выведенную схему."
        ) from None
    return arguments


async def inspect_session(session, query: str, list_only: bool, emit) -> None:
    tools = []
    cursor = None
    seen = set()
    for _ in range(20):
        page = await session.list_tools(cursor=cursor)
        tools.extend(page.tools)
        cursor = page.nextCursor
        if not cursor:
            break
        if cursor in seen:
            raise CheckError("Сервер повторяет курсор списка инструментов.")
        seen.add(cursor)
    else:
        raise CheckError("Превышен лимит страниц списка инструментов (20).")

    if not tools:
        raise CheckError("MCP вернул пустой список инструментов.")
    emit("Инструменты MCP и их реальные входные JSON-схемы:")
    emit(json.dumps([
        {"name": tool.name, "description": tool.description, "inputSchema": tool.inputSchema}
        for tool in tools
    ], ensure_ascii=False, indent=2))
    if list_only:
        emit("Получение схем завершено. search_places не вызывался.")
        return
    search = next((tool for tool in tools if tool.name == "search_places"), None)
    if search is None:
        raise CheckError("Сервер не объявил инструмент search_places.")
    arguments = search_arguments(search.inputSchema, query)
    emit("Вызов search_places: " + json.dumps(arguments, ensure_ascii=False))
    result = await session.call_tool("search_places", arguments=arguments)
    if result.isError:
        raise CheckError(
            "search_places вернул MCP isError=true. Проверьте доступ ключа, "
            "квоту и параметры. Интеграционный поиск не подтверждён."
        )
    if not result.content and not result.structuredContent:
        raise CheckError("search_places вернул пустой ответ; поиск не подтверждён.")
    emit("Ответ Google Maps (включая источники и атрибуцию, если они возвращены):")
    emit(result.model_dump_json(by_alias=True, exclude_none=True, indent=2))
    emit("MCP-вызов завершён. Проверьте содержимое ответа: наличие мест не гарантируется.")


async def run_check(key: str, query: str, list_only: bool, timeout: float, emit) -> None:
    # One total deadline covers connection, initialization, pagination and search.
    async with asyncio.timeout(timeout):
        async with httpx.AsyncClient(
            headers={"X-Goog-Api-Key": key}, timeout=httpx.Timeout(timeout),
            follow_redirects=False,
        ) as client:
            async with streamable_http_client(MCP_URL, http_client=client) as (read, write, _):
                async with ClientSession(
                    read, write, read_timeout_seconds=timedelta(seconds=timeout),
                ) as session:
                    await session.initialize()
                    await inspect_session(session, query, list_only, emit)


def error_message(error: BaseException) -> str:
    if isinstance(error, BaseExceptionGroup):
        return " / ".join(dict.fromkeys(error_message(item) for item in error.exceptions))
    if isinstance(error, CheckError):
        return str(error)
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status in (401, 403):
            return f"HTTP {status}: отказ в доступе. Проверьте ключ, ограничения ключа и доступ к Maps Grounding Lite."
        if status == 429:
            return "HTTP 429: превышена квота. Повторите позже; автоматических повторов нет."
        return f"HTTP {status}: ошибка MCP-сервера или HTTP-запроса."
    if isinstance(error, httpx.TimeoutException):
        return f"Истёк таймаут MCP ({type(error).__name__}). Проверьте сеть и повторите позже."
    if isinstance(error, TimeoutError):
        return "Истёк общий таймаут MCP. Повторите позже."
    if isinstance(error, httpx.ProxyError):
        return "Ошибка сети MCP (ProxyError): не удалось подключиться через прокси. Проверьте настройки прокси процесса."
    if isinstance(error, httpx.ConnectError):
        return "Ошибка сети MCP (ConnectError): соединение не установлено. Проверьте доступ к сети у процесса, сеть и прокси."
    if isinstance(error, httpx.ReadError):
        return "Ошибка сети MCP (ReadError): соединение оборвалось при получении ответа. Повторите позже."
    if isinstance(error, httpx.RemoteProtocolError):
        return "Ошибка сети MCP (RemoteProtocolError): сервер или прокси прервал HTTP-обмен."
    if isinstance(error, httpx.RequestError):
        return "Ошибка сети при подключении к MCP. Проверьте сеть и прокси."
    # Exception bodies can contain request headers, credentials or Maps Content.
    return "Ошибка протокола MCP или обработки ответа. Проверьте доступ к сервису и версии зависимостей."


def positive_timeout(value: str) -> float:
    number = float(value)
    if not 0 < number <= 300:
        raise argparse.ArgumentTypeError("Таймаут должен быть больше 0 и не больше 300 секунд.")
    return number


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-only", action="store_true", help="Только получить схемы, без поиска")
    parser.add_argument("--query", default="Музеи Москвы, Россия", help="Запрос поиска с указанием города")
    parser.add_argument("--timeout", type=positive_timeout, default=60.0, help="Общий таймаут, секунды")
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("Поисковый запрос не должен быть пустым.")
    # Avoid SDK/HTTP diagnostics leaking credentials through exception tracebacks.
    logging.disable(logging.CRITICAL)
    key = ""
    try:
        key = load_key()
        def emit(message: str) -> None:
            print(message.replace(key, "[REDACTED]"), flush=True)
        asyncio.run(run_check(key, args.query, args.list_only, args.timeout, emit))
    except KeyboardInterrupt:
        print("Проверка прервана.", file=sys.stderr)
        return 130
    except Exception as error:
        message = error_message(error)
        print(message.replace(key, "[REDACTED]") if key else message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
