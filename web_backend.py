"""Streamlit adapter: session data only, fresh async transports per action."""

import asyncio
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from time import monotonic

import httpx
from dotenv import dotenv_values
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agent import AgentError, DayOutAgent, GroqClient, GroqConnectionError, SUPPORTED_MODELS, Reply, discover_tools
from dayout import safe_error
from groq_check import quota_pause_message
from mcp_check import MCP_URL

ENV_PATH = Path(__file__).resolve().with_name(".env")
CONFIG_NAMES = ("GROQ_API_KEY", "GOOGLE_MAPS_API_KEY", "GROQ_MODEL")


@dataclass(frozen=True)
class WebConfig:
    groq_key: str = field(repr=False)
    maps_key: str = field(repr=False)
    model: str

    def redact(self, text):
        for key in (self.groq_key, self.maps_key):
            if key:
                text = text.replace(key, "[REDACTED]")
        return text


def load_web_config(secrets=None) -> WebConfig:
    """Environment > Streamlit Secrets > local .env, without mutating os.environ."""
    local = dotenv_values(ENV_PATH) if ENV_PATH.is_file() else {}
    secrets = secrets or {}
    values = {}
    for name in CONFIG_NAMES:
        value = os.environ.get(name, secrets.get(name, local.get(name, "")))
        values[name] = str(value or "").strip()
    if not values["GROQ_API_KEY"] or not values["GOOGLE_MAPS_API_KEY"]:
        raise AgentError("Не настроены ключи сервиса. Заполните .env локально или Secrets на хостинге.")
    if values["GROQ_MODEL"] not in SUPPORTED_MODELS:
        raise AgentError("Неподдерживаемая GROQ_MODEL. Допустимы: " + ", ".join(sorted(SUPPORTED_MODELS)))
    return WebConfig(values["GROQ_API_KEY"], values["GOOGLE_MAPS_API_KEY"], values["GROQ_MODEL"])


@dataclass
class WebMemory:
    # Never put clients, event loops, keys or locks into browser session state.
    turns: list = field(default_factory=list)
    user_history: list = field(default_factory=list)
    pending: tuple | None = None
    usage_totals: dict = field(default_factory=lambda: {
        "requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "missing_usage": 0,
    })
    prompt_ratios: list = field(default_factory=list)
    failed_query: str | None = None
    paused: bool = False

    def reset(self):
        self.turns.clear()
        self.user_history.clear()
        self.pending = None
        self.failed_query = None
        # Resetting a conversation does not repair a failed connection.


class BusyError(AgentError):
    """Request was not accepted; conversation has not been changed."""


class RequestGate:
    """One in-flight action per process, a short cooldown, numeric quota only."""
    def __init__(self, cooldown=5):
        self.lock = threading.Lock()
        self.cooldown = cooldown
        self.last_started = float("-inf")
        self.remaining_tokens = None
        self.reset_at = 0.0
        self.blocked_until = 0.0
        self.quota_kind = None
        self.model = None
        self.model_limits = {}

    @contextmanager
    def slot(self, model=None):
        if not self.lock.acquire(blocking=False):
            raise BusyError("Сервис обрабатывает другой запрос. Попробуйте немного позже.")
        try:
            if model is not None and model != self.model:
                # Provider quotas are per model; changing the configured model
                # must not inherit the previous model's token window/429 pause.
                if self.model is not None:
                    self.model_limits[self.model] = (
                        self.remaining_tokens, self.reset_at, self.blocked_until, self.quota_kind,
                    )
                (self.remaining_tokens, self.reset_at, self.blocked_until,
                 self.quota_kind) = self.model_limits.get(model, (None, 0.0, 0.0, None))
                self.model = model
            quota_delay = self.blocked_until - monotonic()
            if quota_delay > 0:
                raise BusyError(quota_pause_message(self.quota_kind, quota_delay))
            delay = self.cooldown - (monotonic() - self.last_started)
            if delay > 0:
                raise BusyError(f"Подождите ещё {delay + 1:.0f} с перед следующим запросом.")
            self.last_started = monotonic()
            yield
        finally:
            self.lock.release()


@dataclass
class WebResult:
    reply: Reply | None = None
    error: str | None = None
    partial: str = ""
    events: list = field(default_factory=list)


def error_leaves(error):
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            yield from error_leaves(child)
    else:
        yield error


async def _perform(config, memory, gate, action, query, emit):
    result = WebResult()
    agent = llm = None

    def event(text):
        text = config.redact(text)
        if text.startswith("DayOut: "):
            result.partial = text.removeprefix("DayOut: ")
        else:
            result.events.append(text)
            emit(text)

    try:
        async with httpx.AsyncClient(headers={"Authorization": f"Bearer {config.groq_key}"}, timeout=45) as groq:
            llm = GroqClient(groq, config.model, emit=event)
            llm.usage_totals = dict(memory.usage_totals)
            llm.prompt_ratios = list(memory.prompt_ratios)
            llm.remaining_tokens, llm.reset_at = gate.remaining_tokens, gate.reset_at
            llm.blocked_until, llm.quota_kind = gate.blocked_until, gate.quota_kind
            if action == "check":
                event("Проверяю связь с сервисом модели…")
                await llm.check_access()
                memory.paused = False
            else:
                event("Подключаю поиск мест…")
                async with httpx.AsyncClient(headers={"X-Goog-Api-Key": config.maps_key}, timeout=45) as maps:
                    async with streamable_http_client(MCP_URL, http_client=maps) as (read, write, _):
                        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=45)) as session:
                            async with asyncio.timeout(60):
                                await session.initialize()
                                tools = await discover_tools(session)
                            agent = DayOutAgent(session, llm, tools, emit=event)
                            agent.turns, agent.user_history, agent.pending = memory.turns, memory.user_history, memory.pending
                            event("Подбираю варианты…")
                            result.reply = await agent.ask(query, resume=action == "retry")
                            result.reply.text = config.redact(result.reply.text)
                            memory.failed_query = None
    except Exception as error:
        # Closing an MCP transport can fail after the answer has been validated.
        # Keep the committed successful reply instead of offering to repeat it.
        if result.reply is None:
            result.error = config.redact(safe_error(error))
            if any(isinstance(item, GroqConnectionError) for item in error_leaves(error)):
                memory.paused = True
    finally:
        if llm is not None:
            memory.usage_totals = dict(llm.usage_totals)
            memory.prompt_ratios = list(llm.prompt_ratios)
            gate.remaining_tokens, gate.reset_at = llm.remaining_tokens, llm.reset_at
            gate.blocked_until, gate.quota_kind = llm.blocked_until, llm.quota_kind
        if agent is not None:
            memory.turns, memory.user_history, memory.pending = agent.turns, agent.user_history, agent.pending
    return result


def run_action(config, memory, gate, action, query="", emit=lambda _: None):
    """Run from Streamlit's script thread; async resources die in this same call."""
    if action not in {"ask", "retry", "check"}:
        raise AgentError("Неизвестное действие.")
    if action != "check" and memory.paused:
        raise AgentError("Сначала проверьте связь кнопкой «Проверить связь».")
    if action == "retry":
        if not memory.failed_query:
            raise AgentError("Нет запроса для продолжения.")
        query = memory.failed_query
    if action != "check" and not 1 <= len(query.strip()) <= 4000:
        raise AgentError("Введите запрос от 1 до 4000 символов.")
    with gate.slot(config.model):
        if action == "ask":
            memory.pending = None
            memory.failed_query = query
        return asyncio.run(_perform(config, memory, gate, action, query.strip(), emit))


def progress_label(event):
    if event.startswith("MCP → "):
        return {
            "search_places": "Ищу подходящие места…", "lookup_weather": "Проверяю погоду…",
            "compute_routes": "Проверяю маршрут…",
        }.get(event.removeprefix("MCP → "), "Уточняю данные о местах…")
    if "ожидание обновления лимита" in event:
        return "Жду обновления лимита сервиса модели…"
    if "одна попытка исправления" in event:
        return "Проверяю ответ и источники…"
    if "продолжаю незавершённый" in event:
        return "Продолжаю подбор с найденными данными…"
    if event.startswith("Groq:"):
        return "Обрабатываю результат…"
    return event
