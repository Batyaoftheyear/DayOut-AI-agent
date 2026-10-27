"""Check Groq access and a tool-calling round trip using synthetic data only."""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from jsonschema import ValidationError, validate

API_URL = "https://api.groq.com/openai/v1"
ENV_PATH = Path(__file__).resolve().with_name(".env")
DEMO_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string", "enum": ["Москва"]}},
    "required": ["city"],
    "additionalProperties": False,
}
DEMO_TOOL = {"type": "function", "function": {
    "name": "get_demo_activity",
    "description": "Получить искусственный пример активности и его код для теста. Это не реальные места.",
    "parameters": DEMO_SCHEMA,
}}


class GroqCheckError(Exception):
    """Safe diagnostic with no raw request, response or credentials."""


def load_config() -> tuple[str, str]:
    load_dotenv(ENV_PATH, override=False)
    key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_MODEL", "").strip()
    if not key:
        raise GroqCheckError("Не задан GROQ_API_KEY. Добавьте ключ Groq в локальный .env.")
    return key, model


def request_json(client: httpx.Client, method: str, path: str, **kwargs) -> dict:
    response = client.request(method, f"{API_URL}/{path}", **kwargs)
    response.raise_for_status()
    try:
        data = response.json()
    except ValueError:
        raise GroqCheckError("Groq вернул ответ, который не является JSON.") from None
    if not isinstance(data, dict):
        raise GroqCheckError("Groq вернул неожиданный формат ответа.")
    return data


def completion(client: httpx.Client, model: str, messages: list, **kwargs) -> dict:
    data = request_json(client, "POST", "chat/completions", json={
        "model": model, "messages": messages, "max_completion_tokens": 2048,
        **kwargs,
    })
    try:
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise GroqCheckError("Ответ модели обрезан лимитом токенов; проверка не завершена.")
        message = choice["message"]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ValueError
    except (KeyError, IndexError, TypeError, ValueError):
        raise GroqCheckError("Groq не вернул ожидаемое сообщение assistant.") from None
    return message


def execute_demo_call(call: dict) -> dict:
    try:
        function = call["function"]
        if call.get("type") != "function" or function["name"] != "get_demo_activity":
            raise GroqCheckError("Модель запросила неизвестный инструмент; выполнение запрещено.")
        if not isinstance(call["id"], str) or not call["id"]:
            raise ValueError
        arguments = json.loads(function["arguments"])
        validate(arguments, DEMO_SCHEMA)
    except (KeyError, TypeError, ValueError, ValidationError):
        raise GroqCheckError("Модель вернула некорректные аргументы или ID вызова инструмента.") from None
    return {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({
        "city": arguments["city"], "activity": "Вымышленная тестовая активность",
        "demo_code": "DEMO-42", "synthetic": True,
    }, ensure_ascii=False)}


def check_groq(client: httpx.Client, model: str, list_only: bool, emit) -> None:
    data = request_json(client, "GET", "models")
    try:
        models = sorted({item["id"] for item in data["data"] if isinstance(item.get("id"), str)})
    except (KeyError, TypeError, AttributeError):
        raise GroqCheckError("Groq вернул некорректный список моделей.") from None
    if not models:
        raise GroqCheckError("Groq вернул пустой список моделей.")
    emit("Авторизация Groq: OK.")
    if list_only:
        emit("Идентификаторы моделей API (наличие в списке не гарантирует бесплатный доступ):")
        emit("\n".join(models))
        return
    if not model:
        raise GroqCheckError("Не задан GROQ_MODEL. Укажите ID модели из каталога; список: --list-models.")
    if model not in models:
        raise GroqCheckError(
            "GROQ_MODEL отсутствует в реальном списке API. Выполните --list-models "
            "и выберите доступную на бесплатном тарифе модель. Автозамены нет."
        )
    emit("Модель найдена в API: " + model)
    messages = [
        {"role": "system", "content": (
            "Ты проходишь технический тест. Отвечай по-русски. Для получения тестовой "
            "активности используй доступный инструмент. Не придумывай его результат. "
            "После получения результата напиши одно предложение с городом, точным "
            "demo_code и пояснением, что данные вымышленные."
        )},
        {"role": "user", "content": "Получи тестовую активность для Москвы и сообщи её код."},
    ]
    first = completion(client, model, messages, tools=[DEMO_TOOL], tool_choice="auto")
    calls = first.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise GroqCheckError("Модель не выбрала ровно один тестовый инструмент. Tool calling не подтверждён.")
    tool_result = execute_demo_call(calls[0])
    # Forward only the assistant fields belonging to the tool-calling protocol.
    messages.append({"role": "assistant", "content": first.get("content"), "tool_calls": calls})
    messages.append(tool_result)
    emit("Выбор инструмента и проверка аргументов: OK (искусственные данные).")
    final = completion(client, model, messages, tools=[DEMO_TOOL], tool_choice="none")
    content = final.get("content")
    if final.get("tool_calls") or not isinstance(content, str) or not content.strip():
        raise GroqCheckError("Модель не вернула финальный текст после выполнения инструмента.")
    if "DEMO-42" not in content or not re.search("[А-Яа-яЁё]", content):
        raise GroqCheckError("Финальный ответ не содержит тестовый код или русский текст.")
    emit("Ответ модели: " + content.strip())
    emit("Groq: проверка завершена. Google Maps не вызывался и его данные модели не передавались.")


def request_error_reason(response: httpx.Response) -> str:
    """Translate known codes only; provider messages may contain prompts or keys."""
    try:
        data = response.json()
    except ValueError:
        data = None
    detail = data.get("error") if isinstance(data, dict) else None
    if isinstance(detail, dict):
        code = detail.get("code")
        reasons = {
            "tool_use_failed": "Groq отклонил сгенерированный вызов инструмента (tool_use_failed). "
                               "Можно повторить запрос командой /retry; если ошибка повторяется, нужна проверка схем и tool calling.",
            "tool_call_validation_failed": "Вызов инструмента не прошёл проверку Groq (tool_call_validation_failed). "
                                           "Нужна проверка аргументов и схем инструментов.",
            "model_not_found": "Модель не найдена или недоступна (model_not_found). Проверьте GROQ_MODEL и доступ к модели.",
            "context_length_exceeded": "Превышен размер контекста (context_length_exceeded). Сократите запрос или выполните /reset.",
        }
        if isinstance(code, str) and code in reasons:
            return reasons[code]
    return ("API отклонил запрос; точная причина по известным кодам не определена. "
            "Возможны ошибки параметров, схем или генерации вызова инструмента.")


def error_message(error: Exception) -> str:
    if isinstance(error, GroqCheckError):
        return str(error)
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status == 401:
            return "Groq HTTP 401: ключ не принят. Проверьте GROQ_API_KEY и сохранение .env."
        if status == 403:
            suffix = ""
            if "html" in error.response.headers.get("content-type", "").lower():
                suffix = " Получена HTML-страница: проверьте также прокси и сетевую фильтрацию."
            return ("Groq HTTP 403: доступ запрещён. Проверьте разрешения аккаунта, модели "
                    "и доступность API из текущей сети. Причина блокировки не установлена." + suffix)
        if status == 429:
            return "Groq HTTP 429: превышен лимит бесплатного API. Повторите позже; автоматических повторов нет."
        if status in (400, 404):
            return f"Groq HTTP {status}: " + request_error_reason(error.response)
        return f"Groq HTTP {status}: запрос не выполнен."
    if isinstance(error, httpx.TimeoutException):
        return f"Истёк таймаут Groq ({type(error).__name__}). Проверьте сеть и повторите позже."
    if isinstance(error, httpx.ProxyError):
        return "Ошибка сети Groq (ProxyError): не удалось установить соединение через прокси. Проверьте его доступность."
    if isinstance(error, httpx.ConnectError):
        return "Ошибка сети Groq (ConnectError): соединение не установлено. Проверьте сеть, прокси и TLS."
    if isinstance(error, httpx.ReadError):
        return "Ошибка сети Groq (ReadError): соединение прервалось при получении ответа. Проверьте сеть и прокси."
    if isinstance(error, httpx.RemoteProtocolError):
        return "Ошибка сети Groq (RemoteProtocolError): сервер или прокси прервал HTTP-обмен."
    if isinstance(error, httpx.RequestError):
        return "Ошибка сети при подключении к Groq. Проверьте сеть и прокси."
    return "Ошибка обработки ответа Groq. Содержимое исключения скрыто для защиты ключей."


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-models", action="store_true", help="Только авторизация и список ID моделей")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    secrets = []
    def redact(message: str) -> str:
        for secret in secrets:
            message = message.replace(secret, "[REDACTED]")
        return message
    try:
        key, model = load_config()
        secrets = [value for value in (key, os.getenv("GOOGLE_MAPS_API_KEY", "")) if value]
        with httpx.Client(
            headers={"Authorization": f"Bearer {key}"}, timeout=45, follow_redirects=False,
        ) as client:
            check_groq(client, model, args.list_models, lambda text: print(redact(text), flush=True))
    except KeyboardInterrupt:
        print("Проверка Groq прервана.", file=sys.stderr)
        return 130
    except Exception as error:
        print(redact(error_message(error)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
