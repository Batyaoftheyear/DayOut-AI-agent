"""HTTP error and rate-limit helpers for chat-completion clients."""

import math
import re

import httpx


API_URL = "https://api.groq.com/openai/v1"


GPT_OSS_MODEL = "openai/gpt-oss-120b"


def model_options(model: str) -> dict:
    # Hiding reasoning does not remove its token cost. Low effort limits latency.
    return {"reasoning_effort": "low", "include_reasoning": False} if model == GPT_OSS_MODEL else {}


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


QUOTA_LABELS = {
    "TPD": "суточный лимит токенов (TPD)",
    "TPM": "минутный лимит токенов (TPM)",
    "RPD": "суточный лимит запросов (RPD)",
    "RPM": "минутный лимит запросов (RPM)",
    "ITPM": "минутный лимит входных токенов (ITPM)",
    "OTPM": "минутный лимит выходных токенов (OTPM)",
}


def quota_details(response: httpx.Response) -> tuple[str | None, int | None]:
    """Extract a known quota label and numeric Retry-After, never raw messages."""
    kind = None
    try:
        data = response.json()
        detail = data.get("error") if isinstance(data, dict) else None
        message = detail.get("message") if isinstance(detail, dict) else None
        if isinstance(message, str):
            for code in QUOTA_LABELS:
                if re.search(r"\b" + code + r"\b", message, re.I):
                    kind = code
                    break
    except ValueError:
        pass
    value = response.headers.get("retry-after", "")
    wait = None
    if re.fullmatch(r"[0-9]{1,7}(?:\.[0-9]{1,3})?", value):
        seconds = float(value)
        if 0 <= seconds <= 604800:
            wait = math.ceil(seconds)
    return kind, wait


def quota_pause_message(kind: str | None, seconds: float) -> str:
    label = QUOTA_LABELS.get(kind, "лимит API")
    return (f"Groq: {label}; до следующей попытки ещё {math.ceil(seconds)} с. "
            "Запрос не отправлен. После паузы можно продолжить вручную.")


def error_message(error: Exception) -> str:
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
            kind, wait = quota_details(error.response)
            label = QUOTA_LABELS.get(kind, "лимит API (тип квоты не указан)")
            pause = (f"Повторите не раньше чем через {wait} с." if wait is not None else
                     "Дождитесь восстановления квоты; время ожидания сервер не указал.")
            return f"Groq HTTP 429: превышен {label}. {pause} Автоматических повторов нет."
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
