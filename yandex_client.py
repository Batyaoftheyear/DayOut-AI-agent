"""Yandex Chat Completions adapter using existing bounded agent protocol."""
import httpx

from agent import AgentError, GroqClient, GroqConnectionError, QuotaPauseError
from groq_check import quota_details

API_URL = "https://ai.api.cloud.yandex.net/v1"


def provider_error(error):
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        descriptions = {
            400: "API отклонил параметры запроса или схему инструментов.",
            401: "Ключ не принят. Проверьте YANDEX_API_KEY.",
            402: "Проверьте баланс и статус платёжного аккаунта.",
            403: "Доступ запрещён. Проверьте каталог, права ключа и статус платёжного аккаунта.",
            404: "Модель не найдена или недоступна в выбранном каталоге.",
            429: "Достигнут лимит Yandex AI Studio.",
        }
        message = f"Yandex AI Studio HTTP {status}: " + descriptions.get(status, "Запрос не выполнен.")
        if status == 429:
            _, wait = quota_details(error.response)
            if wait is not None:
                message += f" Повторите не раньше чем через {wait} с."
            return QuotaPauseError(message + " Автоматических повторов нет.")
        if status in {401, 402, 403}:
            return GroqConnectionError(message)
        return AgentError(message)
    if isinstance(error, httpx.RequestError):
        return GroqConnectionError(f"Ошибка сети Yandex AI Studio ({type(error).__name__}). Проверьте сеть и доступ к API.")
    return AgentError("Yandex AI Studio: некорректный ответ API. Содержимое ошибки скрыто.")


class YandexClient(GroqClient):
    planning_enabled = True
    provider = "Yandex AI Studio"
    api_url = API_URL
    token_parameter = "max_tokens"

    def request_options(self):
        return {}  # Do not send Groq-only reasoning controls.

    def update_limits(self, headers):
        # Groq's token-window headers and 8000-TPM assumptions do not apply here.
        # Actual HTTP 429/Retry-After is still handled by the shared protocol.
        pass

    def convert_error(self, error):
        return provider_error(error)

    def cache_allowed(self):
        # Disabling request logging is separate from provider inference caching.
        return True


def make_client(http, config, emit=lambda _: None, show_usage=False):
    kind = YandexClient if config.provider == "yandex" else GroqClient
    return kind(http, config.model, emit=emit, show_usage=show_usage)
