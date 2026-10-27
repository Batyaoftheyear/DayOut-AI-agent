"""Explicit provider selection shared by CLI and Streamlit; no automatic fallback."""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

from agent import AgentError, SUPPORTED_MODELS

CONFIG_NAMES = ("LLM_PROVIDER", "GROQ_API_KEY", "GROQ_MODEL", "GOOGLE_MAPS_API_KEY",
                "YANDEX_API_KEY", "YANDEX_FOLDER_ID", "YANDEX_MODEL")
ENV_PATH = Path(__file__).resolve().with_name(".env")


@dataclass(frozen=True)
class WebConfig:
    llm_key: str = field(repr=False)
    maps_key: str = field(repr=False)
    model: str
    provider: str = "groq"
    folder_id: str = ""

    @property
    def groq_key(self):
        # Compatibility for existing local diagnostic scripts.
        return self.llm_key

    @property
    def label(self):
        return "Yandex AI Studio" if self.provider == "yandex" else "Groq"

    @property
    def headers(self):
        headers = {"Authorization": f"Bearer {self.llm_key}"}
        if self.provider == "yandex":
            headers.update({"OpenAI-Project": self.folder_id, "x-data-logging-enabled": "false"})
        return headers

    def redact(self, text):
        for key in (self.llm_key, self.maps_key):
            if key:
                text = text.replace(key, "[REDACTED]")
        return text


def load_settings(secrets=None, env_path=ENV_PATH):
    """Environment > Streamlit Secrets > .env; never mutate process environment."""
    local = dotenv_values(env_path) if env_path.is_file() else {}
    secrets = secrets or {}
    values = {name: str(os.environ.get(name, secrets.get(name, local.get(name, ""))) or "").strip()
              for name in CONFIG_NAMES}
    provider = values["LLM_PROVIDER"] or "groq"
    if provider not in {"groq", "yandex"}:
        raise AgentError("LLM_PROVIDER должен быть groq или yandex. Автоматического переключения нет.")
    if not values["GOOGLE_MAPS_API_KEY"]:
        raise AgentError("Не настроены ключи сервиса: нужен GOOGLE_MAPS_API_KEY в .env или Secrets.")
    if provider == "yandex":
        key, folder, model = (values[name] for name in ("YANDEX_API_KEY", "YANDEX_FOLDER_ID", "YANDEX_MODEL"))
        if not key or not folder or not model:
            raise AgentError("Не настроены YANDEX_API_KEY, YANDEX_FOLDER_ID и YANDEX_MODEL в .env или Secrets.")
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", folder):
            raise AgentError("Некорректный YANDEX_FOLDER_ID: нужен идентификатор каталога, не имя или URL.")
        uri = f"gpt://{folder}/aliceai-llm-flash"
        if model not in {"aliceai-llm-flash", uri, uri + "/latest"}:
            raise AgentError("В YANDEX_MODEL укажите aliceai-llm-flash или её URI из вашего каталога.")
        return WebConfig(key, values["GOOGLE_MAPS_API_KEY"], uri + "/latest",
                         "yandex", folder)
    if not values["GROQ_API_KEY"]:
        raise AgentError("Не настроены ключи сервиса: нужен GROQ_API_KEY в .env или Secrets.")
    if values["GROQ_MODEL"] not in SUPPORTED_MODELS:
        raise AgentError("Неподдерживаемая GROQ_MODEL. Допустимы: " + ", ".join(sorted(SUPPORTED_MODELS)))
    return WebConfig(values["GROQ_API_KEY"], values["GOOGLE_MAPS_API_KEY"], values["GROQ_MODEL"])
