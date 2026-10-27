"""Check Yandex access and a synthetic tool round trip; no Google Maps calls."""
import asyncio
import logging
import sys

import httpx

from agent import AgentError
from dayout import safe_error
from groq_check import DEMO_TOOL, execute_demo_call
from provider_config import load_settings
from yandex_client import make_client


async def check(config):
    async with httpx.AsyncClient(headers=config.headers, timeout=45) as http:
        llm = make_client(http, config)
        await llm.check_access()
        print("Yandex AI Studio: список моделей и генерация доступны.", flush=True)
        messages = [{"role": "system", "content": "Технический тест. Вызови get_demo_activity для Москвы. Затем сообщи по-русски точный demo_code из результата и что данные вымышленные."},
                    {"role": "user", "content": "Получи код искусственной активности для Москвы."}]
        first = await llm.complete(messages, [DEMO_TOOL], max_tokens=512)
        calls = first.get("tool_calls") or []
        if len(calls) != 1:
            raise AgentError("Yandex AI Studio: модель не выбрала один искусственный инструмент.")
        result = execute_demo_call(calls[0])
        messages.extend([first, result])
        final = await llm.complete(messages, [DEMO_TOOL], allow_tools=False, max_tokens=512)
        if final.get("tool_calls") or "DEMO-42" not in (final.get("content") or ""):
            raise AgentError("Yandex AI Studio: финальный ответ не подтвердил код искусственного инструмента.")
        print("Yandex AI Studio: искусственный инструмент и финальный ответ — OK.")
        usage = llm.usage_totals
        print(f"Обращений LLM: {usage['requests']}; вход: {usage['prompt_tokens']}; выход: {usage['completion_tokens']} токенов.")


def main():
    logging.disable(logging.CRITICAL)
    config = None
    try:
        config = load_settings()
        if config.provider != "yandex":
            raise AgentError("Для этой проверки задайте LLM_PROVIDER=yandex.")
        asyncio.run(check(config))
        return 0
    except KeyboardInterrupt:
        print("Проверка остановлена.")
        return 130
    except Exception as error:
        message = safe_error(error)
        print(config.redact(message) if config else message)
        return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
