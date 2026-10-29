import tempfile
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import web_backend as web
from agent import REVIEWED_MODEL, Reply


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("GROQ_API_KEY", "synthetic-web-key")
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "synthetic-maps-key")
    monkeypatch.setenv("GROQ_MODEL", REVIEWED_MODEL)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_initial_screen_loads_without_api_calls(app, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Opening the page must not call an API")
    monkeypatch.setattr(web, "run_action", forbidden)
    app.run()
    assert not app.exception
    assert app.title[0].value == "Куда пойдём?"
    assert not app.chat_input[0].disabled
    assert len(app.chat_message) == 0
    from agent import AGENT_RELEASE
    assert any(AGENT_RELEASE in caption.value for caption in app.caption)


def test_submit_followup_and_reset_render_in_browser_session(app, monkeypatch):
    seen = []
    def action(config, memory, gate, kind, query, emit):
        seen.append((kind, query))
        emit("MCP → search_places")
        memory.user_history.append(query)
        return web.WebResult(reply=Reply("Синтетическое место [Google Maps](https://example.test/place)", ["search_places"], .5))
    monkeypatch.setattr(web, "run_action", action)
    app.run()
    app.chat_input[0].set_value("Найди кафе в Москве").run()
    assert not app.exception
    assert len(app.chat_message) == 2 and len(seen) == 1
    app.chat_input[0].set_value("Теперь в помещении").run()
    assert len(app.chat_message) == 4 and len(seen) == 2
    button(app, "Новый диалог").click().run()
    assert not app.exception and not app.chat_message
    assert not app.session_state.memory.user_history


def test_failure_shows_partial_data_check_and_manual_continuation(app, monkeypatch):
    calls = []
    def action(config, memory, gate, kind, query, emit):
        calls.append(kind)
        if kind == "ask":
            memory.paused = True
            memory.failed_query = query
            return web.WebResult(error="Ошибка связи", partial="Кандидат [Google Maps](https://example.test/fixture)")
        if kind == "check":
            memory.paused = False
            return web.WebResult()
        memory.failed_query = None
        return web.WebResult(reply=Reply("Подбор завершён", [], .5))
    monkeypatch.setattr(web, "run_action", action)
    app.run()
    button(app, "☕ Кофе и разговоры").click().run()
    assert not app.exception
    assert app.chat_input[0].disabled
    assert button(app, "Продолжить запрос").disabled
    assert any("Подбор ещё не завершён" in item.value for item in app.info)
    button(app, "Проверить связь").click().run()
    assert not button(app, "Продолжить запрос").disabled
    button(app, "Продолжить запрос").click().run()
    assert calls == ["ask", "check", "retry"]
    assert not app.exception


def test_busy_request_can_be_resubmitted_without_retyping(app, monkeypatch):
    attempts = []
    def action(config, memory, gate, kind, query, emit):
        attempts.append(query)
        if len(attempts) == 1:
            raise web.BusyError("Сервис занят")
        return web.WebResult(reply=Reply("Готово", [], .5))
    monkeypatch.setattr(web, "run_action", action)
    app.run()
    app.chat_input[0].set_value("Сохранить мой запрос").run()
    assert not app.exception and len(app.chat_message) == 0
    button(app, "Повторить отправку").click().run()
    assert attempts == ["Сохранить мой запрос"] * 2
    assert len(app.chat_message) == 2


def test_weather_details_show_the_date_actually_requested(app, monkeypatch):
    def action(*args, **kwargs):
        return web.WebResult(reply=Reply("Синтетический прогноз.", ["lookup_weather"], .1,
            observations={"task": "weather", "weather_requests": [
                {"date": {"year": 2026, "month": 10, "day": 30}, "hour": None}]}))
    monkeypatch.setattr(web, "run_action", action)
    app.run()
    app.chat_input[0].set_value("Погода завтра в Москве").run()
    assert not app.exception
    assert any("Запрос погоды: 30.10.2026, на день" in item.value for item in app.markdown)
