"""DayOut browser chat. Run: python -m streamlit run app.py"""

from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from agent import AgentError
import agent as agent_runtime
from web_backend import BusyError, CONFIG_NAMES, RequestGate, WebMemory, load_web_config, progress_label, run_action

st.set_page_config(page_title="DayOut — куда пойдём?", page_icon="🧭", layout="centered")
st.html(Path(__file__).with_name("assets").joinpath("style.css"))


@st.cache_resource
def request_gate():
    # Per-model quota state and a lock only; no conversations or Maps data.
    return RequestGate(cooldown=5)


def queue_action(action, query=""):
    st.session_state.action = (action, query)


def clear_chat():
    st.session_state.memory.reset()
    st.session_state.messages = []
    st.session_state.notice = None
    st.session_state.retry_action = None


def render_metrics(reply):
    st.caption(f"{reply.seconds:.1f} с · обращений к инструментам: {len(reply.tools)}")
    with st.expander("Детали запроса"):
        if reply.observations and reply.observations.get("task"):
            task = reply.observations["task"]
            label = {"search": "поиск мест", "weather": "погода", "route": "маршрут", "refine": "уточнение найденного",
                     "chat": "общий вопрос", "clarify": "уточняющий вопрос", "resolve_names": "поиск идентификатора",
                     "resolve_maps_urls": "разбор ссылки"}.get(task, "другая задача")
            st.write("Тип запроса: " + label)
        if reply.tools:
            st.write("Инструменты: " + ", ".join(reply.tools))
        for request in (reply.observations or {}).get("weather_requests", []):
            date = request["date"]
            if date:
                stamp = f"{date['day']:02d}.{date['month']:02d}.{date['year']:04d}"
                stamp += f", {request['hour']:02d}:00" if request["hour"] is not None else ", на день"
            else:
                stamp = "текущая погода (дата не передана)"
            st.write("Запрос погоды: " + stamp)
        if reply.usage:
            usage = reply.usage
            if usage["missing_usage"] == usage["requests"]:
                st.write("API не предоставил расход токенов.")
            else:
                st.write(f"Обращения к LLM: {usage['requests']} · вход: {usage['prompt_tokens']} · выход: {usage['completion_tokens']} токенов.")
                if usage["missing_usage"]:
                    st.caption("Учёт неполный: для части попыток API не вернул расход.")


def render_message(message):
    with st.chat_message(message["role"], avatar="🧭" if message["role"] == "assistant" else "👤"):
        if message.get("kind") == "error":
            st.warning(message["text"])
        elif message.get("kind") == "partial":
            st.info("Подбор ещё не завершён")
            st.markdown(message["text"])
        else:
            # Model text is always Markdown with raw HTML disabled.
            st.markdown(message["text"])
        if message.get("reply"):
            render_metrics(message["reply"])


if "memory" not in st.session_state:
    st.session_state.memory = WebMemory()
    st.session_state.messages = []
    st.session_state.action = None
    st.session_state.notice = None
    st.session_state.retry_action = None

memory = st.session_state.memory
busy = st.session_state.action is not None
config = None
config_error = None
try:
    try:
        secrets = {name: st.secrets.get(name, "") for name in CONFIG_NAMES if name in st.secrets}
    except (StreamlitSecretNotFoundError, FileNotFoundError):
        secrets = {}
    config = load_web_config(secrets)
except AgentError as error:
    config_error = str(error)

with st.sidebar:
    st.markdown("### 🧭 DayOut")
    st.caption("Место найдётся. Осталось выбрать настроение.")
    st.button("Новый диалог", icon="➕", use_container_width=True, disabled=busy, on_click=clear_chat)
    st.divider()
    st.markdown("**Что хочется сегодня?**")
    st.write("Напиши город или район, интересы и пожелания. Время и бюджет помогут сузить выбор.")
    st.caption("Например: «Уютное кафе в центре Москвы, в помещении, сегодня вечером».")
    st.divider()
    if memory.paused:
        st.warning("Нужно проверить связь с моделью.")
    st.button("Проверить связь", use_container_width=True, disabled=busy or config is None,
              on_click=queue_action, args=("check",))
    with st.expander("О проекте"):
        st.write("Учебный ассистент: LLM выбирает инструменты официального Google Maps MCP и составляет ответ по найденным данным.")
        st.caption("Часы работы и стоимость могут быть неизвестны. Проверяйте условия перед посещением.")
        st.caption("Версия агента: " + getattr(agent_runtime, "AGENT_RELEASE", "прежняя — требуется перезапуск приложения"))
        st.link_button("Код проекта ↗", "https://github.com/Batyaoftheyear/DayOut-AI-agent")
        presentation = Path(__file__).with_name("presentation.html")
        if presentation.is_file():
            st.download_button("Презентация для защиты", data=presentation.read_bytes(),
                               file_name="DayOut-presentation.html", mime="text/html")

st.markdown('<div class="eyebrow">ГОРОД · ВПЕЧАТЛЕНИЯ · ВАШИ ПЛАНЫ</div>', unsafe_allow_html=True)
st.title("Куда пойдём?")
st.markdown("Расскажи, чего хочется, — **DayOut** подберёт места и поможет уточнить план.")

if config_error:
    st.info(config_error)
if st.session_state.notice:
    st.info(st.session_state.notice)
if st.session_state.retry_action and not busy:
    st.button("Повторить отправку", on_click=queue_action, args=st.session_state.retry_action,
              disabled=config is None)

if not st.session_state.messages:
    st.markdown('<div class="section-label">Начать с идеи</div>', unsafe_allow_html=True)
    examples = [
        ("☕ Кофе и разговоры", "Найди 2–3 уютных кафе в центре Москвы для спокойного разговора."),
        ("🖼 Что-то новое", "Подбери два интересных музея Москвы для выходного дня. Дай ссылки."),
        ("🌿 Прогулка вдвоём", "Найди красивые места для прогулки вдвоём на юге Москвы."),
    ]
    for column, (label, query) in zip(st.columns(3), examples):
        column.button(label, use_container_width=True, disabled=busy or config is None or memory.paused,
                      on_click=queue_action, args=("ask", query))
    st.caption("Можно выбрать идею или написать свой запрос ниже.")

for message in st.session_state.messages:
    render_message(message)

if memory.failed_query and not busy and config is not None:
    st.button("Продолжить запрос", type="primary", disabled=memory.paused,
              on_click=queue_action, args=("retry",))
    st.caption("В течение 5 минут продолжим с сохранёнными результатами, если они есть. При паузе из-за лимита сначала дождитесь указанного времени.")

prompt = st.chat_input("Например: куда сходить вдвоём в Москве вечером?", max_chars=4000,
                       disabled=busy or config is None or memory.paused)
if prompt:
    queue_action("ask", prompt)
    st.rerun()

st.caption("DayOut · места и источники из Google Maps")

if busy and config is not None:
    action, query = st.session_state.action
    if action == "ask":
        with st.chat_message("user", avatar="👤"):
            st.markdown(config.redact(query))
    try:
        with st.status("Подключаюсь…", expanded=False) as status:
            result = run_action(config, memory, request_gate(), action, query,
                                lambda event: status.update(label=progress_label(event)))
            status.update(label="Готово" if result.error is None else "Запрос остановлен",
                          state="complete" if result.error is None else "error")
        st.session_state.notice = None
        st.session_state.retry_action = None
        if action == "ask":
            st.session_state.messages.append({"role": "user", "text": config.redact(query)})
        if result.reply:
            st.session_state.messages.append({"role": "assistant", "text": result.reply.text, "reply": result.reply})
        if result.partial and result.error:
            st.session_state.messages.append({"role": "assistant", "text": result.partial, "kind": "partial"})
        if result.error:
            st.session_state.messages.append({"role": "assistant", "text": result.error, "kind": "error"})
        elif action == "check":
            st.session_state.notice = "Связь с моделью работает. Можно продолжить запрос или начать новый."
        st.session_state.messages = st.session_state.messages[-40:]
    except AgentError as error:
        st.session_state.notice = config.redact(str(error))
        if isinstance(error, BusyError):
            st.session_state.retry_action = (action, query)
    finally:
        st.session_state.action = None
    st.rerun()
