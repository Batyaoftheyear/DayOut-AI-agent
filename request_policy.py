"""Semantic turn plan from user text, with deterministic constraint merging."""
import copy
import json
from datetime import date
from jsonschema import validate

TOOLS = ["search_places", "lookup_weather", "compute_routes", "resolve_names", "resolve_maps_urls"]
TASK_TO_TOOL = {"search": "search_places", "weather": "lookup_weather", "route": "compute_routes",
                "resolve_names": "resolve_names", "resolve_maps_urls": "resolve_maps_urls"}
PLAN_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "task": {"enum": [*TASK_TO_TOOL, "refine", "chat", "clarify"]},
        "city": {"type": "string", "maxLength": 120},
        "area": {"type": "string", "maxLength": 160},
        "activity": {"type": "string", "maxLength": 160},
        "purpose": {"type": "string", "maxLength": 200},
        "other_constraints": {"type": "string", "maxLength": 500},
        "budget": {"type": "object", "additionalProperties": False, "properties": {
            "action": {"enum": ["keep", "set", "clear"]},
            "amount": {"type": ["number", "null"], "minimum": 0},
            "currency": {"enum": ["RUB", "USD", "EUR", "unknown"]},
            "people": {"type": ["integer", "null"], "minimum": 1, "maximum": 100},
            "scope": {"enum": ["total", "per_person", "per_hour", "unknown"]},
        }, "required": ["action", "amount", "currency", "people", "scope"]},
        "strict_area": {"type": "boolean"},
        "required_tools": {"type": "array", "items": {"enum": TOOLS}, "uniqueItems": True},
        "reference_kind": {"enum": ["places", "weather", "routes", "none"]},
        "price_requested": {"type": "boolean"},
        "weather_date": {"type": ["string", "null"]},
        "weather_hour": {"type": ["integer", "null"], "minimum": 0, "maximum": 23},
        "geo_anchor": {"type": ["string", "null"]},
        "walk_limit_minutes": {"type": ["integer", "null"], "minimum": 1, "maximum": 720},
    },
    "required": ["task", "city", "area", "activity", "purpose", "other_constraints", "budget",
                 "strict_area", "required_tools", "reference_kind", "price_requested", "weather_date", "weather_hour", "geo_anchor", "walk_limit_minutes"],
}


def response_format():
    # Yandex rejects uniqueItems in structured output; the full local validator
    # still enforces uniqueness before any plan can affect execution.
    schema = copy.deepcopy(PLAN_SCHEMA)
    schema["properties"]["required_tools"].pop("uniqueItems")
    return {"type": "json_schema", "json_schema": {"name": "dayout_request_plan", "schema": schema}}

PLAN_PROMPT = """Разбери текущую просьбу для помощника по досугу. Верни только JSON по схеме.
Это внутренний план, не ответ пользователю. Данные о реальных местах здесь отсутствуют.
Сохрани применимые прежние условия, новые заменяют старые. Не угадывай город, цены или предпочтения.
task=search: поиск новых мест, смена категории (кафе→баня), района, времени/бюджета или иных условий поиска.
refine: только сократить/объяснить уже выбранные варианты, без новых фактов. chat: общая беседа или объяснение ошибки.
Явное «не ищи», «только запомни», «перечисли условия» означает refine/chat и required_tools=[], даже если в просьбе есть слово «найди» в описании будущей цели.
weather/route/resolve_names/resolve_maps_urls: соответствующая прямая просьба. clarify: критически не хватает города,
условия противоречивы, неверная сумма/дата; дата не обязательна для обычного поиска.
required_tools: все инструменты, необходимые в этом ходе, включая сочетания поиск+погода+маршрут.
Если ищут места рядом с конкретной станцией метро или адресом, включи compute_routes: нужно проверить пешую близость кандидатов к ориентиру.
Обычный район города без конкретного ориентира сам по себе не требует маршрута.
geo_anchor: конкретная станция/адрес, вокруг которого ищут места, иначе null. Для такого поиска walk_limit_minutes=15 по умолчанию либо явный предел пользователя.
Для составного плана «прогулка и кафе» walk_limit_minutes=30 по умолчанию: не предлагай многокилометровый переход как удобный план.
Для прямого запроса маршрута без ограничения времени walk_limit_minutes=null. Обычный поиск в районе тоже null.
Не включай weather для обычного поиска без даты или когда «холодно/дождь» уже задано как условие помещения.
Не выдумывай необходимость resolver: он нужен для определения места из имени/ссылки, если это требуется задачей.
reference_kind: какие прежние данные нужны (places/weather/routes), иначе none.
budget.action=keep если денежные условия и число людей не меняли; set при новой явной сумме ИЛИ изменении единицы/числа людей; clear если ограничение отменено.
При изменении только числа людей/единицы сохрани прежнюю сумму; если её нет, amount=null. Не придумывай новую сумму.
Сохраняй число людей и единицу: total=на всех, per_person=на человека, per_hour=за час; не угадывай единицу.
strict_area=true только при явном запрете выходить за район; иначе допустима одна попытка поиска рядом при пустой выдаче.
Примеры: «в Нагатино» → strict_area=false; «только Нагатино, другие районы не подходят» → true.
price_requested=true если просят цены/стоимость. Заполни ВСЕ поля, пустую информацию обозначь пустой строкой/null.
weather_date: ISO YYYY-MM-DD для погоды нужного дня; «сегодня/завтра» вычисли по now в Europe/Moscow.
weather_hour: час 0–23 только для явно запрошенного почасового прогноза, иначе null.
Для текущей погоды weather_date=null и weather_hour=null. Эти поля относятся к погоде этого хода, а не к будущим поискам.
"""


def plan_messages(history, query, previous, now):
    return [
        {"role": "system", "content": PLAN_PROMPT + "\nСхема JSON: " + json.dumps(PLAN_SCHEMA, ensure_ascii=False, separators=(",", ":"))},
        {"role": "user", "content": json.dumps({"now": now, "previous": previous,
             "user_history": history, "current_request": query}, ensure_ascii=False, separators=(",", ":"))},
    ]


def parse_plan(message, previous=None):
    if message.get("tool_calls"):
        raise ValueError("Unexpected tool call in planner")
    content = (message.get("content") or "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    plan = json.loads(content)
    validate(plan, PLAN_SCHEMA)
    if plan["weather_date"] is not None and date.fromisoformat(plan["weather_date"]).isoformat() != plan["weather_date"]:
        raise ValueError("invalid forecast date")
    if plan["weather_hour"] is not None and plan["weather_date"] is None:
        raise ValueError("weather hour requires date")
    budget = plan["budget"]
    if budget["action"] == "keep":
        plan["budget"] = copy.deepcopy(previous["budget"]) if previous else {
            "action": "keep", "amount": None, "currency": "unknown", "people": budget["people"], "scope": "unknown",
        }
    elif budget["action"] == "clear":
        plan["budget"] = {"action": "clear", "amount": None, "currency": "unknown", "people": None, "scope": "unknown"}
    elif previous:
        for key, empty in (("amount", None), ("currency", "unknown"), ("scope", "unknown")):
            if budget[key] == empty:
                budget[key] = previous["budget"][key]
    if plan["task"] == "search" and not plan["city"].strip():
        plan["task"] = "clarify"
    required = set(plan["required_tools"])
    if plan["task"] in TASK_TO_TOOL:
        required.add(TASK_TO_TOOL[plan["task"]])
    if plan["task"] == "refine" and plan["reference_kind"] == "none" and previous:
        plan["reference_kind"] = {"search":"places", "weather":"weather", "route":"routes"}.get(previous["task"], previous["reference_kind"])
    if plan["task"] in {"chat", "clarify", "refine"}:
        required.clear()
    plan["required_tools"] = sorted(required)
    return plan


def instructions(plan):
    budget = plan["budget"]
    text = "\nПлан текущего хода (условия пользователя, не данные о местах): " + json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
    text += "\nВыполни необходимые инструменты перед итогом. Новую категорию нельзя объявлять отсутствующей по старой выдаче."
    text += "\nВыбирай и объясняй варианты для цели пользователя; не пересказывай нерелевантные свойства карточек."
    if budget["amount"] is not None:
        text += ("\nДенежное ограничение учитывай при отборе: сумма, единица и число людей хранятся в плане. "
                 "Первичный textQuery — короткий тип места + область + город; не перегружай его бюджетом и всеми пожеланиями. "
                 "Разные единицы цены не сравнивай без данных. Если цен нет, сделай не более одного дополнительного "
                 "поиска цен для выбранных кандидатов; без подтверждения не гарантируй бюджет. "
                 "Не пиши «в рамках бюджета», если цена неизвестна: назови варианты кандидатами при заданном ориентире.")
    else:
        text += "\nБюджет не задан: не добавляй шаблон о попадании в бюджет. Цены обсуждай только по просьбе."
    if plan["strict_area"]:
        text += "\nРайон строгий: географию поиска не расширяй."
    else:
        text += "\nЕсли поиск успешен, но places пуст, сделай одну попытку рядом с районом и явно раскрой расширение. Ошибка сервиса не означает пустую выдачу."
    text += "\nЦитируй только данные, использованные в текущем ответе. Объяснения ошибок и уточнения не сопровождай старыми местами."
    return text
