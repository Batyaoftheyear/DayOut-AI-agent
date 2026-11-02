# Доступ к Yandex AI Studio

DayOut использует Alice AI LLM Flash через `https://ai.api.cloud.yandex.net/v1/chat/completions`.

## Что требуется

1. Каталог Yandex Cloud с активным платёжным аккаунтом.
2. API-ключ с правом генерации в этом каталоге.
3. Идентификатор каталога, а не платёжного аккаунта или облака.

ID каталога можно взять из адреса консоли: `https://console.yandex.cloud/folders/<folder-id>`.

## Локальная настройка

Скопировать `.env.example` в `.env` и заполнить:

```dotenv
LLM_PROVIDER=yandex
YANDEX_API_KEY=your-api-key
YANDEX_FOLDER_ID=your-folder-id
YANDEX_MODEL=aliceai-llm-flash
GOOGLE_MAPS_API_KEY=your-maps-key
```

Короткое имя модели преобразуется в `gpt://<folder-id>/aliceai-llm-flash/latest`. Файл нужно сохранить; редактор не передаёт несохранённые значения процессу Python.

```powershell
.\.venv\Scripts\python.exe yandex_check.py
.\.venv\Scripts\python.exe dayout.py --usage
```

`yandex_check.py` получает список моделей и выполняет небольшие генерации с искусственным инструментом. Это платная проверка связи и tool calling; Google Maps в ней не вызывается. Обычный pytest работает без API и оплаты.

## На хостинге

В Streamlit Community Cloud → Settings → Secrets задать те же значения в TOML:

```toml
LLM_PROVIDER = "yandex"
YANDEX_API_KEY = "your-api-key"
YANDEX_FOLDER_ID = "your-folder-id"
YANDEX_MODEL = "aliceai-llm-flash"
GOOGLE_MAPS_API_KEY = "your-maps-key"
```

После изменения Secrets перезапустить приложение. Локальный `.env` не изменяет облачную конфигурацию.

## Расход и ошибки

Одно сообщение пользователя может требовать нескольких генераций: план, выбор инструмента, итоговый ответ, иногда исправление. Флаг CLI `--usage` и детали веб-ответа показывают расход токенов.

Для оценки Flash используется тариф 0,10 ₽/1000 входных и 0,20 ₽/1000 выходных токенов. Все входные токены считаются по полной ставке без скидки за кэш; это ориентир, не счёт биллинга. [Актуальный тариф](https://aistudio.yandex.ru/ru/docs/ai-studio/pricing).

401/403 — проверить ключ и права; 402 — платёжный аккаунт; 429 — квоту соответствующего сервиса. Баланс Яндекса не увеличивает квоту Google Maps. `finish_reason=length` означает предел генерации одного запроса, а не обязательно исчерпание денег.

Запросы отправляются с `x-data-logging-enabled: false`, без API постоянных Conversations/Responses и без записи сырого содержимого в отчёты Eval. Эта настройка не является универсальной гарантией отсутствия хранения или кэширования у провайдера.
