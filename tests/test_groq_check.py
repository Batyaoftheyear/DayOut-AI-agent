import json

import httpx
import pytest

import groq_check as check


def demo_call(**changes):
    call = {"id": "call-demo", "type": "function", "function": {
        "name": "get_demo_activity", "arguments": json.dumps({"city": "Москва"}),
    }}
    call.update(changes)
    return call


def test_complete_tool_round_trip():
    requests = []
    def handler(request):
        assert request.headers["Authorization"] == "Bearer synthetic-key"
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "synthetic-model"}]})
        body = json.loads(request.content)
        assert body["model"] == "synthetic-model"
        if body["tool_choice"] == "auto":
            message = {"role": "assistant", "content": None, "tool_calls": [demo_call()]}
        else:
            assert body["messages"][-1]["tool_call_id"] == "call-demo"
            assert json.loads(body["messages"][-1]["content"])["synthetic"] is True
            message = {"role": "assistant", "content": "Москва: вымышленная активность, DEMO-42."}
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "stop"}]})
    output = []
    with httpx.Client(transport=httpx.MockTransport(handler), headers={"Authorization": "Bearer synthetic-key"}) as client:
        check.check_groq(client, "synthetic-model", False, output.append)
    assert len(requests) == 3
    assert "DEMO-42" in "\n".join(output)


@pytest.mark.parametrize("list_only", [False, True])
def test_no_generation_for_missing_model_or_list_only(list_only):
    def handler(request):
        assert request.method == "GET"
        return httpx.Response(200, json={"data": [{"id": "available-model"}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        if list_only:
            check.check_groq(client, "missing-model", True, lambda _: None)
        else:
            with pytest.raises(check.GroqCheckError, match="Автозамены нет"):
                check.check_groq(client, "missing-model", False, lambda _: None)


@pytest.mark.parametrize("arguments", ['{"city":"Париж"}', '{"city":"Москва","extra":1}', 'not-json'])
def test_invalid_arguments_are_not_executed(arguments):
    call = demo_call()
    call["function"]["arguments"] = arguments
    with pytest.raises(check.GroqCheckError):
        check.execute_demo_call(call)


def test_unknown_tool_is_not_executed():
    call = demo_call()
    call["function"]["name"] = "run_arbitrary_code"
    with pytest.raises(check.GroqCheckError, match="неизвестный инструмент"):
        check.execute_demo_call(call)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500])
def test_http_error_hides_response_and_credentials(status):
    response = httpx.Response(status, request=httpx.Request("GET", check.API_URL), text="synthetic-secret")
    error = httpx.HTTPStatusError("synthetic-secret", request=response.request, response=response)
    message = check.error_message(error)
    assert str(status) in message
    assert "synthetic-secret" not in message


@pytest.mark.parametrize("code,expected", [
    ("tool_use_failed", "сгенерированный вызов"),
    ("tool_call_validation_failed", "не прошёл проверку"),
    ("model_not_found", "Модель не найдена"),
    ("context_length_exceeded", "размер контекста"),
    ("synthetic-secret", "точная причина"),
    (None, "точная причина"),
    ({"unexpected": "synthetic-secret"}, "точная причина"),
])
def test_request_error_uses_only_known_codes(code, expected):
    response = httpx.Response(400, request=httpx.Request("POST", check.API_URL), json={"error": {
        "code": code, "message": "synthetic-secret", "failed_generation": "synthetic-secret",
        "type": "synthetic-secret",
    }})
    error = httpx.HTTPStatusError("synthetic-secret", request=response.request, response=response)
    message = check.error_message(error)
    assert "Groq HTTP 400" in message
    assert expected in message
    assert "synthetic-secret" not in message


@pytest.mark.parametrize("body", [[], {"error": "synthetic-secret"}, {"error": []}])
def test_request_error_handles_unexpected_json(body):
    response = httpx.Response(400, request=httpx.Request("POST", check.API_URL), json=body)
    error = httpx.HTTPStatusError("synthetic-secret", request=response.request, response=response)
    assert "точная причина" in check.error_message(error)
    assert "synthetic-secret" not in check.error_message(error)


def test_missing_key_before_connection(monkeypatch, tmp_path):
    monkeypatch.setattr(check, "ENV_PATH", tmp_path / ".env")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(check.GroqCheckError, match="GROQ_API_KEY"):
        check.load_config()


def test_truncated_completion_is_not_success():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {
            "role": "assistant", "content": "truncated",
        }}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(check.GroqCheckError, match="обрезан"):
            check.completion(client, "synthetic-model", [])


@pytest.mark.parametrize("kind", ["TPD", "TPM", "RPD", "RPM", "ITPM", "OTPM"])
def test_quota_error_reports_kind_and_retry_after_without_provider_text(kind):
    response = httpx.Response(429, request=httpx.Request("POST", check.API_URL),
                             headers={"retry-after": "139.1"}, json={"error": {
                                 "message": f"Rate limit ({kind}): synthetic-secret", "code": "rate_limit_exceeded",
                             }})
    error = httpx.HTTPStatusError("synthetic-secret", request=response.request, response=response)
    text = check.error_message(error)
    assert check.quota_details(response) == (kind, 140)
    assert kind in text and "140 с" in text and "synthetic-secret" not in text


@pytest.mark.parametrize("value", ["NaN", "inf", "-1", "604801", "synthetic-secret", "1e9", ""])
def test_quota_diagnostic_rejects_untrusted_retry_after(value):
    response = httpx.Response(429, request=httpx.Request("POST", check.API_URL),
                             headers={"retry-after": value}, json={"error": "synthetic-secret"})
    error = httpx.HTTPStatusError("synthetic-secret", request=response.request, response=response)
    assert check.quota_details(response) == (None, None)
    assert "тип квоты не указан" in check.error_message(error)
    assert "synthetic-secret" not in check.error_message(error)
