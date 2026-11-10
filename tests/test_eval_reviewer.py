import asyncio
import json
import httpx

from eval.reviewer import ModelReviewer
from provider_config import WebConfig


def test_review_uses_evidence_but_persists_only_scores_codes_and_usage(monkeypatch):
    import eval.reviewer as module
    captured = []
    real_client = httpx.AsyncClient
    def handler(request):
        body = json.loads(request.content)
        captured.append(body)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": json.dumps({"scores": {"grounding": "fail"}, "codes": ["unsupported_claim"]})}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 15}})
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(handler)))
    config = WebConfig("synthetic-yandex", "synthetic-maps", "synthetic-model", "yandex", "synthetic-folder")
    records = [{"messages": [{"role": "tool", "content": json.dumps({"summary": "synthetic-evidence", "contentToken": "private-token"})}]}]
    result = ModelReviewer(config)({"criteria": ["grounding"], "expected": "synthetic condition"}, records)
    assert result["scores"] == {"grounding": "fail"} and result["usage"]["requests"] == 1
    assert "synthetic-evidence" in captured[0]["messages"][1]["content"]
    assert "private-token" not in captured[0]["messages"][1]["content"]
    assert "synthetic-evidence" not in json.dumps(result)


def test_reviewer_never_grades_private_drafts_repairs_or_local_context_as_user_dialogue():
    from eval.reviewer import review_records
    records=[{'query':'Подбери павильон и ресторан','answer':'Готовый ответ','messages':[
        {'role':'user','content':'PRIVATE_REPAIR_INSTRUCTION'},
        {'role':'assistant','content':'PRIVATE_REJECTED_DRAFT'},
        {'role':'assistant','content':'PRIVATE_PROGRESS','tool_calls':[
            {'id':'maps','function':{'name':'search_places'}},
            {'id':'local','function':{'name':'remember_context'}}]},
        {'role':'tool','tool_call_id':'maps','content':'{"summary":"actual-evidence","contentToken":"private-token"}'},
        {'role':'tool','tool_call_id':'local','content':'{"response_check":"PRIVATE_MEMORY_CHECK"}'}]}]
    material=json.dumps(review_records(records),ensure_ascii=False)
    assert 'actual-evidence' in material and 'Готовый ответ' in material and 'search_places' in material
    assert 'PRIVATE_' not in material and 'private-token' not in material and 'remember_context' not in material
