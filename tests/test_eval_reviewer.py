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
