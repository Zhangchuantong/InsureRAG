# -*- coding: utf-8 -*-

from fastapi.testclient import TestClient

import api.routes
from api.main import app


def test_query_api_returns_answer_evidence_trace_id(monkeypatch):
    def fake_query_insurance_clause(question, collection=None, top_k=None):
        return {
            "answer": "等待期为90天。",
            "evidence": [
                {
                    "content": "第三条 等待期为90天。",
                    "source": "demo.pdf",
                    "clause_id": "第三条",
                    "parent_id": "demo:p0",
                    "score": 0.9,
                }
            ],
            "trace_id": "trace123",
            "latency_ms": 12.3,
            "collection": collection or "insure_rag",
        }

    monkeypatch.setattr(api.routes, "query_insurance_clause", fake_query_insurance_clause)

    client = TestClient(app)
    response = client.post(
        "/api/query",
        json={"question": "等待期多久？", "collection": "insure_rag", "top_k": 1},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"]
    assert data["evidence"]
    assert data["trace_id"] == "trace123"
