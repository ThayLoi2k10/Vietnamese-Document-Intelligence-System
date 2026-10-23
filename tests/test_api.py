"""Unit and integration tests for FastAPI REST API endpoints."""

import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.routes import set_rag_service
from src.api.service import LegalRAGService


class TestLegalRAGAPI(unittest.TestCase):
    def setUp(self):
        # Create a mock lightweight service
        self.mock_service = MagicMock(spec=LegalRAGService)
        self.mock_service.start_time = 1000.0
        self.mock_service.corpus = {
            101: {
                "law_id": "45/2019/QH14",
                "aid": "Điều 25",
                "text": "Thời gian thử việc do hai bên thỏa thuận căn cứ vào tính chất công việc...",
            },
            102: {
                "law_id": "45/2019/QH14",
                "aid": "Điều 35",
                "text": "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động nhưng phải báo trước ít nhất 45 ngày...",
            },
        }
        self.mock_service.bm25_retriever = MagicMock()
        self.mock_service.dense_retriever = MagicMock()
        self.mock_service.reranker = MagicMock()

        # Wire mock methods
        def mock_get_doc(aid: int):
            doc = self.mock_service.corpus.get(aid)
            if not doc:
                return None
            return {
                "article_id": aid,
                "law_id": doc["law_id"],
                "aid": doc["aid"],
                "text": doc["text"],
            }

        self.mock_service.get_document.side_effect = mock_get_doc

        def mock_search(query: str, retriever_type: str = "hybrid_rerank", top_k: int = 5):
            docs = [
                {
                    "rank": 1,
                    "article_id": 102,
                    "score": 0.95,
                    "law_id": "45/2019/QH14",
                    "aid": "Điều 35",
                    "text": self.mock_service.corpus[102]["text"],
                }
            ]
            return docs, query, 12.5

        self.mock_service.search.side_effect = mock_search

        def mock_answer_query(**kwargs):
            return {
                "status": "SUCCESS",
                "query": kwargs.get("query", ""),
                "rewritten_query": None,
                "detected_acronyms": {"HĐLĐ": "hợp đồng lao động"},
                "answer": (
                    "### 🎯 1. KẾT LUẬN TRỰC DIỆN\nBáo trước ít nhất 45 ngày.\n\n"
                    "### ⚖️ 2. CĂN CỨ PHÁP LÝ\nCăn cứ Điều 35 Bộ luật Lao động 2019 (45/2019/QH14).\n\n"
                    "### 📋 3. LƯU Ý & ĐIỀU KIỆN ÁP DỤNG\nÁp dụng cho HĐLĐ không xác định thời hạn."
                ),
                "direct_conclusion": "Báo trước ít nhất 45 ngày.",
                "legal_grounds": "Căn cứ Điều 35 Bộ luật Lao động 2019 (45/2019/QH14).",
                "notes_and_conditions": "Áp dụng cho HĐLĐ không xác định thời hạn.",
                "is_abstention": False,
                "grounding": {
                    "score": 1.0,
                    "verdict": "VERIFIED",
                    "badge": "🟢 [XÁC MINH CĂN CỨ: 100% - HOÀN TOÀN TIN CẬY]",
                    "summary": "Tất cả căn cứ đã được kiểm chứng đối chiếu với dữ liệu luật.",
                    "cited_laws": ["45/2019/QH14"],
                    "cited_articles": ["Điều 35"],
                    "missing_citations": [],
                    "ungrounded_citations": [],
                },
                "sources": [
                    {
                        "rank": 1,
                        "article_id": 102,
                        "score": 0.95,
                        "law_id": "45/2019/QH14",
                        "aid": "Điều 35",
                        "text": self.mock_service.corpus[102]["text"],
                    }
                ],
                "latency": {"search_ms": 12.5, "generation_ms": 250.0, "total_ms": 265.0},
            }

        self.mock_service.answer_query.side_effect = mock_answer_query

        set_rag_service(self.mock_service)
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_root_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")

    def test_api_v1_health(self):
        resp = self.client.get("/api/v1/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["total_articles"], 2)
        self.assertTrue(data["indices_loaded"]["bm25"])

    def test_get_document_success(self):
        resp = self.client.get("/api/v1/documents/102")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["article_id"], 102)
        self.assertEqual(data["law_id"], "45/2019/QH14")
        self.assertIn("45 ngày", data["text"])

    def test_get_document_not_found(self):
        resp = self.client.get("/api/v1/documents/999999")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("Không tìm thấy", resp.json()["detail"])

    def test_search_endpoint(self):
        payload = {
            "query": "thời gian báo trước nghỉ việc",
            "retriever_type": "hybrid_rerank",
            "top_k": 3,
        }
        resp = self.client.post("/api/v1/search", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_results"], 1)
        self.assertEqual(data["documents"][0]["article_id"], 102)
        self.assertIn("latency_ms", data)

    def test_search_validation_empty_query(self):
        payload = {"query": "", "top_k": 5}
        resp = self.client.post("/api/v1/search", json=payload)
        self.assertEqual(resp.status_code, 422)

    def test_query_endpoint(self):
        payload = {
            "query": "NLĐ đơn phương chấm dứt HĐLĐ phải báo trước bao nhiêu ngày?",
            "backend": "colab",
            "colab_url": "https://dummy-colab.trycloudflare.com",
            "top_k": 5,
        }
        resp = self.client.post("/api/v1/query", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("45 ngày", data["direct_conclusion"])
        self.assertEqual(data["grounding"]["verdict"], "VERIFIED")
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual(data["detected_acronyms"]["HĐLĐ"], "hợp đồng lao động")


if __name__ == "__main__":
    unittest.main()
