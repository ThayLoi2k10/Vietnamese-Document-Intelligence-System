# src/rag/evaluation/backends/rule_based_evaluator.py
"""Triển khai các bộ đánh giá dựa trên luật và đối soát mặt đất (Rule-based Evaluators).

Tuân thủ Mục 17.4: Generation evaluation interfaces & Mục 22.1: Test philosophy (không đòi hỏi LLM inference).
"""

from typing import Set

from src.rag.evaluation.interfaces.evaluator import (
    AbstentionCorrectnessEvaluator,
    AnswerRelevanceEvaluator,
    CitationPrecisionEvaluator,
    CitationRecallEvaluator,
    FaithfulnessEvaluator,
    UnsupportedClaimEvaluator,
)
from src.rag.evaluation.metrics.models import EvaluationInput
from src.rag.output.schemas.response_schema import ConfidenceLevel


class RuleBasedCitationPrecisionEvaluator(CitationPrecisionEvaluator):
    """Tính toán tỷ lệ các trích dẫn trong câu trả lời có mặt trong ground truth."""

    def evaluate_sample(self, item: EvaluationInput) -> float:
        citations = item.rag_response.legal_answer.citations
        if not citations:
            return 1.0 if not item.ground_truth_law_ids else 0.0

        gt_laws = {str(lid).strip().lower() for lid in item.ground_truth_law_ids}
        gt_arts = {int(aid) for aid in item.ground_truth_article_ids}

        correct_count = 0
        for c in citations:
            is_correct = False
            if c.global_article_id is not None and c.global_article_id in gt_arts:
                is_correct = True
            elif c.law_id and c.law_id.strip().lower() in gt_laws:
                is_correct = True
            if is_correct:
                correct_count += 1

        return correct_count / len(citations)


class RuleBasedCitationRecallEvaluator(CitationRecallEvaluator):
    """Tính toán tỷ lệ các căn cứ trong ground truth được trích dẫn trong câu trả lời."""

    def evaluate_sample(self, item: EvaluationInput) -> float:
        gt_laws = {str(lid).strip().lower() for lid in item.ground_truth_law_ids}
        gt_arts = {int(aid) for aid in item.ground_truth_article_ids}

        if not gt_laws and not gt_arts:
            return 1.0

        cited_laws = {
            c.law_id.strip().lower()
            for c in item.rag_response.legal_answer.citations
            if c.law_id
        }
        cited_arts = {
            c.global_article_id
            for c in item.rag_response.legal_answer.citations
            if c.global_article_id is not None
        }

        matched_count = 0
        total_gt = max(len(gt_arts), len(gt_laws))

        if gt_arts:
            matched_count = len(gt_arts & cited_arts)
        elif gt_laws:
            matched_count = len(gt_laws & cited_laws)

        return matched_count / total_gt if total_gt > 0 else 1.0


class RuleBasedAbstentionCorrectnessEvaluator(AbstentionCorrectnessEvaluator):
    """Đánh giá tính chính xác của quyết định từ chối:

    - Nếu is_answerable=True -> Quyết định đúng khi status != INSUFFICIENT
    - Nếu is_answerable=False -> Quyết định đúng khi status == INSUFFICIENT
    """

    def evaluate_sample(self, item: EvaluationInput) -> float:
        is_answerable = item.is_answerable
        decision_status = item.rag_response.abstention_decision.status

        if not is_answerable:
            # Câu hỏi không thể trả lời được: hệ thống nên từ chối
            return 1.0 if decision_status == ConfidenceLevel.INSUFFICIENT else 0.0
        else:
            # Câu hỏi trả lời được: hệ thống không nên từ chối toàn bộ
            return 1.0 if decision_status != ConfidenceLevel.INSUFFICIENT else 0.0


class HeuristicFaithfulnessEvaluator(FaithfulnessEvaluator):
    """Đánh giá độ trung thực dựa trên độ bao phủ từ vựng (Lexical Grounding)."""

    def evaluate_sample(self, item: EvaluationInput) -> float:
        answer_text = item.rag_response.legal_answer.answer
        if not answer_text or not answer_text.strip():
            return 0.0

        # Lấy toàn bộ từ trong ngữ cảnh
        context_words: Set[str] = set()
        for ctx_item in item.rag_response.assembled_context.items:
            context_words.update(ctx_item.text.lower().split())

        if not context_words:
            return 0.0

        answer_words = [w for w in answer_text.lower().split() if len(w) > 2]
        if not answer_words:
            return 1.0

        grounded_count = sum(1 for w in answer_words if w in context_words)
        return grounded_count / len(answer_words)
