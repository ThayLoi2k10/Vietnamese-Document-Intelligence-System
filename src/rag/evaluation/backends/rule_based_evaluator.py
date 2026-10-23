"""Rule-based evaluators calculating grounded precision, recall, and abstention correctness."""

from typing import Set

from src.rag.evaluation.interfaces.evaluator import (
    AbstentionCorrectnessEvaluator,
    CitationPrecisionEvaluator,
    CitationRecallEvaluator,
    FaithfulnessEvaluator,
)
from src.rag.evaluation.metrics.models import EvaluationInput
from src.rag.output.schemas.response_schema import ConfidenceLevel


class RuleBasedCitationPrecisionEvaluator(CitationPrecisionEvaluator):
    """Calculates precision of answer citations against ground truth."""

    def evaluate_sample(self, item: EvaluationInput) -> float:
        citations = item.rag_response.legal_answer.citations
        if not citations:
            return 1.0 if not item.ground_truth_law_ids and not item.ground_truth_article_ids else 0.0

        gt_laws = {str(lid).strip().lower() for lid in item.ground_truth_law_ids}
        gt_arts = {int(aid) for aid in item.ground_truth_article_ids}

        context_by_number = {
            ctx.context_position: ctx
            for ctx in item.rag_response.assembled_context.items
        }

        correct_count = 0
        for c in citations:
            is_correct = False
            ctx = context_by_number.get(c.citation_number)
            if ctx is not None:
                if ctx.global_article_id is not None and ctx.global_article_id in gt_arts:
                    is_correct = True
                elif ctx.law_id and ctx.law_id.strip().lower() in gt_laws:
                    is_correct = True

            if not is_correct:
                if c.global_article_id is not None and c.global_article_id in gt_arts:
                    is_correct = True
                elif c.law_id and c.law_id.strip().lower() in gt_laws:
                    is_correct = True

            if is_correct:
                correct_count += 1

        return correct_count / len(citations)


class RuleBasedCitationRecallEvaluator(CitationRecallEvaluator):
    """Calculates recall of ground truth evidence covered by citations."""

    def evaluate_sample(self, item: EvaluationInput) -> float:
        gt_laws = {str(lid).strip().lower() for lid in item.ground_truth_law_ids}
        gt_arts = {int(aid) for aid in item.ground_truth_article_ids}

        if not gt_laws and not gt_arts:
            return 1.0

        context_by_number = {
            ctx.context_position: ctx
            for ctx in item.rag_response.assembled_context.items
        }

        cited_arts = set()
        cited_laws = set()

        for c in item.rag_response.legal_answer.citations:
            ctx = context_by_number.get(c.citation_number)
            if ctx is not None:
                if ctx.global_article_id is not None:
                    cited_arts.add(ctx.global_article_id)
                if ctx.law_id:
                    cited_laws.add(ctx.law_id.strip().lower())
            else:
                if c.global_article_id is not None:
                    cited_arts.add(c.global_article_id)
                if c.law_id:
                    cited_laws.add(c.law_id.strip().lower())

        if gt_arts:
            matched_count = len(gt_arts & cited_arts)
            total_gt = len(gt_arts)
        elif gt_laws:
            matched_count = len(gt_laws & cited_laws)
            total_gt = len(gt_laws)
        else:
            return 1.0

        return matched_count / total_gt if total_gt > 0 else 1.0


class RuleBasedAbstentionCorrectnessEvaluator(AbstentionCorrectnessEvaluator):
    """Evaluates whether abstention aligns with answerability."""

    def evaluate_sample(self, item: EvaluationInput) -> float:
        is_answerable = item.is_answerable
        decision_status = item.rag_response.abstention_decision.status

        if not is_answerable:
            return 1.0 if decision_status == ConfidenceLevel.INSUFFICIENT else 0.0
        return 1.0 if decision_status != ConfidenceLevel.INSUFFICIENT else 0.0


class HeuristicFaithfulnessEvaluator(FaithfulnessEvaluator):
    """Evaluates lexical grounding of the answer against context items."""

    def evaluate_sample(self, item: EvaluationInput) -> float:
        answer_text = item.rag_response.legal_answer.answer
        if not answer_text or not answer_text.strip():
            return 0.0

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
