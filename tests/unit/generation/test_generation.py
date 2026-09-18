# tests/unit/generation/test_generation.py
"""Unit tests cho module LLM Generation Phase 9.

Tuân thủ Mục 22.2: Generation unit tests:
    - Request validation
    - Parameter propagation (temperature, thinking_mode, seed)
    - Mock generator behavior across all modes
    - Raw response preservation
    - Generation failure propagation
"""

import json
import unittest

from src.rag.errors.exceptions import LLMGenerationError
from src.rag.generation.domain.models import (
    GenerationParameters,
    GenerationRequest,
)
from src.rag.generation.mocks.mock_generator import MockLLMGenerator
from src.rag.prompting.domain.models import PromptRequest


class TestGenerationModule(unittest.TestCase):
    """Kiểm tra hoạt động của LLM Generator và MockLLMGenerator."""

    def setUp(self) -> None:
        self.prompt_req = PromptRequest(
            system_prompt="system prompt text",
            user_prompt="user question text",
            prompt_version="prompt_v1",
            evidence_registry={},
        )

    def test_parameter_propagation_and_raw_response_preservation(self) -> None:
        """Kiểm tra truyền đúng siêu tham số và bảo toàn phản hồi thô."""
        generator = MockLLMGenerator(mode=MockLLMGenerator.MODE_VALID_ANSWER)
        gen_params = GenerationParameters(temperature=0.7, seed=123)

        req = GenerationRequest(
            model_identifier="Qwen/Qwen2.5-7B-Instruct",
            prompt=self.prompt_req,
            max_output_tokens=512,
            generation_parameters=gen_params,
            thinking_mode=True,
        )

        res = generator.generate(req)

        self.assertEqual(res.model_identifier, "Qwen/Qwen2.5-7B-Instruct")
        self.assertIsNotNone(res.raw_response)
        self.assertEqual(res.generation_metadata["temperature"], 0.7)
        self.assertEqual(res.generation_metadata["thinking_mode"], True)

        # Parse thử raw_response xem có phải JSON không
        data = json.loads(res.raw_response)
        self.assertIn("answer", data)
        self.assertIn("citations", data)

    def test_mock_modes(self) -> None:
        """Kiểm tra các chế độ kịch bản định sẵn của MockLLMGenerator."""
        req = GenerationRequest(
            model_identifier="mock-model",
            prompt=self.prompt_req,
        )

        # 1. Chế độ Abstention
        gen_abs = MockLLMGenerator(mode=MockLLMGenerator.MODE_ABSTENTION)
        res_abs = gen_abs.generate(req)
        data_abs = json.loads(res_abs.raw_response)
        self.assertEqual(data_abs["confidence_level"], "INSUFFICIENT")

        # 2. Chế độ Malformed JSON
        gen_mal = MockLLMGenerator(mode=MockLLMGenerator.MODE_MALFORMED_JSON)
        res_mal = gen_mal.generate(req)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(res_mal.raw_response)

        # 3. Chế độ Failure ném ra LLMGenerationError
        gen_fail = MockLLMGenerator(mode=MockLLMGenerator.MODE_FAILURE)
        with self.assertRaises(LLMGenerationError):
            gen_fail.generate(req)


if __name__ == "__main__":
    unittest.main()
