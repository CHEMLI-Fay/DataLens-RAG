"""Offline checks for full answer generation and source labeling."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx

from rag_assistant.config import GenerationSettings
from rag_assistant.generation import (
    _format_context, generate_answer, translate_retrieval_query,
)


ROOT = Path(__file__).resolve().parents[1]
CHUNKS = [
    {
        "section_path": ["Unrelated section"],
        "source_url": "https://example.org/one",
        "text": "Unrelated evidence.",
        "score": 0.9,
    },
    {
        "section_path": ["Imputation of missing values"],
        "source_url": "https://example.org/two",
        "text": "Missing values can be imputed from known data.",
        "score": 0.8,
    },
]


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.settings = GenerationSettings("ollama", "test-model", 0.0, 6000)
        self.client = Mock()

    def answer(self, content: str, reason: str = "stop") -> str:
        self.client.chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content=content), done_reason=reason
        )
        with patch("rag_assistant.generation._load_ollama_client", return_value=self.client), patch(
            "rag_assistant.generation._select_relevant_labels", return_value=[2]
        ):
            return generate_answer(
                "Comment traiter les valeurs manquantes ?", CHUNKS, self.settings,
                ROOT / "prompts/answer.txt", retrieval_query="How to handle missing values?",
            )

    def test_full_answer_is_returned_without_citation_filtering(self):
        answer = "On peut imputer les données manquantes. [2]\nIl faut vérifier la méthode."
        self.assertEqual(self.answer(answer), answer)
        self.client.close.assert_called_once()

    def test_output_limit_keeps_partial_answer_visible(self):
        self.assertIn("Début de réponse", self.answer("Début de réponse", "length"))
        self.assertIn("output limit", self.answer("Début de réponse", "length"))

    def test_context_keeps_original_labels_when_reordered(self):
        context = _format_context(CHUNKS, [2], 500)
        self.assertTrue(context.startswith("[2] Imputation of missing values"))
        self.assertNotIn("Unrelated evidence", context)

    def test_empty_relevance_selection_abstains_without_generation(self):
        with patch("rag_assistant.generation._load_ollama_client", return_value=self.client), patch(
            "rag_assistant.generation._select_relevant_labels", return_value=[]
        ):
            answer = generate_answer(
                "Comment anonymiser mes clients ?", CHUNKS, self.settings,
                ROOT / "prompts/answer.txt",
            )
        self.assertIn("informations suffisantes", answer)
        self.client.chat.assert_not_called()

    def test_french_question_translation(self):
        self.client.chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content="How to handle missing values?")
        )
        with patch("rag_assistant.generation._load_ollama_client", return_value=self.client):
            translated = translate_retrieval_query("Comment traiter les valeurs manquantes ?", self.settings)
        self.assertEqual(translated, "How to handle missing values?")

    def test_timeout_is_reported(self):
        self.client.chat.side_effect = httpx.ReadTimeout("simulated")
        with patch("rag_assistant.generation._load_ollama_client", return_value=self.client):
            with self.assertRaisesRegex(RuntimeError, "ollama ps"):
                generate_answer("How?", CHUNKS, self.settings, ROOT / "prompts/answer.txt")
        self.client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
