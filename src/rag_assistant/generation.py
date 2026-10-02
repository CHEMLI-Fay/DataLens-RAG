"""Grounded local answer generation from retrieved source chunks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx

from rag_assistant.config import GenerationSettings


FRENCH_QUESTION = re.compile(r"[éèêàùç]|\b(?:comment|pourquoi|quels?|quelles?|valeurs|modèle|entraîner)\b", re.IGNORECASE)
GENERIC_QUERY_TERMS = {
    "data", "with", "from", "into", "while", "what", "when", "which", "should",
    "before", "after", "using", "without", "keep", "keeping", "control", "controlling",
}
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I cannot provide a source-grounded answer from the retrieved documentation."
)


def generate_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    settings: GenerationSettings,
    prompt_path: Path,
    retrieval_query: str | None = None,
) -> str:
    """Ask a local model to answer using only retrieved source chunks."""
    if not question.strip():
        raise ValueError("The question must not be empty.")
    if not retrieved_chunks:
        return "I do not have sufficient information in the selected documentation."

    client = _load_ollama_client(settings.timeout_seconds)
    import ollama

    try:
        labels = _select_relevant_labels(
            client, question, retrieval_query or question, retrieved_chunks, settings
        )
        if not labels:
            return (
                "Je ne dispose pas d'informations suffisantes dans la documentation sélectionnée."
                if FRENCH_QUESTION.search(question) else INSUFFICIENT_EVIDENCE_MESSAGE
            )
        context = _format_context(
            retrieved_chunks, labels, settings.max_context_characters
        )
        prompt_template = prompt_path.read_text(encoding="utf-8")
        prompt = prompt_template.format(context=context, question=question)
        response = client.chat(
            model=settings.model_name,
            messages=[{"role": "user", "content": prompt}],
            keep_alive=settings.keep_alive_seconds,
            options={
                "temperature": settings.temperature,
                "num_predict": settings.max_output_tokens,
            },
        )
        answer = response.message.content.strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty answer.")
        if not any(f"[{label}]" in answer for label in labels):
            source_list = " ".join(f"[{label}]" for label in labels)
            prefix = "Sources utilisées" if FRENCH_QUESTION.search(question) else "Sources used"
            answer = f"{answer}\n\n{prefix}: {source_list}"
        if response.done_reason == "length":
            return answer + "\n\n[Answer stopped at the configured output limit.]"
        return answer
    except httpx.TimeoutException as error:
        raise RuntimeError(
            "Ollama stopped responding within the configured network timeout. "
            "Check model activity with: ollama ps"
        ) from error
    except (ConnectionError, httpx.NetworkError) as error:
        raise RuntimeError("Cannot reach Ollama. Start the Ollama application and retry.") from error
    except ollama.ResponseError as error:
        raise RuntimeError(f"Ollama error: {error.error}") from error
    finally:
        client.close()


def translate_retrieval_query(question: str, settings: GenerationSettings) -> str:
    """Translate French questions for retrieval over English documentation."""
    if not FRENCH_QUESTION.search(question):
        return question
    client = _load_ollama_client(settings.timeout_seconds)
    try:
        response = client.chat(
            model=settings.model_name,
            messages=[{"role": "user", "content": (
                "Translate this data science question to English. "
                "Output only the translation.\n" + question
            )}],
            keep_alive=settings.keep_alive_seconds,
            options={"temperature": 0, "num_predict": 80},
        )
        translated = response.message.content.strip().strip('"')
        return translated if translated and len(translated) < 300 else question
    finally:
        client.close()


def _format_context(
    chunks: list[dict[str, Any]], labels: list[int], max_characters: int
) -> str:
    """Format only passages selected as relevant, preserving result labels."""
    allowance = max_characters // len(labels)
    passages = []
    for label in labels:
        chunk = chunks[label - 1]
        heading = " > ".join(chunk["section_path"])
        header = f"[{label}] {heading}\nSource: {chunk['source_url']}\n"
        passages.append(header + chunk["text"][:max(0, allowance - len(header))])
    return "\n\n".join(passages)


def _select_relevant_labels(
    client: Any,
    question: str,
    retrieval_query: str,
    chunks: list[dict[str, Any]],
    settings: GenerationSettings,
) -> list[int]:
    """Select passages that directly contain evidence, allowing abstention."""
    candidates = []
    for label, chunk in enumerate(chunks, start=1):
        heading = " > ".join(chunk["section_path"])
        preview = " ".join(chunk["text"].split())[:700]
        candidates.append(f"[{label}] {heading}\n{preview}")
    prompt = (
        "Select only passages that directly contain evidence needed to answer the question. "
        "Return an empty list when the question cannot be answered from these passages. "
        "Do not select a passage merely because it shares general words with the question.\n\n"
        f"Question: {question}\nRetrieval query: {retrieval_query}\n\nPassages:\n"
        + "\n\n".join(candidates)
    )
    response = client.chat(
        model=settings.model_name,
        messages=[{"role": "user", "content": prompt}],
        format={
            "type": "object",
            "properties": {
                "relevant_labels": {
                    "type": "array", "items": {"type": "integer"}, "maxItems": 3
                }
            },
            "required": ["relevant_labels"],
        },
        keep_alive=settings.keep_alive_seconds,
        options={"temperature": 0, "num_predict": 100},
    )
    try:
        labels = json.loads(response.message.content)["relevant_labels"]
    except (ValueError, KeyError, TypeError, AttributeError):
        return []
    if not isinstance(labels, list):
        return []
    selected = list(dict.fromkeys(
        label for label in labels if isinstance(label, int) and 1 <= label <= len(chunks)
    ))[:3]
    query_terms = {
        term.rstrip("s") for term in re.findall(r"[a-z]{3,}", retrieval_query.lower())
        if term not in GENERIC_QUERY_TERMS
    }
    overlaps: list[tuple[int, int]] = []
    for label, chunk in enumerate(chunks, start=1):
        searchable = " ".join(chunk["section_path"]) + " " + chunk["text"]
        passage_terms = {
            term.rstrip("s") for term in re.findall(r"[a-z]{3,}", searchable.lower())
        }
        overlaps.append((len(query_terms & passage_terms), label))
    lexical = [label for score, label in sorted(overlaps, reverse=True) if score][:3]
    return list(dict.fromkeys(lexical + selected))[:3]


def _choose_source(chunks: list[dict[str, Any]], retrieval_query: str) -> tuple[int, dict[str, Any]]:
    """Prefer a section whose own heading names the queried topic."""
    query_terms = set(re.findall(r"[a-z]{3,}", retrieval_query.lower()))

    def rank(position: int) -> tuple[int, float]:
        chunk = chunks[position]
        leaf_heading = chunk["section_path"][-1].lower()
        heading_terms = set(re.findall(r"[a-z]{3,}", leaf_heading))
        return len(query_terms & heading_terms), float(chunk.get("score", 0))

    position = max(range(len(chunks)), key=rank)
    return position + 1, chunks[position]


def _load_ollama_client(timeout_seconds: float = 120.0) -> Any:
    """Import the optional local-generation dependency only when needed."""
    try:
        import ollama
    except ImportError as error:
        raise RuntimeError(
            "Install generation dependencies with: python -m pip install -e '.[generation]'"
        ) from error
    return ollama.Client(timeout=httpx.Timeout(timeout_seconds, connect=10.0))
