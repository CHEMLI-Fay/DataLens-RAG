"""Batch evaluation for retrieval and locally generated answers."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from rag_assistant.config import EmbeddingSettings, GenerationSettings, RetrievalSettings
from rag_assistant.generation import generate_answer, translate_retrieval_query
from rag_assistant.retrieval import RetrievalData, search_chunks


CITATION_PATTERN = re.compile(r"\[(\d+)\]")
ABSTENTION_MARKERS = (
    "insufficient", "not enough information", "cannot answer", "can't answer",
    "pas suffisamment", "informations insuffisantes", "ne permet pas de répondre",
)


def evaluate_questions(
    questions_file: Path,
    output_directory: Path,
    chunks_file: Path,
    embedding_file: Path,
    index_file: Path,
    prompt_file: Path,
    embedding_settings: EmbeddingSettings,
    retrieval_settings: RetrievalSettings,
    generation_settings: GenerationSettings,
    embedding_model: Any,
    retrieval_data: RetrievalData,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Run every question while reusing loaded retrieval artifacts."""
    questions = _load_questions(questions_file)
    results: list[dict[str, Any]] = []
    for position, item in enumerate(questions, start=1):
        if progress_callback:
            progress_callback(position, len(questions), item["id"])
        started = perf_counter()
        answer = ""
        error = ""
        retrieved: list[dict[str, Any]] = []
        try:
            retrieval_query = translate_retrieval_query(item["question"], generation_settings)
            retrieved = search_chunks(
                query=retrieval_query,
                chunks_file=chunks_file,
                embedding_file=embedding_file,
                index_file=index_file,
                embedding_settings=embedding_settings,
                retrieval_settings=retrieval_settings,
                embedding_model=embedding_model,
                retrieval_data=retrieval_data,
            )
            answer = generate_answer(
                question=item["question"],
                retrieved_chunks=retrieved,
                settings=generation_settings,
                prompt_path=prompt_file,
                retrieval_query=retrieval_query,
            )
        except (OSError, ValueError, RuntimeError) as caught:
            error = str(caught)

        document_ids = [chunk["document_id"] for chunk in retrieved]
        expected = item["expected_document_ids"]
        expected_ranks = [
            rank for rank, document_id in enumerate(document_ids, start=1)
            if document_id in expected
        ]
        citations = [int(label) for label in CITATION_PATTERN.findall(answer)]
        citations_valid = bool(citations) and all(1 <= label <= len(retrieved) for label in citations)
        abstained = any(marker in answer.lower() for marker in ABSTENTION_MARKERS)
        retrieval_pass = bool(expected_ranks) if item["answerable"] else not expected
        answer_pass = (
            bool(answer) and citations_valid and not abstained
            if item["answerable"] else abstained
        )
        results.append(
            {
                **item,
                "expected_rank": min(expected_ranks) if expected_ranks else "",
                "top_score": round(float(retrieved[0]["score"]), 4) if retrieved else "",
                "retrieval_pass": retrieval_pass,
                "retrieved_document_ids": " | ".join(document_ids),
                "retrieved_sources": " | ".join(chunk["source_url"] for chunk in retrieved),
                "answer": answer,
                "answer_word_count": len(answer.split()),
                "citations_valid": citations_valid,
                "abstained": abstained,
                "answer_check_pass": answer_pass,
                "automatic_status": "PASS" if retrieval_pass and answer_pass and not error else "REVIEW",
                "duration_seconds": round(perf_counter() - started, 2),
                "error": error,
            }
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "evaluation_results.csv", results)
    _write_report(output_directory / "evaluation_report.md", results)
    return results


def _load_questions(path: Path) -> list[dict[str, Any]]:
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(questions) != 15:
        raise ValueError(f"Expected exactly 15 evaluation questions, found {len(questions)}.")
    return questions


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = list(results[0])
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = dict(result)
            row["expected_document_ids"] = " | ".join(row["expected_document_ids"])
            writer.writerow(row)


def _write_report(path: Path, results: list[dict[str, Any]]) -> None:
    total = len(results)
    retrieval_passes = sum(bool(result["retrieval_pass"]) for result in results)
    answer_passes = sum(bool(result["answer_check_pass"]) for result in results)
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- Questions: {total}",
        f"- Retrieval checks passed: {retrieval_passes}/{total}",
        f"- Answer format checks passed: {answer_passes}/{total}",
        "- Automatic checks cover expected-source retrieval, citation labels, abstention, and execution errors.",
        "- They do not prove factual correctness or complete support; answers marked PASS still require human review.",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"## {result['id']} — {result['automatic_status']}",
                "",
                f"**Question:** {result['question']}",
                "",
                f"**Expected documents:** {', '.join(result['expected_document_ids']) or 'Out of corpus'}",
                "",
                f"**Expected source rank:** {result['expected_rank'] or 'Not retrieved'}",
                "",
                f"**Checks:** retrieval={result['retrieval_pass']}, citations={result['citations_valid']}, "
                f"abstained={result['abstained']}, duration={result['duration_seconds']}s",
                "",
                "**Answer:**",
                "",
                result["answer"] or f"ERROR: {result['error']}",
                "",
                "**Retrieved sources:**",
                "",
            ]
        )
        lines.extend(f"- {source}" for source in result["retrieved_sources"].split(" | ") if source)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
