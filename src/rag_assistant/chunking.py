"""Fixed-window chunking baseline for source-attributed RAG sections."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Callable

from rag_assistant.config import ChunkingSettings


TOKEN_PATTERN = re.compile(r"\S+")
ProgressCallback = Callable[[int, int, str], None]
REQUIRED_SECTION_FIELDS = {
    "document_id",
    "section_id",
    "library",
    "document_title",
    "source_url",
    "source_sha256",
    "section_path",
    "text",
}


def chunk_sections(
    input_file: Path,
    output_file: Path,
    settings: ChunkingSettings,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Split extracted sections into overlapping fixed-window chunks."""
    sections = _load_sections(input_file)
    chunks: list[dict[str, Any]] = []
    total_sections = len(sections)
    for completed_sections, section in enumerate(sections, start=1):
        chunks.extend(chunk_section(section, settings))
        if progress_callback is not None:
            progress_callback(completed_sections, total_sections, section["section_id"])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="\n") as destination:
        for chunk in chunks:
            destination.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return chunks


def chunk_section(
    section: dict[str, Any], settings: ChunkingSettings
) -> list[dict[str, Any]]:
    """Create fixed-size, overlapping chunks while preserving source metadata.

    The baseline counts whitespace-separated units. It preserves the original
    characters inside each window, including newlines and code formatting.
    """
    _validate_section(section)
    matches = list(TOKEN_PATTERN.finditer(section["text"]))
    if not matches:
        return []

    chunks: list[dict[str, Any]] = []
    start_index = 0
    chunk_number = 1
    step = settings.chunk_size - settings.chunk_overlap

    while start_index < len(matches):
        end_index = min(start_index + settings.chunk_size, len(matches))
        text = section["text"][matches[start_index].start() : matches[end_index - 1].end()]
        chunks.append(
            {
                "chunk_id": f"{section['section_id']}:chunk:{chunk_number:03d}",
                "parent_section_id": section["section_id"],
                "document_id": section["document_id"],
                "library": section["library"],
                "document_title": section["document_title"],
                "source_url": section["source_url"],
                "source_sha256": section["source_sha256"],
                "section_path": section["section_path"],
                "unit_start": start_index,
                "unit_end": end_index,
                "unit_count": end_index - start_index,
                "text": text,
            }
        )
        if end_index == len(matches):
            break
        start_index += step
        chunk_number += 1
    return chunks


def _load_sections(input_file: Path) -> list[dict[str, Any]]:
    """Read and validate the extracted section JSONL file."""
    sections = [
        json.loads(line)
        for line in input_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not sections:
        raise ValueError("The input sections file is empty.")
    for section in sections:
        _validate_section(section)
    return sections


def _validate_section(section: dict[str, Any]) -> None:
    """Ensure every input section contains traceable source metadata."""
    missing_fields = REQUIRED_SECTION_FIELDS - section.keys()
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Section is missing required fields: {missing}")
    if not isinstance(section["text"], str) or not section["text"].strip():
        raise ValueError("Section text must be a non-empty string.")
