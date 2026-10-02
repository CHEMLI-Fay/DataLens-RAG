"""Extract source-attributed sections from downloaded documentation HTML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag


HEADING_LEVELS = {f"h{level}": level for level in range(1, 5)}
CONTENT_TAGS = set(HEADING_LEVELS) | {"p", "li", "pre"}


def extract_sources(input_directory: Path, output_file: Path) -> list[dict[str, Any]]:
    """Extract sections from every HTML file referenced by a raw corpus manifest."""
    manifest_path = input_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_records = manifest.get("sources")
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("The corpus manifest must contain a non-empty sources list.")

    sections: list[dict[str, Any]] = []
    for source_record in source_records:
        file_name = source_record.get("file_name")
        if not isinstance(file_name, str) or not file_name.endswith(".html"):
            raise ValueError("Each source record requires an HTML file_name.")

        html = (input_directory / file_name).read_bytes()
        sections.extend(_extract_document(html, source_record))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="\n") as destination:
        for section in sections:
            destination.write(json.dumps(section, ensure_ascii=False) + "\n")
    return sections


def _extract_document(html: bytes, source_record: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn one documentation article into heading-aware sections."""
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one(".bd-article") or soup.select_one("article") or soup.select_one("main")
    if article is None:
        raise ValueError(f"No article content found for {source_record['id']}.")

    for unwanted in article.select("script, style, nav, aside, footer"):
        unwanted.decompose()

    heading_path: dict[int, str] = {}
    sections: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_path: list[str] = [source_record["title"]]

    def save_current_section() -> None:
        text = "\n\n".join(current_parts).strip()
        if not text:
            return
        section_number = len(sections) + 1
        sections.append(
            {
                "document_id": source_record["id"],
                "section_id": f"{source_record['id']}:{section_number:03d}",
                "library": source_record["library"],
                "document_title": source_record["title"],
                "source_url": source_record["retrieved_url"],
                "source_sha256": source_record["sha256"],
                "section_path": current_path,
                "text": text,
            }
        )

    for element in article.find_all(CONTENT_TAGS):
        if not isinstance(element, Tag) or _is_nested_content(element):
            continue

        if element.name in HEADING_LEVELS:
            save_current_section()
            current_parts = []
            level = HEADING_LEVELS[element.name]
            anchor_link = element.select_one("a.headerlink")
            if anchor_link is not None:
                anchor_link.decompose()
            heading = element.get_text(" ", strip=True)
            heading_path[level] = heading
            for deeper_level in range(level + 1, 5):
                heading_path.pop(deeper_level, None)
            current_path = [heading_path[key] for key in sorted(heading_path)]
            continue

        text = (
            element.get_text().strip()
            if element.name == "pre"
            else element.get_text(" ", strip=True)
        )
        if not text:
            continue
        if element.name == "pre":
            current_parts.append(f"```\n{text}\n```")
        elif element.name == "li":
            current_parts.append(f"- {text}")
        else:
            current_parts.append(text)

    save_current_section()
    return sections


def _is_nested_content(element: Tag) -> bool:
    """Avoid duplicate text from paragraphs nested inside list items."""
    parent = element.parent
    while isinstance(parent, Tag):
        if parent.name in CONTENT_TAGS:
            return True
        parent = parent.parent
    return False
