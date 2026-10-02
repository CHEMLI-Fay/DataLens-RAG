"""Controlled download of allowlisted documentation pages."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from rag_assistant.config import Source, load_sources


USER_AGENT = "rag-data-science-docs/0.1 (+local educational project)"


def download_sources(
    config_path: Path,
    output_directory: Path,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    """Download configured pages and save a provenance manifest beside them."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero.")

    sources = load_sources(config_path)
    downloads: list[tuple[dict[str, Any], bytes]] = []
    records: list[dict[str, Any]] = []

    with httpx.Client(
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout_seconds,
    ) as client:
        for source in sources:
            record, content = _download_source(client, source)
            records.append(record)
            downloads.append((record, content))

    output_directory.mkdir(parents=True, exist_ok=True)
    for record, content in downloads:
        (output_directory / record["file_name"]).write_bytes(content)
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps({"sources": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return records


def _download_source(
    client: httpx.Client,
    source: Source,
) -> tuple[dict[str, Any], bytes]:
    """Download one source and reject a redirect outside its configured host."""
    response = client.get(source.url)
    response.raise_for_status()

    configured_host = urlparse(source.url).hostname
    final_host = response.url.host
    if configured_host != final_host:
        raise ValueError(
            f"Source {source.identifier} redirected from {configured_host} to {final_host}."
        )

    content = response.content
    file_name = f"{source.identifier}.html"
    record = {
        "id": source.identifier,
        "library": source.library,
        "title": source.title,
        "configured_url": source.url,
        "retrieved_url": str(response.url),
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "content_type": response.headers.get("content-type", ""),
        "file_name": file_name,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    return record, content
