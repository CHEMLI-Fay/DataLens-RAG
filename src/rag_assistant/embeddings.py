"""Local embedding generation for retrieved RAG chunks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rag_assistant.config import EmbeddingSettings


def create_embeddings(
    input_file: Path,
    output_directory: Path,
    settings: EmbeddingSettings,
    batch_size: int,
    show_progress_bar: bool,
) -> dict[str, Any]:
    """Embed chunks locally and save vectors with their stable chunk identifiers."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    chunks = _load_chunks(input_file)
    model = _load_model(settings.model_name)
    texts = [f"{settings.document_prefix}{chunk['text']}" for chunk in chunks]
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=settings.normalize_embeddings,
        convert_to_numpy=True,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / "chunk_embeddings.npz",
        vectors=vectors,
        chunk_ids=np.array([chunk["chunk_id"] for chunk in chunks]),
    )
    manifest = {
        "model_name": settings.model_name,
        "normalize_embeddings": settings.normalize_embeddings,
        "document_prefix": settings.document_prefix,
        "query_prefix": settings.query_prefix,
        "input_file": str(input_file),
        "input_sha256": hashlib.sha256(input_file.read_bytes()).hexdigest(),
        "chunk_count": len(chunks),
        "embedding_dimension": int(vectors.shape[1]),
        "artifact_file": "chunk_embeddings.npz",
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_embedding_model(settings: EmbeddingSettings) -> Any:
    """Load the configured embedding model for one or more operations."""
    return _load_model(settings.model_name, local_files_only=True)


def embed_query(
    query: str, settings: EmbeddingSettings, model: Any | None = None
) -> np.ndarray:
    """Embed one user question with the model's query prefix."""
    if not query.strip():
        raise ValueError("The query must not be empty.")
    embedding_model = model if model is not None else load_embedding_model(settings)
    return embedding_model.encode(
        [f"{settings.query_prefix}{query}"],
        normalize_embeddings=settings.normalize_embeddings,
        convert_to_numpy=True,
    )


def _load_chunks(input_file: Path) -> list[dict[str, Any]]:
    """Read chunks and ensure every vector will have a stable identifier."""
    chunks = [
        json.loads(line)
        for line in input_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not chunks:
        raise ValueError("The input chunks file is empty.")
    if any(not isinstance(chunk.get("chunk_id"), str) or not chunk.get("text") for chunk in chunks):
        raise ValueError("Every chunk requires a non-empty chunk_id and text field.")
    return chunks


def _load_model(model_name: str, local_files_only: bool = False) -> Any:
    """Import the optional retrieval dependency only when embeddings are requested."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Install the retrieval dependencies with: python -m pip install -e '.[retrieval]'"
        ) from error
    if local_files_only:
        from transformers.utils.logging import disable_progress_bar

        disable_progress_bar()
    try:
        return SentenceTransformer(model_name, local_files_only=local_files_only)
    except OSError as error:
        if local_files_only:
            raise RuntimeError(
                f"Embedding model {model_name} is missing from the local cache. "
                "Run rag-assistant embed once to download it."
            ) from error
        raise
