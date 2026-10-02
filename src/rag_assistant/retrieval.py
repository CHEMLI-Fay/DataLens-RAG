"""FAISS index creation and source-attributed vector retrieval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rag_assistant.config import EmbeddingSettings, RetrievalSettings
from rag_assistant.embeddings import embed_query


@dataclass
class RetrievalData:
    """Index and source metadata retained for the lifetime of a session."""

    index: Any
    chunk_ids: np.ndarray
    chunks_by_id: dict[str, dict[str, Any]]


def load_retrieval_data(
    chunks_file: Path, embedding_file: Path, index_file: Path
) -> RetrievalData:
    """Load retrieval artifacts once without decompressing unused vectors."""
    with np.load(embedding_file, allow_pickle=False) as artifact:
        chunk_ids = artifact["chunk_ids"].copy()
    chunks_by_id = _load_chunks_by_id(chunks_file)
    index = _load_faiss().read_index(str(index_file))
    if chunk_ids.ndim != 1 or index.ntotal != len(chunk_ids) or not index.ntotal:
        raise ValueError("The FAISS index and chunk identifiers must have matching non-zero counts.")
    if any(str(identifier) not in chunks_by_id for identifier in chunk_ids):
        raise ValueError("Some indexed chunks are missing from the source metadata.")
    return RetrievalData(index, chunk_ids, chunks_by_id)


def build_faiss_index(embedding_file: Path, output_directory: Path) -> dict[str, Any]:
    """Build an exact inner-product index from normalized chunk embeddings."""
    faiss = _load_faiss()
    vectors, chunk_ids = _load_embedding_artifact(embedding_file)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    output_directory.mkdir(parents=True, exist_ok=True)
    index_path = output_directory / "chunks.index"
    faiss.write_index(index, str(index_path))
    manifest = {
        "index_type": "IndexFlatIP",
        "embedding_dimension": int(vectors.shape[1]),
        "vector_count": int(vectors.shape[0]),
        "embedding_artifact": str(embedding_file),
        "embedding_artifact_sha256": hashlib.sha256(embedding_file.read_bytes()).hexdigest(),
        "index_file": index_path.name,
        "chunk_ids": chunk_ids.tolist(),
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def search_chunks(
    query: str,
    chunks_file: Path,
    embedding_file: Path,
    index_file: Path,
    embedding_settings: EmbeddingSettings,
    retrieval_settings: RetrievalSettings,
    embedding_model: Any | None = None,
    retrieval_data: RetrievalData | None = None,
) -> list[dict[str, Any]]:
    """Return the most similar source chunks for one user question."""
    data = retrieval_data if retrieval_data is not None else load_retrieval_data(
        chunks_file, embedding_file, index_file
    )
    index, chunk_ids, chunks_by_id = data.index, data.chunk_ids, data.chunks_by_id

    query_vector = embed_query(
        query, embedding_settings, model=embedding_model
    ).astype(np.float32)
    if query_vector.shape != (1, index.d):
        raise ValueError("The query model dimension does not match the FAISS index.")
    result_count = min(retrieval_settings.top_k, index.ntotal)
    scores, positions = index.search(query_vector, result_count)

    results: list[dict[str, Any]] = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue
        chunk_id = str(chunk_ids[position])
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            raise ValueError(f"Chunk {chunk_id} is missing from {chunks_file}.")
        results.append({"score": float(score), **chunk})
    return results


def _load_embedding_artifact(embedding_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load vectors and IDs, checking FAISS-compatible shape and data type."""
    with np.load(embedding_file, allow_pickle=False) as artifact:
        vectors = artifact["vectors"].astype(np.float32)
        chunk_ids = artifact["chunk_ids"]
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Embedding vectors must be a non-empty two-dimensional array.")
    if len(chunk_ids) != vectors.shape[0]:
        raise ValueError("Every embedding vector requires a chunk identifier.")
    return vectors, chunk_ids


def _load_chunks_by_id(chunks_file: Path) -> dict[str, dict[str, Any]]:
    """Load chunk metadata used to make search results understandable."""
    chunks = [
        json.loads(line)
        for line in chunks_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mapping = {chunk["chunk_id"]: chunk for chunk in chunks}
    if not mapping or len(mapping) != len(chunks):
        raise ValueError("Chunk IDs must be present and unique.")
    return mapping


def _load_faiss() -> Any:
    """Import the optional FAISS dependency only when vector search is requested."""
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError(
            "Install the retrieval dependencies with: python -m pip install -e '.[retrieval]'"
        ) from error
    return faiss
