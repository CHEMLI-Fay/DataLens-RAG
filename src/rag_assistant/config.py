"""Loading and validation for project configuration files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from urllib.parse import urlparse


@dataclass(frozen=True)
class Source:
    """A documentation page that is permitted to enter the corpus."""

    identifier: str
    library: str
    title: str
    url: str


@dataclass(frozen=True)
class ChunkingSettings:
    """Validated settings for the fixed-window chunking baseline."""

    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class EmbeddingSettings:
    """Validated settings for the embedding model."""

    model_name: str
    normalize_embeddings: bool
    document_prefix: str
    query_prefix: str


@dataclass(frozen=True)
class RetrievalSettings:
    """Validated settings for vector retrieval."""

    top_k: int


@dataclass(frozen=True)
class GenerationSettings:
    """Validated settings for local answer generation."""

    provider: str
    model_name: str
    temperature: float
    max_context_characters: int
    max_output_tokens: int = 1200
    keep_alive_seconds: int = 1800
    timeout_seconds: float = 120.0


def load_sources(config_path: Path) -> list[Source]:
    """Load a non-empty, unique and HTTP(S)-only source allowlist."""
    with config_path.open("rb") as config_file:
        configuration = tomllib.load(config_file)

    source_entries = configuration.get("sources")
    if not isinstance(source_entries, list) or not source_entries:
        raise ValueError("The source configuration must define at least one [[sources]] entry.")

    sources: list[Source] = []
    identifiers: set[str] = set()
    for entry in source_entries:
        if not isinstance(entry, dict):
            raise ValueError("Each source entry must be a TOML table.")

        values = {field: entry.get(field) for field in ("id", "library", "title", "url")}
        if not all(isinstance(value, str) and value.strip() for value in values.values()):
            raise ValueError("Every source requires non-empty id, library, title and url fields.")

        identifier = values["id"].strip()
        if identifier in identifiers:
            raise ValueError(f"Duplicate source id: {identifier}")

        parsed_url = urlparse(values["url"])
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError(f"Source URL must be an HTTPS URL: {values['url']}")

        identifiers.add(identifier)
        sources.append(
            Source(
                identifier=identifier,
                library=values["library"].strip(),
                title=values["title"].strip(),
                url=values["url"].strip(),
            )
        )
    return sources


def load_chunking_settings(config_path: Path) -> ChunkingSettings:
    """Load fixed-window chunking settings from TOML configuration."""
    with config_path.open("rb") as config_file:
        configuration = tomllib.load(config_file)

    chunking = configuration.get("chunking")
    if not isinstance(chunking, dict):
        raise ValueError("The settings file must define a [chunking] table.")
    if chunking.get("strategy") != "fixed_tokens":
        raise ValueError("Only the fixed_tokens chunking baseline is implemented.")

    chunk_size = chunking.get("chunk_size")
    chunk_overlap = chunking.get("chunk_overlap")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if not isinstance(chunk_overlap, int) or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be at least zero and smaller than chunk_size.")
    return ChunkingSettings(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def load_embedding_settings(config_path: Path) -> EmbeddingSettings:
    """Load the embedding model settings from TOML configuration."""
    with config_path.open("rb") as config_file:
        configuration = tomllib.load(config_file)

    embedding = configuration.get("embedding")
    if not isinstance(embedding, dict):
        raise ValueError("The settings file must define an [embedding] table.")

    model_name = embedding.get("model_name")
    normalize_embeddings = embedding.get("normalize_embeddings")
    document_prefix = embedding.get("document_prefix")
    query_prefix = embedding.get("query_prefix")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("embedding.model_name must be a non-empty string.")
    if not isinstance(normalize_embeddings, bool):
        raise ValueError("embedding.normalize_embeddings must be a boolean.")
    if not isinstance(document_prefix, str) or not isinstance(query_prefix, str):
        raise ValueError("Embedding prefixes must be strings.")
    return EmbeddingSettings(
        model_name=model_name,
        normalize_embeddings=normalize_embeddings,
        document_prefix=document_prefix,
        query_prefix=query_prefix,
    )


def load_retrieval_settings(config_path: Path) -> RetrievalSettings:
    """Load the number of chunks returned by vector search."""
    with config_path.open("rb") as config_file:
        configuration = tomllib.load(config_file)

    retrieval = configuration.get("retrieval")
    if not isinstance(retrieval, dict):
        raise ValueError("The settings file must define a [retrieval] table.")
    top_k = retrieval.get("top_k")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("retrieval.top_k must be a positive integer.")
    return RetrievalSettings(top_k=top_k)


def load_generation_settings(config_path: Path) -> GenerationSettings:
    """Load local generation settings from TOML configuration."""
    with config_path.open("rb") as config_file:
        configuration = tomllib.load(config_file)

    generation = configuration.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("The settings file must define a [generation] table.")
    provider = generation.get("provider")
    model_name = generation.get("model_name")
    temperature = generation.get("temperature")
    max_context_characters = generation.get("max_context_characters")
    if provider != "ollama":
        raise ValueError("Only the ollama generation provider is implemented.")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("generation.model_name must be a non-empty string.")
    if not isinstance(temperature, (float, int)) or not 0 <= temperature <= 1:
        raise ValueError("generation.temperature must be a number between zero and one.")
    if not isinstance(max_context_characters, int) or max_context_characters <= 0:
        raise ValueError("generation.max_context_characters must be a positive integer.")
    limits = {
        "max_output_tokens": generation.get("max_output_tokens", 1200),
        "keep_alive_seconds": generation.get("keep_alive_seconds", 1800),
    }
    for name, value in limits.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"generation.{name} must be a positive integer.")
    timeout = generation.get("timeout_seconds", 120.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
        raise ValueError("generation.timeout_seconds must be a finite positive number.")
    return GenerationSettings(
        provider=provider,
        model_name=model_name,
        temperature=float(temperature),
        max_context_characters=max_context_characters,
        **limits,
        timeout_seconds=float(timeout),
    )
