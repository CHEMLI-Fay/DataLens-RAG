"""Command-line entry point for the RAG assistant."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_assistant.chunking import chunk_sections
from rag_assistant.config import (
    load_chunking_settings,
    load_embedding_settings,
    load_generation_settings,
    load_retrieval_settings,
)
from rag_assistant.embeddings import create_embeddings, load_embedding_model
from rag_assistant.evaluation import evaluate_questions
from rag_assistant.extract import extract_sources
from rag_assistant.generation import generate_answer, translate_retrieval_query
from rag_assistant.ingest import download_sources
from rag_assistant.retrieval import (
    RetrievalData, build_faiss_index, load_retrieval_data, search_chunks,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Build a local corpus from an allowlisted set of sources."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Download the configured documentation sources."
    )
    ingest_parser.add_argument(
        "--sources-config",
        type=Path,
        default=Path("config/sources.toml"),
        help="Path to the source allowlist in TOML format.",
    )
    ingest_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for downloaded HTML and its manifest.",
    )
    ingest_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Network timeout for each source request (default: 120 seconds).",
    )

    extract_parser = subparsers.add_parser(
        "extract", help="Extract structured sections from downloaded HTML."
    )
    extract_parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing downloaded HTML and manifest.json.",
    )
    extract_parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/processed/sections.jsonl"),
        help="JSONL destination for extracted sections.",
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Split extracted sections into retrieval chunks."
    )
    chunk_parser.add_argument(
        "--settings-config",
        type=Path,
        default=Path("config/settings.toml"),
        help="Path to the project settings in TOML format.",
    )
    chunk_parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/processed/sections.jsonl"),
        help="JSONL file containing extracted source sections.",
    )
    chunk_parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="JSONL destination for retrieval chunks.",
    )
    chunk_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show progress while processing source sections.",
    )

    embedding_parser = subparsers.add_parser(
        "embed", help="Generate local vectors for retrieval chunks."
    )
    embedding_parser.add_argument(
        "--settings-config",
        type=Path,
        default=Path("config/settings.toml"),
        help="Path to the project settings in TOML format.",
    )
    embedding_parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="JSONL file containing chunks to embed.",
    )
    embedding_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/embeddings"),
        help="Directory for local embedding artifacts.",
    )
    embedding_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of chunks processed by the model at once.",
    )
    embedding_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show the embedding model progress bar.",
    )

    index_parser = subparsers.add_parser(
        "index", help="Build the local FAISS index from chunk embeddings."
    )
    index_parser.add_argument(
        "--embedding-file",
        type=Path,
        default=Path("artifacts/embeddings/chunk_embeddings.npz"),
        help="Local chunk embedding artifact.",
    )
    index_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/faiss"),
        help="Directory for the local FAISS index.",
    )

    search_parser = subparsers.add_parser(
        "search", help="Search the local FAISS index for supporting chunks."
    )
    search_parser.add_argument("query", help="Question to retrieve evidence for.")
    search_parser.add_argument(
        "--settings-config",
        type=Path,
        default=Path("config/settings.toml"),
        help="Path to the project settings in TOML format.",
    )
    search_parser.add_argument(
        "--chunks-file",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="JSONL file containing source chunks.",
    )
    search_parser.add_argument(
        "--embedding-file",
        type=Path,
        default=Path("artifacts/embeddings/chunk_embeddings.npz"),
        help="Local chunk embedding artifact.",
    )
    search_parser.add_argument(
        "--index-file",
        type=Path,
        default=Path("artifacts/faiss/chunks.index"),
        help="Local FAISS index file.",
    )

    ask_parser = subparsers.add_parser(
        "ask", help="Answer a question using retrieved documentation evidence."
    )
    ask_parser.add_argument("question", help="Question to answer.")
    ask_parser.add_argument(
        "--settings-config",
        type=Path,
        default=Path("config/settings.toml"),
        help="Path to the project settings in TOML format.",
    )
    ask_parser.add_argument(
        "--chunks-file",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="JSONL file containing source chunks.",
    )
    ask_parser.add_argument(
        "--embedding-file",
        type=Path,
        default=Path("artifacts/embeddings/chunk_embeddings.npz"),
        help="Local chunk embedding artifact.",
    )
    ask_parser.add_argument(
        "--index-file",
        type=Path,
        default=Path("artifacts/faiss/chunks.index"),
        help="Local FAISS index file.",
    )
    ask_parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path("prompts/answer.txt"),
        help="Prompt template used for grounded answer generation.",
    )

    session_parser = subparsers.add_parser(
        "ask-session",
        help="Answer several questions while keeping the embedding model in memory.",
    )
    session_parser.add_argument(
        "--settings-config",
        type=Path,
        default=Path("config/settings.toml"),
        help="Path to the project settings in TOML format.",
    )
    session_parser.add_argument(
        "--chunks-file",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="JSONL file containing source chunks.",
    )
    session_parser.add_argument(
        "--embedding-file",
        type=Path,
        default=Path("artifacts/embeddings/chunk_embeddings.npz"),
        help="Local chunk embedding artifact.",
    )
    session_parser.add_argument(
        "--index-file",
        type=Path,
        default=Path("artifacts/faiss/chunks.index"),
        help="Local FAISS index file.",
    )
    session_parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path("prompts/answer.txt"),
        help="Prompt template used for grounded answer generation.",
    )

    evaluation_parser = subparsers.add_parser(
        "evaluate", help="Run the 15-question local RAG evaluation."
    )
    evaluation_parser.add_argument(
        "--settings-config", type=Path, default=Path("config/settings.toml")
    )
    evaluation_parser.add_argument(
        "--questions-file", type=Path, default=Path("data/evaluation/questions.jsonl")
    )
    evaluation_parser.add_argument(
        "--chunks-file", type=Path, default=Path("data/processed/chunks.jsonl")
    )
    evaluation_parser.add_argument(
        "--embedding-file", type=Path,
        default=Path("artifacts/embeddings/chunk_embeddings.npz")
    )
    evaluation_parser.add_argument(
        "--index-file", type=Path, default=Path("artifacts/faiss/chunks.index")
    )
    evaluation_parser.add_argument(
        "--prompt-file", type=Path, default=Path("prompts/answer.txt")
    )
    evaluation_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/evaluation")
    )
    return parser


def main() -> None:
    """Run the selected command."""
    arguments = build_parser().parse_args()
    if arguments.command == "ingest":
        records = download_sources(
            config_path=arguments.sources_config,
            output_directory=arguments.output_dir,
            timeout_seconds=arguments.timeout_seconds,
        )
        print(f"Downloaded {len(records)} source document(s) to {arguments.output_dir}.")
    elif arguments.command == "extract":
        sections = extract_sources(
            input_directory=arguments.input_dir,
            output_file=arguments.output_file,
        )
        print(f"Extracted {len(sections)} section(s) to {arguments.output_file}.")
    elif arguments.command == "chunk":
        settings = load_chunking_settings(arguments.settings_config)

        def report_progress(completed: int, total: int, section_id: str) -> None:
            if completed == 1 or completed == total or completed % 25 == 0:
                print(f"Processed {completed}/{total} sections; latest: {section_id}")

        chunks = chunk_sections(
            input_file=arguments.input_file,
            output_file=arguments.output_file,
            settings=settings,
            progress_callback=report_progress if arguments.verbose else None,
        )
        print(f"Created {len(chunks)} chunk(s) to {arguments.output_file}.")
    elif arguments.command == "embed":
        settings = load_embedding_settings(arguments.settings_config)
        manifest = create_embeddings(
            input_file=arguments.input_file,
            output_directory=arguments.output_dir,
            settings=settings,
            batch_size=arguments.batch_size,
            show_progress_bar=arguments.verbose,
        )
        print(
            f"Created {manifest['chunk_count']} embeddings with "
            f"{manifest['embedding_dimension']} dimensions in {arguments.output_dir}."
        )
    elif arguments.command == "index":
        manifest = build_faiss_index(
            embedding_file=arguments.embedding_file,
            output_directory=arguments.output_dir,
        )
        print(
            f"Created {manifest['index_type']} with {manifest['vector_count']} vectors "
            f"in {arguments.output_dir}."
        )
    elif arguments.command == "search":
        results = search_chunks(
            query=arguments.query,
            chunks_file=arguments.chunks_file,
            embedding_file=arguments.embedding_file,
            index_file=arguments.index_file,
            embedding_settings=load_embedding_settings(arguments.settings_config),
            retrieval_settings=load_retrieval_settings(arguments.settings_config),
        )
        for rank, result in enumerate(results, start=1):
            heading = " > ".join(result["section_path"])
            preview = " ".join(result["text"].split())[:400]
            print(f"{rank}. score={result['score']:.4f} | {heading}")
            print(f"   source={result['source_url']}")
            print(f"   {preview}\n")
    elif arguments.command == "ask":
        try:
            print("[1/3] Loading local models and index...", flush=True)
            embedding_model = load_embedding_model(
                load_embedding_settings(arguments.settings_config)
            )
            retrieval_data = load_retrieval_data(
                arguments.chunks_file, arguments.embedding_file, arguments.index_file
            )
            _answer_question(
                arguments, arguments.question,
                embedding_model=embedding_model, retrieval_data=retrieval_data,
            )
        except KeyboardInterrupt:
            print("\nRequest cancelled.")
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Error: {error}", flush=True)
    elif arguments.command == "ask-session":
        try:
            _run_answer_session(arguments)
        except KeyboardInterrupt:
            print("\nSession closed.")
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Error: {error}", flush=True)
    elif arguments.command == "evaluate":
        try:
            print("[1/3] Loading local models and index...", flush=True)
            embedding_settings = load_embedding_settings(arguments.settings_config)
            embedding_model = load_embedding_model(embedding_settings)
            retrieval_data = load_retrieval_data(
                arguments.chunks_file, arguments.embedding_file, arguments.index_file
            )
            print("[2/3] Running 15 evaluation questions...", flush=True)

            def report(position: int, total: int, identifier: str) -> None:
                print(f"  Question {position}/{total}: {identifier}", flush=True)

            results = evaluate_questions(
                questions_file=arguments.questions_file,
                output_directory=arguments.output_dir,
                chunks_file=arguments.chunks_file,
                embedding_file=arguments.embedding_file,
                index_file=arguments.index_file,
                prompt_file=arguments.prompt_file,
                embedding_settings=embedding_settings,
                retrieval_settings=load_retrieval_settings(arguments.settings_config),
                generation_settings=load_generation_settings(arguments.settings_config),
                embedding_model=embedding_model,
                retrieval_data=retrieval_data,
                progress_callback=report,
            )
            passed = sum(result["automatic_status"] == "PASS" for result in results)
            print("[3/3] Evaluation files saved.", flush=True)
            print(f"Automatic checks: {passed}/{len(results)} PASS; review the full report.")
            print(f"CSV: {arguments.output_dir / 'evaluation_results.csv'}")
            print(f"Report: {arguments.output_dir / 'evaluation_report.md'}")
        except KeyboardInterrupt:
            print("\nEvaluation cancelled; incomplete results were not saved.")
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Error: {error}", flush=True)


def _run_answer_session(arguments: argparse.Namespace) -> None:
    """Keep the embedding model in memory while answering several questions."""
    settings_path = arguments.settings_config
    embedding_settings = load_embedding_settings(settings_path)
    print("[1/3] Loading local models and index...", flush=True)
    embedding_model = load_embedding_model(embedding_settings)
    retrieval_data = load_retrieval_data(
        arguments.chunks_file, arguments.embedding_file, arguments.index_file
    )
    load_generation_settings(settings_path)
    print("Ready. Type a question, or 'exit'.", flush=True)
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession closed.")
            return
        if question.lower() in {"exit", "quit"}:
            print("Session closed.")
            return
        if not question:
            print("Please enter a non-empty question.")
            continue
        try:
            _answer_question(
                arguments, question, embedding_model=embedding_model,
                retrieval_data=retrieval_data,
            )
        except KeyboardInterrupt:
            print("\nRequest cancelled. You can enter another question or 'exit'.", flush=True)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Error: {error}\nSession is still open; you can retry.", flush=True)


def _answer_question(
    arguments: argparse.Namespace, question: str, embedding_model: object | None = None,
    retrieval_data: RetrievalData | None = None,
) -> None:
    """Retrieve evidence, generate a guarded answer, and show its sources."""
    settings_path = arguments.settings_config
    print("[2/3] Searching documentation...", flush=True)
    generation_settings = load_generation_settings(settings_path)
    retrieval_query = translate_retrieval_query(question, generation_settings)
    retrieved_chunks = search_chunks(
        query=retrieval_query,
        chunks_file=arguments.chunks_file,
        embedding_file=arguments.embedding_file,
        index_file=arguments.index_file,
        embedding_settings=load_embedding_settings(settings_path),
        retrieval_settings=load_retrieval_settings(settings_path),
        embedding_model=embedding_model,
        retrieval_data=retrieval_data,
    )
    print("[3/3] Generating answer...", flush=True)
    answer = generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        settings=generation_settings,
        prompt_path=arguments.prompt_file,
        retrieval_query=retrieval_query,
    )
    print(answer, flush=True)
    print("\nRetrieved sources:")
    for label, chunk in enumerate(retrieved_chunks, start=1):
        print(f"[{label}] {' > '.join(chunk['section_path'])}")
        print(f"    {chunk['source_url']}")
