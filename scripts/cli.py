"""Prax RAG command-line interface.

Interact with the RAG pipeline directly from the terminal (no HTTP server needed).

Examples:
    PYTHONPATH=. .venv/bin/python scripts/cli.py ingest-file docs/notes.txt --title Notes
    PYTHONPATH=. .venv/bin/python scripts/cli.py ingest-text "Prax uses pgvector." --source note
    PYTHONPATH=. .venv/bin/python scripts/cli.py query "What vector db does Prax use?"
    PYTHONPATH=. .venv/bin/python scripts/cli.py query "..." --top-k 3 --json
    PYTHONPATH=. .venv/bin/python scripts/cli.py stats
"""

import argparse
import json
import sys
from pathlib import Path

from app.db.session import SessionLocal, init_db
from app.repositories.vector_store import VectorStore
from app.services.embeddings import EmbeddingService
from app.services.ingestion import IngestionService
from app.services.llm import LLMService
from app.services.rag import RAGService
from prax_kb import UnsupportedFileError


def _session():
    init_db()
    return SessionLocal()


def cmd_ingest_text(args: argparse.Namespace) -> int:
    session = _session()
    try:
        service = IngestionService(session, EmbeddingService())
        result = service.ingest_text(
            text=args.text, source=args.source, title=args.title
        )
    finally:
        session.close()
    print(f"✅ Ingested {result['chunks_ingested']} chunk(s) "
          f"(document_id={result['document_id']}, source={result['source']})")
    return 0


def cmd_ingest_file(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.is_file():
        print(f"❌ File not found: {path}", file=sys.stderr)
        return 1

    session = _session()
    try:
        service = IngestionService(session, EmbeddingService())
        result = service.ingest_file(
            path.read_bytes(), filename=path.name, title=args.title or path.name
        )
    except UnsupportedFileError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    finally:
        session.close()
    print(f"✅ Ingested {result['chunks_ingested']} chunk(s) from {path} "
          f"(document_id={result['document_id']})")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    session = _session()
    try:
        service = RAGService(session, EmbeddingService(), LLMService())
        result = service.answer(question=args.question, top_k=args.top_k)
    finally:
        session.close()

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result["answer"])
    t = result.get("timings")
    if t:
        print(
            f"\n⏱️  retrieval {t['retrieval_ms']:.0f} ms · "
            f"generation {t['generation_ms']:.0f} ms · total {t['total_ms']:.0f} ms"
        )
    if result["sources"]:
        print("\n" + "-" * 70)
        print("SOURCES")
        print("-" * 70)
        for s in result["sources"]:
            title = s["title"] or s["source"]
            print(f"[{s['rank']}] {title}  (score={s['score']}, source={s['source']})")
            print(f"    {s['preview']}")
    print()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    session = _session()
    try:
        store = VectorStore(session)
        docs, chunks = store.count_documents(), store.count_chunks()
    finally:
        session.close()
    print(f"📊 documents={docs}  chunks={chunks}")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    if not args.yes:
        confirm = input("⚠️  Delete ALL documents and chunks? [y/N] ").strip().lower()
        if confirm not in {"y", "yes"}:
            print("Aborted.")
            return 1
    session = _session()
    try:
        removed = VectorStore(session).clear()
    finally:
        session.close()
    print(f"🧹 Removed {removed} document(s). Knowledge base is empty.")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    session = _session()
    try:
        removed = VectorStore(session).delete_by_source(args.source)
    finally:
        session.close()
    print(f"🗑️  Removed {removed} document(s) with source={args.source}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prax-rag", description="Prax RAG terminal interface"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_text = sub.add_parser("ingest-text", help="Ingest raw text")
    p_text.add_argument("text")
    p_text.add_argument("--source", required=True)
    p_text.add_argument("--title", default=None)
    p_text.set_defaults(func=cmd_ingest_text)

    p_file = sub.add_parser("ingest-file", help="Ingest a PDF, DOCX, or text file")
    p_file.add_argument("path")
    p_file.add_argument("--title", default=None)
    p_file.set_defaults(func=cmd_ingest_file)

    p_query = sub.add_parser("query", help="Ask a grounded question")
    p_query.add_argument("question")
    p_query.add_argument("--top-k", type=int, default=None, dest="top_k")
    p_query.add_argument("--json", action="store_true", help="Print raw JSON")
    p_query.set_defaults(func=cmd_query)

    p_stats = sub.add_parser("stats", help="Show document/chunk counts")
    p_stats.set_defaults(func=cmd_stats)

    p_reset = sub.add_parser("reset", help="Delete ALL documents and chunks")
    p_reset.add_argument("--yes", action="store_true", help="Skip confirmation")
    p_reset.set_defaults(func=cmd_reset)

    p_delete = sub.add_parser("delete", help="Delete documents by source")
    p_delete.add_argument("source")
    p_delete.set_defaults(func=cmd_delete)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
