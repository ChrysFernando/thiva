"""
Knowledge Base / RAG module for the Winrich multilingual voice agent.

Uses ChromaDB as the vector store and sentence-transformers (all-MiniLM-L6-v2)
for local embeddings. Provides document ingestion, chunking, and semantic
retrieval so the voice agent can ground its answers in company-specific facts.
"""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Graceful imports – the module stays importable even when deps are missing.
# ---------------------------------------------------------------------------

_CHROMADB_AVAILABLE = False
_SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings

    _CHROMADB_AVAILABLE = True
except ImportError:
    print(
        "[knowledge_base] WARNING: chromadb is not installed. "
        "Install it with: pip install chromadb"
    )

try:
    from sentence_transformers import SentenceTransformer

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print(
        "[knowledge_base] WARNING: sentence-transformers is not installed. "
        "Install it with: pip install sentence-transformers"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "winrich_kb"
PERSIST_DIRECTORY = "./chroma_db"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
DEFAULT_DOCS_DIRECTORY = "knowledge_docs"

# ---------------------------------------------------------------------------
# Module-level globals
# ---------------------------------------------------------------------------

_chroma_client: Optional["chromadb.ClientAPI"] = None
_collection: Optional["chromadb.Collection"] = None
_embedding_model: Optional["SentenceTransformer"] = None


def _get_chroma_client() -> Optional["chromadb.ClientAPI"]:
    """Return (and lazily create) the persistent ChromaDB client."""
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client
    if not _CHROMADB_AVAILABLE:
        return None
    try:
        _chroma_client = chromadb.Client(
            Settings(
                persist_directory=PERSIST_DIRECTORY,
                anonymized_telemetry=False,
                is_persistent=True,
            )
        )
    except Exception as exc:
        print(f"[knowledge_base] Failed to create ChromaDB client: {exc}")
        _chroma_client = None
    return _chroma_client


def _get_embedding_model() -> Optional["SentenceTransformer"]:
    """Return (and lazily load) the sentence-transformer model."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    try:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as exc:
        print(f"[knowledge_base] Failed to load embedding model: {exc}")
        _embedding_model = None
    return _embedding_model


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Returns empty list on failure."""
    model = _get_embedding_model()
    if model is None:
        return []
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def _stable_id(text: str, index: int = 0) -> str:
    """Generate a deterministic ID for a chunk so re-ingestion is idempotent."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"chunk_{digest}_{index}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split *text* into chunks of approximately *chunk_size* characters
    with *overlap* characters shared between consecutive chunks.

    Tries to break on paragraph/sentence boundaries when possible so that
    chunks remain coherent.
    """
    if not text or not text.strip():
        return []

    # Normalise whitespace for cleaner chunks.
    text = text.strip()

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        if end < text_len:
            # Try to break at a paragraph boundary first, then sentence, then
            # word boundary to avoid cutting mid-word.
            breakpoint = text.rfind("\n\n", start, end)
            if breakpoint == -1 or breakpoint <= start:
                breakpoint = text.rfind(". ", start, end)
                if breakpoint != -1:
                    breakpoint += 1  # include the period
            if breakpoint == -1 or breakpoint <= start:
                breakpoint = text.rfind(" ", start, end)
            if breakpoint != -1 and breakpoint > start:
                end = breakpoint

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward, accounting for overlap.
        start = max(start + 1, end - overlap)

    return chunks


def _read_file(path: Path) -> str:
    """Read the text content of a supported file."""
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        try:
            import PyPDF2  # type: ignore

            text_parts: list[str] = []
            with open(path, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n".join(text_parts)
        except ImportError:
            print(
                f"[knowledge_base] Skipping PDF {path.name} – "
                "PyPDF2 is not installed."
            )
            return ""
        except Exception as exc:
            print(f"[knowledge_base] Error reading PDF {path.name}: {exc}")
            return ""

    return ""


def initialize_kb(
    docs_directory: str = DEFAULT_DOCS_DIRECTORY,
) -> Optional["chromadb.Collection"]:
    """Scan *docs_directory* for supported files, chunk them, embed, and
    store in ChromaDB.  The operation is **idempotent** – if the collection
    already contains documents it is returned as-is.

    Returns the ChromaDB collection, or ``None`` on failure.
    """
    global _collection

    client = _get_chroma_client()
    if client is None:
        print("[knowledge_base] ChromaDB client unavailable; skipping init.")
        return None

    # Get or create the collection.
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Idempotency: skip ingestion if collection already has documents.
    if _collection.count() > 0:
        print(
            f"[knowledge_base] Collection '{COLLECTION_NAME}' already has "
            f"{_collection.count()} document(s). Skipping ingestion."
        )
        return _collection

    # Discover files.
    docs_path = Path(docs_directory)
    if not docs_path.is_dir():
        print(
            f"[knowledge_base] Directory '{docs_directory}' not found. "
            "No documents ingested."
        )
        return _collection

    all_chunks: list[str] = []
    all_metadatas: list[dict] = []
    all_ids: list[str] = []

    for file_path in sorted(docs_path.iterdir()):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = _read_file(file_path)
        if not text.strip():
            continue

        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            chunk_id = _stable_id(chunk, idx)
            all_chunks.append(chunk)
            all_metadatas.append(
                {
                    "source": file_path.name,
                    "chunk_index": idx,
                }
            )
            all_ids.append(chunk_id)

    if not all_chunks:
        print("[knowledge_base] No document content found to ingest.")
        return _collection

    # Embed all chunks at once for efficiency.
    embeddings = _embed_texts(all_chunks)
    if not embeddings:
        print(
            "[knowledge_base] Embedding failed. "
            "Storing documents without custom embeddings."
        )
        _collection.add(
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids,
        )
    else:
        _collection.add(
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadatas,
            ids=all_ids,
        )

    print(
        f"[knowledge_base] Ingested {len(all_chunks)} chunk(s) from "
        f"'{docs_directory}' into collection '{COLLECTION_NAME}'."
    )
    return _collection


def retrieve_context(query: str, n_results: int = 3) -> str:
    """Search the knowledge base for chunks relevant to *query*.

    Returns a formatted string of the top-*n_results* chunks, or a fallback
    message when the KB is unavailable or empty.
    """
    global _collection

    fallback = "No knowledge base loaded. Answering from general knowledge."

    if _collection is None:
        # Try to reconnect in case initialize_kb was called after import.
        client = _get_chroma_client()
        if client is not None:
            try:
                _collection = client.get_collection(name=COLLECTION_NAME)
            except Exception:
                return fallback
        else:
            return fallback

    if _collection.count() == 0:
        return fallback

    # Embed the query.
    query_embedding = _embed_texts([query])

    try:
        if query_embedding:
            results = _collection.query(
                query_embeddings=query_embedding,
                n_results=min(n_results, _collection.count()),
            )
        else:
            # Fallback: let ChromaDB handle embedding (uses its default).
            results = _collection.query(
                query_texts=[query],
                n_results=min(n_results, _collection.count()),
            )
    except Exception as exc:
        print(f"[knowledge_base] Query failed: {exc}")
        return fallback

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return fallback

    parts: list[str] = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        source = meta.get("source", "unknown") if meta else "unknown"
        parts.append(f"[{i}] (source: {source})\n{doc}")

    return "\n\n---\n\n".join(parts)


def add_document(text: str, metadata: Optional[dict] = None) -> None:
    """Add a single document (or chunk) to the knowledge base.

    If the text is longer than the default chunk size it will be split
    automatically.
    """
    global _collection

    client = _get_chroma_client()
    if client is None:
        print("[knowledge_base] ChromaDB unavailable; cannot add document.")
        return

    if _collection is None:
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    chunks = chunk_text(text)
    if not chunks:
        return

    ids: list[str] = []
    metadatas: list[dict] = []
    for idx, chunk in enumerate(chunks):
        ids.append(_stable_id(chunk, idx))
        meta = dict(metadata) if metadata else {}
        meta["chunk_index"] = idx
        metadatas.append(meta)

    embeddings = _embed_texts(chunks)

    if embeddings:
        _collection.upsert(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
    else:
        _collection.upsert(
            documents=chunks,
            metadatas=metadatas,
            ids=ids,
        )

    print(f"[knowledge_base] Added {len(chunks)} chunk(s) to the KB.")


# ---------------------------------------------------------------------------
# Module-level initialisation
# ---------------------------------------------------------------------------

# Eagerly create the client so it is ready when the first query arrives.
_get_chroma_client()


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Knowledge Base self-test ===\n")

    # 1. Initialise from the sample docs directory.
    collection = initialize_kb()
    if collection is not None:
        print(f"Collection count: {collection.count()}\n")

    # 2. Run a few sample queries.
    sample_queries = [
        "What is the warranty policy?",
        "Where is the Kandy service center?",
        "How much does the rice cooker cost?",
        "How do I return a product?",
    ]

    for q in sample_queries:
        print(f"Q: {q}")
        print(f"A context:\n{retrieve_context(q)}\n")
        print("-" * 60)
