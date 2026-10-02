"""
rag.py — NEXUS Retrieval Augmented Generation (RAG)
Indexes user documents into a FAISS vector store using sentence-transformers
embeddings, then retrieves relevant chunks at query time so the LLM can
answer questions grounded in the user's actual files.

Phase 20 of the NEXUS implementation plan.

How it works:
  1. build_index(folder_path) reads all supported files, chunks them, embeds
     each chunk with a sentence-transformer model, and stores them in a FAISS
     index alongside metadata (filename, chunk number, raw text).
  2. search(query, top_k) embeds the query and finds the nearest chunks.
  3. ask(question) combines search results with the LLM to produce a
     document-grounded answer.
"""

import json
import os
import pickle

import numpy as np

from nexus import config

# ---------------------------------------------------------------------------
# Paths — resolved relative to the project root (one level up from src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(config.PROJECT_ROOT)
RAG_INDEX_DIR = str(config.RAG_INDEX_DIR)
INDEX_FILE = os.path.join(RAG_INDEX_DIR, "faiss.index")
METADATA_FILE = os.path.join(RAG_INDEX_DIR, "metadata.pkl")
CONFIG_FILE = os.path.join(RAG_INDEX_DIR, "index_config.json")

# Embedding model — small, fast, runs on CPU (~90 MB)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Module-level state (lazy-loaded)
# ---------------------------------------------------------------------------
_embedding_model = None
_faiss_index = None
_metadata = None  # list of dicts: {"filename", "chunk_num", "text"}


# ---------------------------------------------------------------------------
# Embedding model loading
# ---------------------------------------------------------------------------
def _load_embedding_model():
    """
    Load the sentence-transformer embedding model (lazy, one-time).
    Uses all-MiniLM-L6-v2 which is small (~90 MB) and fast on CPU.
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[ERROR] sentence-transformers is not installed.")
        print("  Run: pip install sentence-transformers")
        return None

    print(f"  Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    try:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("  Embedding model loaded.")
        return _embedding_model
    except Exception as e:
        print(f"[ERROR] Failed to load embedding model: {e}")
        return None


# ---------------------------------------------------------------------------
# Document indexing (Section 20.2)
# ---------------------------------------------------------------------------
def build_index(folder_path):
    """
    Read all supported files in a folder, chunk them, generate embeddings,
    and store everything in a FAISS index for fast retrieval.

    Args:
        folder_path: Absolute path to the folder to index.

    Returns:
        A status message string describing what was indexed.
    """
    try:
        import faiss
    except ImportError:
        return ("FAISS is not installed. Please run:\n"
                "  pip install faiss-cpu")

    from nexus.security.permissions import is_allowed
    from nexus.utils import chunk_text, clean_text

    # --- Validate folder ---
    folder_path = os.path.abspath(folder_path)

    if not is_allowed(folder_path):
        return "Permission denied for this folder."
    if not os.path.isdir(folder_path):
        return f"Folder not found: {folder_path}"

    # --- Load embedding model ---
    model = _load_embedding_model()
    if model is None:
        return "Failed to load embedding model. Cannot build index."

    # --- Find supported files ---
    supported_exts = (".txt", ".pdf", ".xlsx", ".xls")
    files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and f.lower().endswith(supported_exts)
        and f != "summary_output.txt"
    ]

    if not files:
        return "No supported files found in this folder. Supported: .txt, .pdf, .xlsx"

    # --- Read and chunk all files ---
    from nexus.tools import files

    all_chunks = []   # list of {"filename", "chunk_num", "text"}
    total_files = len(files)

    for i, filename in enumerate(files, 1):
        filepath = os.path.join(folder_path, filename)
        print(f"  Reading file {i}/{total_files}: {filename}...")

        raw_text = files.read_file(filepath)

        # Check if the reader returned an error
        if raw_text.startswith(files._ERROR_PREFIXES):
            print(f"  Skipping {filename} (read error)")
            continue

        cleaned = clean_text(raw_text)
        if not cleaned:
            print(f"  Skipping {filename} (empty after cleaning)")
            continue

        # Chunk the text (smaller chunks for RAG — 500 words with 50 overlap)
        chunks = chunk_text(cleaned, chunk_size=500, overlap=50)

        for chunk_num, chunk_text_str in enumerate(chunks, 1):
            all_chunks.append({
                "filename": filename,
                "chunk_num": chunk_num,
                "text": chunk_text_str,
            })

    if not all_chunks:
        return "No readable content found in any files."

    # --- Generate embeddings ---
    print(f"  Generating embeddings for {len(all_chunks)} chunks...")

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    # --- Build FAISS index ---
    print("  Building FAISS index...")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine sim with normalized vectors)
    index.add(embeddings)

    # --- Save to disk ---
    os.makedirs(RAG_INDEX_DIR, exist_ok=True)

    faiss.write_index(index, INDEX_FILE)

    with open(METADATA_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    # Save config for reference
    config = {
        "folder_path": folder_path,
        "total_files": total_files,
        "total_chunks": len(all_chunks),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "dimension": dimension,
    }
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # --- Update module state ---
    global _faiss_index, _metadata
    _faiss_index = index
    _metadata = all_chunks

    return (
        f"RAG index built successfully!\n"
        f"  Folder: {folder_path}\n"
        f"  Files indexed: {total_files}\n"
        f"  Total chunks: {len(all_chunks)}\n"
        f"  Index saved to: {RAG_INDEX_DIR}"
    )


# ---------------------------------------------------------------------------
# Load existing index from disk
# ---------------------------------------------------------------------------
def _load_index():
    """
    Load a previously saved FAISS index and metadata from disk.
    Returns True if loaded successfully, False otherwise.
    """
    global _faiss_index, _metadata

    if _faiss_index is not None and _metadata is not None:
        return True  # already loaded

    if not os.path.isfile(INDEX_FILE) or not os.path.isfile(METADATA_FILE):
        return False

    try:
        import faiss
    except ImportError:
        return False

    try:
        _faiss_index = faiss.read_index(INDEX_FILE)

        with open(METADATA_FILE, "rb") as f:
            _metadata = pickle.load(f)

        return True
    except Exception as e:
        print(f"[WARNING] Could not load RAG index: {e}")
        _faiss_index = None
        _metadata = None
        return False


# ---------------------------------------------------------------------------
# Search (Section 20.3)
# ---------------------------------------------------------------------------
def search(query, top_k=3):
    """
    Search the FAISS index for the most relevant document chunks.

    Args:
        query:  The user's question/search string.
        top_k:  Number of top results to return (default 3).

    Returns:
        List of dicts with keys: filename, chunk_num, text, score.
        Returns an empty list if no index is available.
    """
    # Try loading from disk if not already in memory
    if not _load_index():
        return []

    model = _load_embedding_model()
    if model is None:
        return []

    # Embed the query
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype="float32")

    # Search FAISS index
    scores, indices = _faiss_index.search(query_embedding, min(top_k, len(_metadata)))

    results = []
    for score, idx in zip(scores[0], indices[0], strict=False):
        if idx < 0 or idx >= len(_metadata):
            continue  # invalid index
        entry = _metadata[idx].copy()
        entry["score"] = float(score)
        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# RAG-powered Q&A (Section 20.4)
# ---------------------------------------------------------------------------
def ask(question):
    """
    Answer a question using RAG: search the index for relevant chunks,
    build a context-augmented prompt, and ask the LLM.

    Args:
        question: The user's question about their documents.

    Returns:
        The LLM's answer grounded in the retrieved document chunks,
        or an error/status message.
    """
    if not question or not question.strip():
        return "Please ask a question."

    # Check if index exists
    if not _load_index():
        return ("No RAG index found. Please index a folder first.\n"
                "  Example: \"reindex my files in C:\\Documents\"")

    print("  Searching your documents...")

    # Search for relevant chunks
    results = search(question, top_k=3)

    if not results:
        return ("No relevant documents found for your question.\n"
                "Try reindexing your files or rephrasing the question.")

    # Build context from retrieved chunks
    context_parts = []
    sources = set()
    for r in results:
        context_parts.append(
            f"[From: {r['filename']}, Chunk {r['chunk_num']}]\n{r['text']}"
        )
        sources.add(r["filename"])

    context = "\n\n---\n\n".join(context_parts)

    # Build the RAG prompt
    prompt = (
        "Based on the following documents, answer the user's question.\n"
        "Use ONLY information from the documents below. If the documents don't "
        "contain enough information to answer, say so.\n\n"
        f"=== DOCUMENTS ===\n{context}\n\n"
        f"=== QUESTION ===\n{question}\n\n"
        "=== ANSWER ==="
    )

    # Call the LLM
    from nexus.llm.loader import generate
    print("  Generating answer from your documents...")
    answer = generate(prompt, max_tokens=300)

    # Append source references
    source_list = ", ".join(sorted(sources))
    return f"{answer}\n\n📄 Sources: {source_list}"


# ---------------------------------------------------------------------------
# Check if RAG index exists
# ---------------------------------------------------------------------------
def has_index():
    """Return True if a RAG index exists on disk."""
    return os.path.isfile(INDEX_FILE) and os.path.isfile(METADATA_FILE)


def get_index_info():
    """Return a summary string about the current RAG index, or None."""
    if not os.path.isfile(CONFIG_FILE):
        return None

    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            config = json.load(f)
        return (
            f"RAG Index Status:\n"
            f"  Folder: {config.get('folder_path', 'unknown')}\n"
            f"  Files: {config.get('total_files', '?')}\n"
            f"  Chunks: {config.get('total_chunks', '?')}\n"
            f"  Model: {config.get('embedding_model', '?')}"
        )
    except Exception:
        return None
