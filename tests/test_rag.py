"""Tests for the RAG pipeline: indexing, retrieval, and graceful degradation.

Exercises real embedding-based retrieval against a small fixture folder --
not mocked embeddings -- since the whole point is confirming the actual
sentence-transformers + FAISS pipeline finds the right chunk. Requires the
optional `rag` extra (sentence-transformers, faiss-cpu); skipped entirely
when it isn't installed, matching the "RAG stays optional" decision in
docs/decisions.md (45.1) -- a missing extra is not a test failure.
"""

import pytest

pytest.importorskip("faiss", reason="faiss-cpu not installed (optional `rag` extra)")
pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed (optional `rag` extra)")

from nexus.security import permissions  # noqa: E402
from nexus.tools import rag  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_rag(tmp_path, monkeypatch):
    """Point the RAG index + permissions at a throwaway location."""
    index_dir = tmp_path / "rag-index"
    monkeypatch.setattr(rag, "RAG_INDEX_DIR", str(index_dir))
    monkeypatch.setattr(rag, "INDEX_FILE", str(index_dir / "faiss.index"))
    monkeypatch.setattr(rag, "METADATA_FILE", str(index_dir / "metadata.pkl"))
    monkeypatch.setattr(rag, "CONFIG_FILE", str(index_dir / "index_config.json"))
    monkeypatch.setattr(permissions, "CONFIG_PATH", str(tmp_path / "permissions.json"))
    rag._faiss_index = None
    rag._metadata = None
    yield
    rag._faiss_index = None
    rag._metadata = None


@pytest.fixture
def fixture_folder(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "pets.txt").write_text(
        "My cat's name is Whiskers. Whiskers loves to sleep on the windowsill all day.",
        encoding="utf-8",
    )
    (d / "finance.txt").write_text(
        "Quarterly revenue grew 12% year over year, driven by subscription renewals.",
        encoding="utf-8",
    )
    permissions.save_permissions([str(d)])
    return d


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def test_build_index_reports_files_and_chunks(fixture_folder):
    result = rag.build_index(str(fixture_folder))
    assert "RAG index built successfully" in result
    assert "Files indexed: 2" in result
    assert rag.has_index()


def test_build_index_denied_outside_allowed_folder(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.txt").write_text("hello")
    assert rag.build_index(str(outside)) == "Permission denied for this folder."


def test_build_index_reports_no_supported_files(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(permissions, "CONFIG_PATH", str(tmp_path / "perms2.json"))
    permissions.save_permissions([str(empty)])
    assert "No supported files found" in rag.build_index(str(empty))


# ---------------------------------------------------------------------------
# Retrieval -- the hero-feature behaviour: a known query should surface
# the chunk that actually answers it, not just any chunk.
# ---------------------------------------------------------------------------
def test_search_returns_the_right_chunk_for_a_known_query(fixture_folder):
    rag.build_index(str(fixture_folder))
    results = rag.search("what is my cat's name", top_k=1)
    assert results
    assert results[0]["filename"] == "pets.txt"


def test_search_with_no_index_returns_empty_list():
    assert rag.search("anything") == []


# ---------------------------------------------------------------------------
# ask() -- grounded answers and graceful degradation
# ---------------------------------------------------------------------------
def test_ask_grounds_the_answer_in_retrieved_sources(fixture_folder, monkeypatch):
    rag.build_index(str(fixture_folder))
    monkeypatch.setattr("nexus.llm.loader.generate", lambda prompt, max_tokens=300: "Whiskers.")
    answer = rag.ask("what is my cat's name?")
    assert "Whiskers" in answer
    assert "pets.txt" in answer


def test_ask_with_no_relevant_chunks_degrades_gracefully(fixture_folder, monkeypatch):
    rag.build_index(str(fixture_folder))
    monkeypatch.setattr(rag, "search", lambda query, top_k=3: [])
    answer = rag.ask("anything")
    assert "No relevant documents found" in answer


def test_ask_with_no_index_prompts_to_index_first():
    answer = rag.ask("what is my cat's name?")
    assert "No RAG index found" in answer


def test_ask_rejects_empty_question():
    assert rag.ask("   ") == "Please ask a question."
