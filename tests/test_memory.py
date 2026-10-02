"""Tests for persistent long-term memory."""

import json

import pytest

from nexus.agent import memory
from nexus.agent.router import fast_route


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    """Point memory at a throwaway file so tests never touch real user data."""
    target = tmp_path / "user_memory.json"
    monkeypatch.setattr(memory, "MEMORY_FILE", str(target))
    return target


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def test_remember_then_load(isolated_memory):
    memory.remember_fact("my birthday is 15 March")
    facts = memory.load_all_facts()
    assert len(facts) == 1
    assert "15 March" in facts[0]["text"]


def test_load_all_facts_returns_a_list_not_a_string():
    # The About dialog calls len() on this; recall_facts() returns display text.
    assert isinstance(memory.load_all_facts(), list)


def test_empty_store_starts_empty():
    assert memory.load_all_facts() == []


def test_missing_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MEMORY_FILE", str(tmp_path / "absent.json"))
    assert memory.load_memory() == {"facts": []}


def test_corrupt_file_degrades_to_empty(isolated_memory):
    isolated_memory.write_text("not json at all", encoding="utf-8")
    assert memory.load_memory() == {"facts": []}


def test_wrong_shape_degrades_to_empty(isolated_memory):
    isolated_memory.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert memory.load_memory() == {"facts": []}


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------
def test_recall_with_no_query_lists_everything():
    memory.remember_fact("I like tea")
    memory.remember_fact("I live in Nashik")
    out = memory.recall_facts(None)
    assert "tea" in out
    assert "Nashik" in out


def test_recall_filters_by_keyword():
    memory.remember_fact("I like tea")
    memory.remember_fact("I live in Nashik")
    out = memory.recall_facts("Nashik")
    assert "Nashik" in out
    assert "tea" not in out


# ---------------------------------------------------------------------------
# Forget
# ---------------------------------------------------------------------------
def test_forget_by_index():
    memory.remember_fact("first fact")
    memory.remember_fact("second fact")
    memory.forget_fact("1")
    remaining = memory.load_all_facts()
    assert len(remaining) == 1
    assert "second" in remaining[0]["text"]


def test_clear_all_removes_everything():
    memory.remember_fact("a")
    memory.remember_fact("b")
    memory.clear_all_memories()
    assert memory.load_all_facts() == []


# ---------------------------------------------------------------------------
# Router integration -- regression for the "search for the whole sentence" bug
# ---------------------------------------------------------------------------
def test_broad_recall_phrase_produces_no_filter():
    # "what do you remember about me" used to be passed through as a search
    # term, so it matched nothing and reported an empty store.
    assert fast_route("what do you remember about me") == ("ACTION:RECALL", "")


def test_targeted_recall_keeps_the_search_term():
    assert fast_route("recall my birthday") == ("ACTION:RECALL", "birthday")
