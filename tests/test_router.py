"""
Tests for the keyword fast-path router.

The keyword router matches on bare word membership, so ordinary English that
happens to contain a trigger word gets dispatched to a tool. These tests pin
down both the intents that must work and the false positives that must not
fire.
"""

import pytest

from nexus.agent.router import fast_route


def action_of(text):
    """Return just the ACTION verb from fast_route, or None if it falls through."""
    result = fast_route(text)
    return result[0] if result else None


# ---------------------------------------------------------------------------
# Intents that must be recognised
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("what is my battery percentage?", "ACTION:BATTERY"),
        ("check battery", "ACTION:BATTERY"),
        ("am i charging", "ACTION:BATTERY"),
        ("list top 10 running processes", "ACTION:PROCESSES"),
        ("show running processes", "ACTION:PROCESSES"),
        ("what's running", "ACTION:PROCESSES"),
        ("shut down my computer", "ACTION:SHUTDOWN"),
        ("restart my pc", "ACTION:RESTART"),
        ("put my computer to sleep", "ACTION:SLEEP"),
        ("cancel shutdown", "ACTION:CANCEL_SHUTDOWN"),
        ("clear chat", "ACTION:CLEAR_HISTORY"),
        ("what model are you", "ACTION:MODEL_INFO"),
    ],
)
def test_recognised_intents(text, expected):
    assert action_of(text) == expected


# ---------------------------------------------------------------------------
# Persistent memory
# ---------------------------------------------------------------------------
def test_remember_extracts_the_fact_not_the_whole_sentence():
    action, params = fast_route("remember that my birthday is 15 March")
    assert action == "ACTION:REMEMBER"
    assert params == "my birthday is 15 March"


def test_recall_is_recognised():
    assert action_of("what do you remember about me") == "ACTION:RECALL"


def test_remember_beats_clear_history():
    # "forget everything" is a CLEAR_HISTORY keyword; "remember that" must win
    # when the user is clearly storing a fact.
    assert action_of("remember that I never forget everything") == "ACTION:REMEMBER"


# ---------------------------------------------------------------------------
# False positives -- ordinary English that must NOT trigger a tool
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "what is the price of freedom",
        "explain market economics to me",
        "what factors affect share prices in general",
        "how does the stock market work",
    ],
)
def test_price_words_do_not_trigger_stock_lookup(text):
    assert action_of(text) != "ACTION:FETCH_STOCK"


@pytest.mark.parametrize(
    "text",
    [
        "how do I copy a file in windows",
        "what does copy on write mean",
    ],
)
def test_copy_questions_do_not_trigger_file_copy(text):
    assert action_of(text) != "ACTION:COPY_FILE"


@pytest.mark.parametrize(
    "text",
    [
        "how do I move to a new city",
        "what is brownian motion",
    ],
)
def test_move_questions_do_not_trigger_file_move(text):
    assert action_of(text) != "ACTION:MOVE_FILE"


@pytest.mark.parametrize(
    "text",
    [
        "I need to charge my phone tonight",
        "explain how a battery works chemically",
    ],
)
def test_battery_words_in_general_questions(text):
    # A genuine question about batteries should be answered, not answered with
    # the laptop's charge level.
    assert action_of(text) != "ACTION:BATTERY"


def test_process_as_a_concept_is_not_a_task():
    assert action_of("explain the process of photosynthesis") != "ACTION:PROCESSES"


def test_sleep_as_a_topic_does_not_suspend_the_machine():
    assert action_of("how many hours of sleep do I need") != "ACTION:SLEEP"


def test_delete_question_does_not_delete():
    assert action_of("how do I delete a file permanently") != "ACTION:DELETE_FILE"


# ---------------------------------------------------------------------------
# Fall-through
# ---------------------------------------------------------------------------
def test_general_question_falls_through_or_is_qa():
    # Either outcome is acceptable; what matters is that it is not a tool call.
    assert action_of("what is machine learning") in (None, "ACTION:QA")


def test_greeting_is_qa():
    assert action_of("hello") == "ACTION:QA"


# ---------------------------------------------------------------------------
# Single source of truth (V4.0 Phase 38) -- commands.py's /task keyword
# table is built from this module's ACTIONS registry instead of a second,
# independently hand-typed dict. These guard against that invariant
# silently breaking as actions are added or renamed.
# ---------------------------------------------------------------------------
def test_task_keywords_only_reference_known_actions():
    from nexus.agent import commands
    from nexus.agent.router import ACTIONS

    task_only_extras = {"DISK_CLEANUP", "SET_WALLPAPER"}  # no NL/LLM equivalent
    for keyword, action_name in commands._TASK_KEYWORDS.items():
        assert action_name in ACTIONS or action_name in task_only_extras, (
            f"/task keyword {keyword!r} maps to {action_name!r}, which is "
            "neither in router.ACTIONS nor a known /task-only extra"
        )


def test_natural_language_and_task_command_agree_on_action_name():
    from nexus.agent import commands

    for text, task_keyword in [
        ("what is my battery", "battery"),
        ("show running processes", "processes"),
        ("cancel shutdown", "cancel shutdown"),
    ]:
        assert action_of(text) == f"ACTION:{commands._TASK_KEYWORDS[task_keyword]}"
