"""
Tests for the destructive-action confirmation gateway.

The critical property is that it fails closed: anything that is not an explicit
approval must be treated as a refusal, because callers use the return value to
decide whether to delete files or power off the machine.
"""

import pytest

from nexus.security import confirm


@pytest.fixture(autouse=True)
def _clear_handler():
    """Never leak a handler between tests."""
    confirm.set_handler(None)
    yield
    confirm.set_handler(None)


def test_denies_when_no_handler_registered():
    assert confirm.request("Delete everything?", "really") is False


def test_has_handler_reflects_registration():
    assert confirm.has_handler() is False
    confirm.set_handler(lambda title, details: True)
    assert confirm.has_handler() is True
    confirm.set_handler(None)
    assert confirm.has_handler() is False


def test_approval_is_passed_through():
    confirm.set_handler(lambda title, details: True)
    assert confirm.request("Proceed?") is True


def test_refusal_is_passed_through():
    confirm.set_handler(lambda title, details: False)
    assert confirm.request("Proceed?") is False


def test_handler_receives_title_and_details():
    seen = {}

    def handler(title, details):
        seen["title"] = title
        seen["details"] = details
        return True

    confirm.set_handler(handler)
    confirm.request("Delete report.pdf?", "Location: D:/docs")
    assert seen["title"] == "Delete report.pdf?"
    assert seen["details"] == "Location: D:/docs"


def test_broken_handler_is_not_read_as_approval():
    def exploding(title, details):
        raise RuntimeError("UI is gone")

    confirm.set_handler(exploding)
    assert confirm.request("Shut down?") is False


def test_non_boolean_truthy_return_is_coerced():
    confirm.set_handler(lambda title, details: "yes")
    assert confirm.request("Proceed?") is True


def test_none_return_is_a_refusal():
    confirm.set_handler(lambda title, details: None)
    assert confirm.request("Proceed?") is False


def test_denied_message_explains_when_no_frontend():
    msg = confirm.denied_message("Delete")
    assert "no confirmation prompt" in msg.lower()


def test_denied_message_is_plain_cancel_when_frontend_exists():
    confirm.set_handler(lambda title, details: False)
    assert confirm.denied_message("Delete") == "Delete cancelled."
