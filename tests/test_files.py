"""Tests for tools/files.py permission edges and confirmation gating."""

import os

import pytest

from nexus.security import confirm, permissions
from nexus.tools import files


@pytest.fixture(autouse=True)
def isolated_permissions(tmp_path, monkeypatch):
    """Point permissions at a throwaway file; never touch the real config/."""
    monkeypatch.setattr(permissions, "CONFIG_PATH", str(tmp_path / "permissions.json"))
    confirm.set_handler(None)
    yield
    confirm.set_handler(None)


@pytest.fixture
def allowed_dir(tmp_path):
    d = tmp_path / "allowed"
    d.mkdir()
    permissions.save_permissions([str(d)])
    return d


@pytest.fixture
def outside_dir(tmp_path):
    d = tmp_path / "outside"
    d.mkdir()
    return d


def _approve():
    confirm.set_handler(lambda title, details: True)


def _deny():
    confirm.set_handler(lambda title, details: False)


# ---------------------------------------------------------------------------
# read_txt / resolve_filepath permission edges
# ---------------------------------------------------------------------------
def test_read_txt_denied_outside_allowed_folder(outside_dir):
    f = outside_dir / "secret.txt"
    f.write_text("top secret")
    assert files.read_txt(str(f)) == "Permission denied for this file."


def test_read_txt_allowed_inside_folder(allowed_dir):
    f = allowed_dir / "note.txt"
    f.write_text("hello")
    assert files.read_txt(str(f)) == "hello"


def test_resolve_filepath_finds_file_in_allowed_folder(allowed_dir):
    f = allowed_dir / "note.txt"
    f.write_text("hello")
    resolved = files.resolve_filepath("note.txt")
    assert os.path.abspath(resolved) == os.path.abspath(str(f))


def test_resolve_filepath_falls_back_to_original_when_not_found(allowed_dir):
    assert files.resolve_filepath("ghost.txt") == "ghost.txt"


# ---------------------------------------------------------------------------
# rename_file -- the Phase 36 destination-check fix: new_name could
# previously contain ".." or an absolute path and escape the allowed
# folder via os.rename, since only the source was permission-checked.
# ---------------------------------------------------------------------------
def test_rename_denied_when_source_outside_allowed(outside_dir):
    f = outside_dir / "a.txt"
    f.write_text("x")
    assert files.rename_file(str(f), "b.txt") == "Permission denied for this file."


def test_rename_denied_when_destination_escapes_allowed_folder(allowed_dir):
    f = allowed_dir / "a.txt"
    f.write_text("x")
    result = files.rename_file(str(f), "..\\escaped.txt")
    assert result == "Permission denied for the destination path."
    assert f.exists()  # original file must be untouched


def test_rename_succeeds_when_approved(allowed_dir):
    f = allowed_dir / "a.txt"
    f.write_text("x")
    _approve()
    result = files.rename_file(str(f), "b.txt")
    assert "renamed successfully" in result
    assert (allowed_dir / "b.txt").exists()
    assert not f.exists()


def test_rename_cancelled_when_denied(allowed_dir):
    f = allowed_dir / "a.txt"
    f.write_text("x")
    _deny()
    result = files.rename_file(str(f), "b.txt")
    assert "cancelled" in result.lower()
    assert f.exists()


def test_rename_rejects_existing_destination_name(allowed_dir):
    a = allowed_dir / "a.txt"
    a.write_text("x")
    b = allowed_dir / "b.txt"
    b.write_text("y")
    _approve()
    result = files.rename_file(str(a), "b.txt")
    assert "already exists" in result


# ---------------------------------------------------------------------------
# copy_file / move_file
# ---------------------------------------------------------------------------
def test_copy_denied_for_source_outside_allowed(outside_dir, allowed_dir):
    f = outside_dir / "a.txt"
    f.write_text("x")
    assert files.copy_file(str(f), str(allowed_dir)) == "Permission denied for the source file."


def test_copy_denied_for_destination_outside_allowed(allowed_dir, outside_dir):
    f = allowed_dir / "a.txt"
    f.write_text("x")
    result = files.copy_file(str(f), str(outside_dir / "a.txt"))
    assert result == "Permission denied for the destination folder."


def test_copy_succeeds_when_approved(allowed_dir):
    f = allowed_dir / "a.txt"
    f.write_text("x")
    dest_dir = allowed_dir / "sub"
    dest_dir.mkdir()
    _approve()
    result = files.copy_file(str(f), str(dest_dir))
    assert "copied successfully" in result
    assert (dest_dir / "a.txt").exists()
    assert f.exists()  # original remains after a copy


def test_move_succeeds_when_approved(allowed_dir):
    f = allowed_dir / "a.txt"
    f.write_text("x")
    dest_dir = allowed_dir / "sub"
    dest_dir.mkdir()
    _approve()
    result = files.move_file(str(f), str(dest_dir))
    assert "moved successfully" in result
    assert (dest_dir / "a.txt").exists()
    assert not f.exists()


# ---------------------------------------------------------------------------
# delete_file
# ---------------------------------------------------------------------------
def test_delete_denied_outside_allowed(outside_dir):
    f = outside_dir / "a.txt"
    f.write_text("x")
    assert files.delete_file(str(f)) == "Permission denied for this file."
    assert f.exists()


def test_delete_cancelled_when_denied(allowed_dir):
    f = allowed_dir / "a.txt"
    f.write_text("x")
    _deny()
    result = files.delete_file(str(f))
    assert "cancelled" in result.lower()
    assert f.exists()


def test_delete_succeeds_when_approved(allowed_dir):
    f = allowed_dir / "a.txt"
    f.write_text("x")
    _approve()
    result = files.delete_file(str(f))
    assert "deleted" in result.lower()
    assert not f.exists()


def test_delete_missing_file_reports_not_found(allowed_dir):
    missing = allowed_dir / "ghost.txt"
    result = files.delete_file(str(missing))
    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# save_text
# ---------------------------------------------------------------------------
def test_save_text_denied_outside_allowed(outside_dir):
    target = outside_dir / "log.txt"
    result = files.save_text("hello", str(target))
    assert result.startswith("Permission denied")
    assert not target.exists()


def test_save_text_creates_new_file(allowed_dir):
    target = allowed_dir / "log.txt"
    result = files.save_text("hello", str(target))
    assert "saved to" in result.lower()
    assert "hello" in target.read_text(encoding="utf-8")


def test_save_text_appends_to_existing_file(allowed_dir):
    target = allowed_dir / "log.txt"
    files.save_text("first", str(target))
    result = files.save_text("second", str(target))
    assert "appended to" in result.lower()
    content = target.read_text(encoding="utf-8")
    assert "first" in content and "second" in content


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------
def test_list_files_denied_outside_allowed(outside_dir):
    assert files.list_files(str(outside_dir)) == "Permission denied for this folder."


def test_list_files_reports_empty_folder(allowed_dir):
    assert files.list_files(str(allowed_dir)) == f"Folder is empty: {allowed_dir}"


def test_list_files_lists_entries(allowed_dir):
    (allowed_dir / "a.txt").write_text("x")
    (allowed_dir / "sub").mkdir()
    result = files.list_files(str(allowed_dir))
    assert "a.txt" in result
    assert "[DIR]" in result and "sub" in result


def test_list_files_missing_folder(allowed_dir):
    missing = str(allowed_dir / "nope")
    assert files.list_files(missing) == f"Folder not found: {missing}"
