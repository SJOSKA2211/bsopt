from pathlib import Path

import pytest

from src.utils.filesystem import sanitize_path


def test_sanitize_path_valid():
    base = Path("/tmp/rick").resolve()
    # Path might not exist, but resolve() still works
    res = sanitize_path(base, "morty.txt")
    assert res == base / "morty.txt"


def test_sanitize_path_traversal():
    base = Path("/tmp/rick").resolve()
    with pytest.raises(ValueError) as excinfo:
        sanitize_path(base, "../../etc/passwd")
    assert "Path traversal detected" in str(excinfo.value)


def test_sanitize_path_relative_base():
    # It should resolve relative base dirs
    base = Path(".")
    res = sanitize_path(base, "file.txt")
    assert res.is_absolute()
