"""Tests for aim_resolve.__version__."""

from aim_resolve.__version__ import __version__


def test_version_is_string():
    assert isinstance(__version__, str)


def test_version_has_parts():
    parts = __version__.split(".")
    assert len(parts) >= 2
    for part in parts:
        assert part.isdigit()
