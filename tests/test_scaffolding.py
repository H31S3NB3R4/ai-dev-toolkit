"""Basic scaffolding and import tests."""

import ai_dev_toolkit


def test_version() -> None:
    """Verify that version string is defined."""
    assert ai_dev_toolkit.__version__ == "0.1.0"
