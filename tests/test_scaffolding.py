"""Basic scaffolding and import tests."""

import ai_dev_toolkit


def test_version() -> None:
    """Verify that version string is defined and non-empty."""
    assert isinstance(ai_dev_toolkit.__version__, str)
    assert len(ai_dev_toolkit.__version__.split(".")) >= 3
