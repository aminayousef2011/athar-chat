import pytest

from athar.timeexpr import cairo


@pytest.fixture
def ref():
    """Wednesday 7 Oct 2026, 10:00 Cairo (UTC+03:00, DST until 29 Oct)."""
    return cairo("2026-10-07T10:00")
