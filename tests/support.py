"""Shared helpers for the test suite."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def temporary_directory() -> Iterator[Path]:
    """Yield a temporary directory with every path component resolved.

    macOS places TMPDIR under ``/var/folders`` and ``/var`` is a symlink to
    ``/private/var``. The production symlink guards would therefore reject every
    temporary path on that platform, which both fails honest tests and makes the
    dedicated symlink-rejection tests pass for the wrong reason.
    """
    with tempfile.TemporaryDirectory() as raw_directory:
        yield Path(raw_directory).resolve()
