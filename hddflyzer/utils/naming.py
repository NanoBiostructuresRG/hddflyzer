# SPDX-License-Identifier: LGPL-3.0-or-later

"""Naming helpers."""

import os
from pathlib import Path


def sanitize_tag(tag: str) -> str:
    """
    Sanitize a tag string for safe use in file paths.

    Replaces path separators with underscores and strips whitespace.
    """
    return tag.strip().replace("/", "_").replace("\\", "_")


def validate_tag(tag: str) -> str:
    """Return a safe tag or raise ValueError for path-like input."""
    if not isinstance(tag, str):
        raise ValueError("Tag must be a string.")
    clean = tag.strip()
    if not clean:
        raise ValueError("Tag must not be empty.")
    if clean in {".", ".."}:
        raise ValueError(f"Unsafe tag: {tag!r}")
    if os.path.isabs(clean):
        raise ValueError(f"Tag must not be an absolute path: {tag!r}")
    if "/" in clean or "\\" in clean:
        raise ValueError(f"Tag must not contain path separators: {tag!r}")
    if ".." in Path(clean).parts:
        raise ValueError(f"Tag must not contain traversal: {tag!r}")
    return clean
