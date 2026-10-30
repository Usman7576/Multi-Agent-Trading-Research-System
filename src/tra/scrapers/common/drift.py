"""Page-structure drift detection: stop loudly instead of writing silently mis-parsed data."""

from __future__ import annotations


class DriftError(RuntimeError):
    pass


def require_headers(found: list[str], expected: set[str], source: str) -> dict[str, int]:
    """Check every expected header is present and return a header -> column-index map.

    Columns are always located by header text, never by position, so a reordered table still parses
    while a renamed or removed column is rejected.
    """
    normalised = [" ".join(h.split()).upper() for h in found]
    missing = sorted(expected - set(normalised))
    if missing:
        raise DriftError(f"{source}: expected headers {missing} not found; got {normalised}")
    return {h: i for i, h in enumerate(normalised)}
