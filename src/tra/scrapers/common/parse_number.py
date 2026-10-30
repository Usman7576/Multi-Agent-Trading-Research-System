"""One shared number parser for scraped tables."""

from __future__ import annotations

import math
import re

_MISSING = {"", "-", "--", "n/a", "na", "nil"}
_CLEAN = re.compile(r"[,\s%]|Rs\.?|PKR", re.IGNORECASE)


def parse_number(text: str | None) -> float:
    """Parse '1,234.5', '(120)', '-0.15', '0.42%', 'Rs. 12' and blanks.

    Parentheses mean negative (accounting style). Missing markers return NaN, never 0. Percent values are
    returned as written (``'0.42%'`` -> 0.42); callers decide whether to divide by 100.
    """
    if text is None:
        return math.nan
    s = text.strip()
    if s.lower() in _MISSING:
        return math.nan
    negative = s.startswith("(") and s.endswith(")")
    s = _CLEAN.sub("", s.strip("()"))
    if s.lower() in _MISSING:
        return math.nan
    try:
        value = float(s)
    except ValueError as exc:
        raise ValueError(f"Cannot parse number from {text!r}") from exc
    return -value if negative else value
