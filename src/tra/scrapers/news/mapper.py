"""Map article text to PSX symbols.

A wrong link is worse than a missing one: it would feed one company's news into another company's signal.
So a match needs the company's name as whole words, or an upper-case ticker that cannot be an ordinary word.
Short or generic names and ambiguous aliases are not matched.
"""

from __future__ import annotations

import re

from tra.scrapers.news.sources import company_search_name

# Tickers that are ordinary words or common abbreviations in business news
AMBIGUOUS_TICKERS = {
    "ABL", "AGP", "APL", "ATLH", "BOP", "CEO", "FFC", "HUM", "ICI", "IMF", "LUCK", "MARI", "NATF", "NBP",
    "NML", "OGDC", "PACE", "PIA", "POL", "POWER", "PSO", "PSX", "PTC", "SBP", "SECP", "SYS", "TRG", "UBL",
}  # fmt: skip
# Names that are also ordinary phrases once 'Limited' is removed
GENERIC_NAMES = {
    "power cement",
    "image pakistan",
    "the bank of punjab",
    "pakistan stock exchange",  # headlines naming the exchange are market news, not company news
}
# Common short forms used by Pakistani business press
ALIASES = {
    "OGDC": ["OGDCL", "Oil and Gas Development"],
    "PPL": ["Pakistan Petroleum"],
    "PSO": ["Pakistan State Oil"],
    "HBL": ["Habib Bank"],
    "UBL": ["United Bank"],
    "NBP": ["National Bank of Pakistan"],
    "MCB": ["MCB Bank"],
    "MEBL": ["Meezan Bank"],
    "LUCK": ["Lucky Cement"],
    "ENGROH": ["Engro Holdings"],
    "EFERT": ["Engro Fertilizers", "Engro Fertiliser"],
    "FFC": ["Fauji Fertilizer Company", "Fauji Fertiliser Company"],
    "SYS": ["Systems Limited"],
    "HUBC": ["Hub Power", "Hubco"],
    "TRG": ["TRG Pakistan"],
}


class SymbolMapper:
    def __init__(self, names: dict[str, str]) -> None:
        """``names`` maps symbol to full company name."""
        phrases: dict[str, set[str]] = {}
        for symbol, name in names.items():
            short = company_search_name(name)
            forms = [short, name, *ALIASES.get(symbol, [])]
            for form in forms:
                key = " ".join(form.lower().split())
                if len(key.split()) >= 2 and key not in GENERIC_NAMES:
                    phrases.setdefault(key, set()).add(symbol)
        # A phrase shared by several companies identifies none of them
        unique = {k: next(iter(v)) for k, v in phrases.items() if len(v) == 1}
        self._phrase_re = (
            re.compile(
                r"\b(" + "|".join(re.escape(k) for k in sorted(unique, key=len, reverse=True)) + r")\b", re.I
            )
            if unique
            else None
        )
        self._phrases = unique
        tickers = [s for s in names if len(s) >= 3 and s not in AMBIGUOUS_TICKERS]
        self._ticker_re = (
            re.compile(r"(?<![A-Za-z-])(" + "|".join(map(re.escape, tickers)) + r")(?![A-Za-z-])")
            if tickers
            else None
        )

    def symbols_in(self, text: str) -> set[str]:
        found: set[str] = set()
        if self._phrase_re:
            for m in self._phrase_re.finditer(text):
                found.add(self._phrases[" ".join(m.group(1).lower().split())])
        if self._ticker_re:
            found |= set(self._ticker_re.findall(text))
        return found
