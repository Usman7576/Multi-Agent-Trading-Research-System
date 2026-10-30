"""Parsers for company announcements and payouts. Pure functions: no network access."""

from __future__ import annotations

import hashlib
import re

import pandas as pd
from bs4 import BeautifulSoup

from tra.scrapers.common.drift import DriftError, require_headers

TIMEZONE = "Asia/Karachi"
_TOTAL = re.compile(r"of\s+([\d,]+)\s+entries")
_DOC = re.compile(r"/download/document/(\d+)\.pdf")
_PERIOD = re.compile(r"(\d{2}/\d{2}/\d{4})\s*\(([A-Z]+)\)")
_PAYOUT = re.compile(r"([\d.]+)\s*%\s*(?:\([ivx]+\)|\(F\))?\s*\(([DBR])\)", re.IGNORECASE)


def parse_announcements(html: str, source: str = "announcements") -> tuple[pd.DataFrame, int]:
    """Rows of one announcements page and the total number of entries reported by the page."""
    soup = BeautifulSoup(html, "lxml")
    total_match = _TOTAL.search(soup.get_text(" ", strip=True))
    table = soup.find("table", id="announcementsTable")
    if table is None:
        if total_match and int(total_match.group(1).replace(",", "")) == 0:
            return _empty(), 0
        raise DriftError(f"{source}: announcements table not found")
    headers = [th.get_text(" ", strip=True) for th in table.select("thead th")]
    idx = require_headers(headers, {"DATE", "TIME", "SYMBOL", "NAME", "TITLE"}, source)
    rows = []
    for tr in table.select("tbody tr"):
        cells = tr.find_all("td")
        if len(cells) < len(idx):
            continue
        pdf = tr.find("a", href=_DOC)
        doc_id = _DOC.search(pdf["href"]).group(1) if pdf else None
        date_text = cells[idx["DATE"]].get_text(" ", strip=True)
        time_text = cells[idx["TIME"]].get_text(" ", strip=True)
        local = pd.to_datetime(f"{date_text} {time_text}", format="%b %d, %Y %I:%M %p")
        symbol = cells[idx["SYMBOL"]].get_text(" ", strip=True)
        title = " ".join(cells[idx["TITLE"]].get_text(" ", strip=True).split())
        rows.append(
            {
                "announcement_id": doc_id or f"{symbol}-{local:%Y%m%d%H%M}-{_digest(title)}",
                "symbol": symbol,
                "title": title,
                "announced_local": local,
                "known_at": local.tz_localize(TIMEZONE).tz_convert("UTC"),
                "doc_path": f"/download/document/{doc_id}.pdf" if doc_id else None,
            }
        )
    total = int(total_match.group(1).replace(",", "")) if total_match else len(rows)
    return (pd.DataFrame(rows) if rows else _empty()), total


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["announcement_id", "symbol", "title", "announced_local", "known_at", "doc_path"]
    )


RESULT_TITLE = re.compile(
    r"financial results? for the (?:quarter|half[- ]year|year|period|nine months?)\s+ended\s+"
    r"(\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


def result_period_end(title: str) -> pd.Timestamp | None:
    """Period end from titles like 'Financial Results for the Quarter Ended 2026-09-30'."""
    m = RESULT_TITLE.search(title)
    return pd.Timestamp(m.group(1)) if m else None


def parse_company_payouts(html: str, symbol: str, source: str = "payouts") -> pd.DataFrame:
    """Dividend (D), bonus (B) and right (R) announcements from a company's payouts table."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="tbl")
    if table is None:
        raise DriftError(f"{source}: payouts table not found")
    headers = [th.get_text(" ", strip=True) for th in table.select("thead th")]
    idx = require_headers(headers, {"DATE", "FINANCIAL RESULTS", "DETAILS", "BOOK CLOSURE"}, source)
    rows = []
    for tr in table.select("tbody tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(cells) < len(idx):
            continue
        announced = pd.to_datetime(cells[idx["DATE"]], format="%B %d, %Y %I:%M %p")
        period = _PERIOD.search(cells[idx["FINANCIAL RESULTS"]])
        closure = re.findall(r"\d{2}/\d{2}/\d{4}", cells[idx["BOOK CLOSURE"]])
        details = " ".join(cells[idx["DETAILS"]].split())
        for pct, kind in _PAYOUT.findall(details):
            rows.append(
                {
                    "symbol": symbol,
                    "announced_at": announced.tz_localize(TIMEZONE).tz_convert("UTC"),
                    "period_end": pd.to_datetime(period.group(1), format="%d/%m/%Y") if period else pd.NaT,
                    "period_code": period.group(2) if period else None,
                    "kind": {"D": "dividend", "B": "bonus", "R": "right"}[kind.upper()],
                    "percent": float(pct),
                    "details": details,
                    "book_closure_from": pd.to_datetime(closure[0], format="%d/%m/%Y") if closure else pd.NaT,
                }
            )
    return pd.DataFrame(rows)
