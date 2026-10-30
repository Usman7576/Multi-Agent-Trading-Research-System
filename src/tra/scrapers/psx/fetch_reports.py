"""Download result-announcement PDFs and turn them into point-in-time fundamentals.

Input: stored announcements whose title is "Financial Results for the ... Ended <date>". Each PDF is archived
under ``data/raw/psx/reports/<symbol>/<announcement_id>.pdf``, read (text or OCR), extracted by rules and,
when a local model is given, by the LLM as well. One row per announcement goes to ``psx_results``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from tra.data.lake import ParquetLake
from tra.data.psx_financials import agreement, llm_extract, needs_llm, pdf_text, rule_extract, to_record
from tra.scrapers.psx.client import PSXClient
from tra.scrapers.psx.parsers.announcements import result_period_end

log = logging.getLogger(__name__)
MARKET = "PSX"


def result_announcements(lake: ParquetLake, symbols: Iterable[str], since: str) -> pd.DataFrame:
    ann = lake.read("announcements", MARKET)
    ann = ann[ann["symbol"].isin(set(symbols)) & ann["doc_path"].notna()].copy()
    ann["period_end"] = ann["title"].map(result_period_end)
    ann = ann[ann["period_end"].notna() & (ann["period_end"] >= pd.Timestamp(since))]
    return ann.sort_values(["symbol", "known_at"])


def process_reports(
    client: PSXClient,
    lake: ParquetLake,
    raw_dir: Path,
    announcements: pd.DataFrame,
    llm=None,
    audit_every: int = 20,
) -> pd.DataFrame:
    """Rules first; the LLM runs when rules are incomplete or fail the accounting checks, and on every
    ``audit_every``-th report to measure rules-vs-LLM agreement. Filings that still fail are stored with
    method 'failed' and are not used as features."""
    done: set[str] = set()
    if lake.exists("psx_results", MARKET):
        done = set(lake.read("psx_results", MARKET)["announcement_id"].astype(str))
    rows = []
    for a in announcements.itertuples(index=False):
        if str(a.announcement_id) in done:
            continue
        path = raw_dir / "reports" / a.symbol / f"{a.announcement_id}.pdf"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(client.download(a.doc_path))
        try:
            text, text_method = pdf_text(path.read_bytes())
        except Exception as exc:  # corrupt or encrypted PDF: record it, never guess
            log.warning("%s %s: unreadable PDF (%s)", a.symbol, a.announcement_id, exc)
            text, text_method = "", "unreadable"
        rules = rule_extract(text, a.title)
        final, method = rules, "rules"
        rec: dict[str, object] = {
            "announcement_id": str(a.announcement_id),
            "symbol": a.symbol,
            "title": a.title,
            "period_end": a.period_end,
            "known_at": a.known_at,
            "text_method": text_method,
            "text_chars": len(text),
        }
        fallback = needs_llm(rules)
        audit = len(rows) % audit_every == 0  # a sample where both run, to measure agreement
        if llm is not None and text_method in {"text", "ocr"} and (fallback or audit):
            try:
                # units are a document-level fact detected deterministically; the 3B model often misses them
                model = llm_extract(llm, text, a.title).model_copy(update={"unit": rules.unit})
                rec.update({f"llm_{k}": v for k, v in to_record(model, "llm").items()})
                rec.update({f"agree_{k}": v for k, v in agreement(rules, model).items()})
                if fallback and not needs_llm(model):
                    final, method = model, "llm"
            except Exception as exc:
                rec["llm_error"] = str(exc)[:300]
        if needs_llm(final):
            method = "failed"  # incomplete or inconsistent: never stored as a usable figure
        rec.update(to_record(final, method))
        rows.append(rec)
        if len(rows) % 25 == 0:
            lake.upsert("psx_results", MARKET, pd.DataFrame(rows))
            log.info("Processed %d result PDFs (latest %s %s)", len(rows), a.symbol, a.period_end.date())
    if rows:
        lake.upsert("psx_results", MARKET, pd.DataFrame(rows))
    return pd.DataFrame(rows)
