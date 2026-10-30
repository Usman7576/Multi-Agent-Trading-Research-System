"""PSX financial results: PDF -> text (OCR for scans) -> numbers -> accounting checks.

Listed companies file a "Financial Results for the ... Ended YYYY-MM-DD" announcement with a short profit
and loss summary. There is no structured (XBRL-like) feed, so this module builds one:

1. text: pdfplumber for text PDFs; scanned PDFs are rendered with pypdfium2 and read with RapidOCR
   (both pip-only, optional ``ocr`` extra). Scans without OCR available are marked ``needs_ocr``.
2. extraction: a rule-based reader (label dictionary + unit detection), and optionally the local LLM
   (``LocalLLM``) constrained to the ``ResultsExtract`` schema. When both run, agreement is recorded.
3. checks: profit before tax - tax = profit after tax (within 1%), EPS sign matches profit sign, and the
   share count implied by profit / EPS is stable across a company's filings (catches unit errors).

``known_at`` of every figure is the announcement timestamp, never the period end.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------- text
MIN_TEXT_CHARS = 200


def _has_pnl(text: str) -> bool:
    """True once the text holds a complete profit and loss block (profit after tax and EPS)."""
    low = re.sub(r"\s+", "", text.lower())
    return "aftertax" in low and "pershare" in low


def pdf_text(pdf: bytes, max_pages: int = 12) -> tuple[str, str]:
    """Return (text, method) where method is 'text', 'ocr' or 'needs_ocr'.

    Pages are read in order and reading stops at the first page that completes a profit and loss block,
    which keeps OCR of long annexed reports affordable.
    """
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf)) as doc:
        for page in doc.pages[:max_pages]:
            pages.append(page.extract_text() or "")
            if _has_pnl("\n".join(pages)):
                break
    text = "\n".join(pages)
    if len(text.strip()) >= MIN_TEXT_CHARS:
        return text, "text"
    try:
        return _ocr(pdf, max_pages), "ocr"
    except ImportError:
        return text, "needs_ocr"


def _ocr(pdf: bytes, max_pages: int) -> str:
    import pypdfium2 as pdfium
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    lines: list[str] = []
    doc = pdfium.PdfDocument(pdf)
    for i in range(min(len(doc), max_pages)):
        if _has_pnl("\n".join(lines)):
            break
        result = None
        for scale in (2.0, 1.4):  # large scanned pages can exhaust memory at full resolution
            try:
                result, _ = engine(doc[i].render(scale=scale).to_numpy())
                break
            except Exception as exc:  # ONNX runtime raises its own error types (e.g. bad allocation)
                log.warning("OCR page %d at scale %.1f failed: %s", i + 1, scale, str(exc)[:120])
        # group detected boxes into lines by their vertical position
        boxes = sorted(result or [], key=lambda r: (round(r[0][0][1] / 12), r[0][0][0]))
        current_y, current = None, []
        for box, txt, _score in boxes:
            y = round(box[0][1] / 12)
            if current_y is not None and y != current_y:
                lines.append(" ".join(current))
                current = []
            current_y = y
            current.append(txt)
        if current:
            lines.append(" ".join(current))
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------- extraction
class ResultsExtract(BaseModel):
    """Current-period figures from one results announcement, in the units stated in the document."""

    period_months: int | None = Field(None, description="Length of the reported period: 3, 6, 9 or 12")
    revenue: float | None = Field(None, description="Net sales / revenue / turnover / total income")
    gross_profit: float | None = None
    profit_before_tax: float | None = None
    taxation: float | None = Field(None, description="Tax expense as a positive number")
    profit_after_tax: float | None = Field(None, description="Profit (loss) after tax for the period")
    eps: float | None = Field(None, description="Basic earnings per share in rupees")
    unit: str = Field("PKR", description="One of PKR, PKR_000, PKR_mn as stated in the document")


UNIT_MULTIPLIER = {"PKR": 1.0, "PKR_000": 1e3, "PKR_mn": 1e6, "PKR_bn": 1e9}

# Patterns run on lower-cased lines with all whitespace removed (robust to OCR spacing).
LABELS: dict[str, list[str]] = {
    "revenue": [
        r"^net(?:sales|revenue|turnover)",
        r"^(?:sales|revenue|turnover)[-–]?net",
        r"^(?:sales|revenue|turnover)(?:\(net\))?[\d(,]",
        r"^totalincome",
        r"^mark-?up/return/interestearned",
    ],
    "gross_profit": [r"^gross(?:profit|\(loss\)|loss)"],
    "profit_before_tax": [r"^(?:profit|\(loss\)|loss)[^\d]{0,20}beforetax"],
    "taxation": [r"^tax(?:ation)?(?:-net)?[\d(,-]", r"^incometaxexpense"],
    "profit_after_tax": [
        r"^(?:profit|\(loss\)|loss)[^\d]{0,20}aftertax",
        r"^net(?:profit|income)forthe(?:period|quarter|year)",
        r"^(?:profit|\(loss\)|profit/\(loss\))forthe(?:period|quarter|year)",
    ],
    "eps": [
        r"^(?:basic)?(?:and)?(?:diluted)?earnings(?:/\(?loss\)?)?pershare",
        r"^(?:loss|profit)pershare",
        r"^eps",
    ],
}
_NUMBER = re.compile(r"\(?-?[\d,]+(?:\.\d+)?\)?")
_UNITS = [
    (re.compile(r"(?:rs\.?|rupees|pkr)\s*(?:in\s*)?(?:'?000|thousands?)|\(?'000\)?", re.I), "PKR_000"),
    (re.compile(r"(?:rs\.?|rupees|pkr)\s*(?:in\s*)?(?:millions?|mn|m\b)|in\s+millions?", re.I), "PKR_mn"),
    (re.compile(r"(?:rs\.?|rupees|pkr)\s*(?:in\s*)?(?:billions?|bn)", re.I), "PKR_bn"),
]
_MONTHS = [
    (re.compile(r"nine\s+months|9\s*months|third\s+quarter", re.I), 9),
    (re.compile(r"half[- ]year|six\s+months|6\s*months|second\s+quarter", re.I), 6),
    (re.compile(r"(?:year|annual)\s+ended|twelve\s+months|12\s*months", re.I), 12),
    (re.compile(r"(?:first\s+)?qua?rter\s+ended|three\s+months|3\s*months", re.I), 3),
]


def _numbers(line: str, amounts: bool = True) -> list[float]:
    """Numbers on a line. For amounts (not per-share figures) OCR often reads thousands separators as
    dots ("131.863.008"), so a dot followed by exactly three digits is treated as a separator."""
    out = []
    if amounts:
        line = re.sub(r"(?<=\d)\.(?=\d{3}(?:\D|$))", ",", line)
    for tok in _NUMBER.findall(line):
        neg = tok.startswith("(") and tok.endswith(")") or tok.startswith("-")
        digits = tok.strip("()-").replace(",", "")
        if not digits or digits == ".":
            continue
        if re.fullmatch(r"(19|20)\d\d", digits):  # a year in a column header, not a figure
            continue
        out.append(-float(digits) if neg else float(digits))
    return out


def rule_extract(text: str, title: str = "") -> ResultsExtract:
    """Label-dictionary extraction of the first (current-period) figure on each matching line."""
    found: dict[str, float] = {}
    for line in text.splitlines():
        # OCR often drops spaces ("Profitbeforetaxation"), so labels are matched without whitespace
        low = re.sub(r"\s+", "", line.lower())
        for field, patterns in LABELS.items():
            if field in found or not any(re.search(p, low) for p in patterns):
                continue
            nums = _numbers(line, amounts=field != "eps")
            if field != "eps":
                nums = [n for n in nums if abs(n) >= 1000]  # skip note references and stray small numbers
            if nums:
                found[field] = nums[0]
    unit = "PKR"
    for pattern, name in _UNITS:
        if pattern.search(text):
            unit = name
            break
    # PSX statements put the cumulative column first ("Nine months ended" before "Quarter ended"), so the
    # period length comes from the statement's own headers; the title decides only for annual reports.
    title_months = next((m for p, m in _MONTHS if p.search(title)), None)
    text_months = next((m for p, m in _MONTHS if p.search(_relevant_text(text))), None)
    months = 12 if title_months == 12 else (text_months or title_months)
    if "taxation" in found:
        found["taxation"] = abs(found["taxation"])
    return ResultsExtract(period_months=months, unit=unit, **found)


LLM_SYSTEM = (
    "You extract figures from a Pakistan Stock Exchange financial results announcement. Report only the "
    "CURRENT reporting period (the first figure column of the cumulative period, not the comparative "
    "period). Use the units stated in the document (PKR, PKR_000 for rupees in thousands, PKR_mn for "
    'millions). The unit is usually written above the figures, e.g. "Rupees in \'000" or "PKR in '
    "'000\" means PKR_000. Losses are negative. Use null for anything not present. Never invent numbers."
)
PROMPT_VERSION = "psx-results-v2"


def llm_extract(llm, text: str, title: str) -> ResultsExtract:  # llm: tra.llm.local.LocalLLM
    snippet = _relevant_text(text)
    return llm.extract(ResultsExtract, LLM_SYSTEM, f"Title: {title}\n\nDocument:\n{snippet}", PROMPT_VERSION)


def _relevant_text(text: str, before: int = 2500, after: int = 1200) -> str:
    """The profit and loss block: text around the first 'after tax' line (spacing-insensitive)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "aftertax" in re.sub(r"\s+", "", line.lower()):
            head = "\n".join(lines[:i])
            tail = "\n".join(lines[i:])
            return head[-before:] + "\n" + tail[:after]
    return text[: before + after]


def needs_llm(x: ResultsExtract) -> bool:
    """Rules are trusted when profit after tax and EPS were found and the accounting checks pass."""
    checks = accounting_checks(x)
    return (
        x.profit_after_tax is None or x.eps is None or checks.pbt_tax_pat is False or checks.eps_sign is False
    )


# ----------------------------------------------------------------------------------------------- checks
@dataclass(frozen=True)
class CheckResult:
    pbt_tax_pat: bool | None
    eps_sign: bool | None


def accounting_checks(x: ResultsExtract) -> CheckResult:
    pbt_ok = None
    if None not in (x.profit_before_tax, x.taxation, x.profit_after_tax):
        expected = x.profit_before_tax - x.taxation
        pbt_ok = abs(expected - x.profit_after_tax) <= 0.01 * max(abs(x.profit_after_tax), 1.0) or abs(
            x.profit_before_tax + x.taxation - x.profit_after_tax
        ) <= 0.01 * max(abs(x.profit_after_tax), 1.0)
    eps_ok = None
    if x.eps is not None and x.profit_after_tax is not None and x.eps != 0 and x.profit_after_tax != 0:
        eps_ok = bool(np.sign(x.eps) == np.sign(x.profit_after_tax))
    return CheckResult(pbt_ok, eps_ok)


def implied_share_stability(df: pd.DataFrame, tolerance: float = 0.35) -> pd.Series:
    """Flag filings whose implied share count (PAT in rupees / EPS) is far from the company's median.

    A unit error (thousands read as millions) moves the implied share count by 1000x, which this catches.
    Bonus issues move it modestly, hence the wide tolerance.
    """
    shares = (df["profit_after_tax_pkr"] / df["eps"]).where(df["eps"].abs() > 0)
    med = shares.groupby(df["symbol"]).transform("median")
    return ((shares / med - 1).abs() <= tolerance) | shares.isna()


def to_record(x: ResultsExtract, method: str) -> dict[str, object]:
    mult = UNIT_MULTIPLIER.get(x.unit, 1.0)
    checks = accounting_checks(x)
    rec: dict[str, object] = {
        "method": method,
        "unit": x.unit,
        "period_months": x.period_months,
        "eps": x.eps,
        "check_pbt_tax_pat": checks.pbt_tax_pat,
        "check_eps_sign": checks.eps_sign,
    }
    for f in ("revenue", "gross_profit", "profit_before_tax", "taxation", "profit_after_tax"):
        v = getattr(x, f)
        rec[f"{f}_pkr"] = None if v is None else v * mult
    return rec


def agreement(a: ResultsExtract, b: ResultsExtract, rel_tol: float = 0.01) -> dict[str, bool | None]:
    out: dict[str, bool | None] = {}
    for f in ("revenue", "profit_after_tax", "eps"):
        va, vb = getattr(a, f), getattr(b, f)
        if va is None or vb is None:
            out[f] = None
        else:
            sa, sb = (
                va * UNIT_MULTIPLIER[a.unit if f != "eps" else "PKR"],
                vb * UNIT_MULTIPLIER[b.unit if f != "eps" else "PKR"],
            )
            out[f] = abs(sa - sb) <= rel_tol * max(abs(sa), abs(sb), 1e-9)
    return out


def results_to_facts(results: pd.DataFrame) -> pd.DataFrame:
    """Usable results -> rows of the ``fundamentals`` table (same schema as US EDGAR facts).

    Duration figures get ``period_start = period_end - months``; the share count is implied by profit
    after tax / EPS and stored as an instant fact. Failed extractions and filings whose implied share
    count is inconsistent with the company's other filings (likely unit errors) are dropped.
    """
    ok = results[(results["method"] != "failed") & results["period_months"].notna()].copy()
    if ok.empty:
        return pd.DataFrame()
    ok = ok[implied_share_stability(ok)]
    ok["period_end"] = pd.to_datetime(ok["period_end"])
    start = ok.apply(
        lambda r: r["period_end"] - pd.DateOffset(months=int(r["period_months"])) + pd.Timedelta(days=1),
        axis=1,
    )
    shares = (ok["profit_after_tax_pkr"] / ok["eps"]).where(ok["eps"].abs() > 0)
    frames = []
    for metric, values, period_start in [
        ("net_income", ok["profit_after_tax_pkr"], start),
        ("revenue", ok["revenue_pkr"], start),
        ("shares_outstanding", shares.abs(), pd.Series(pd.NaT, index=ok.index)),
    ]:
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ok["symbol"],
                    "metric": metric,
                    "concept": f"psx_results_{metric}",
                    "priority": 0,
                    "value": values,
                    "period_start": period_start,
                    "period_end": ok["period_end"],
                    "form": "PSX",
                    "fy": ok["period_end"].dt.year,
                    "fp": ok["period_months"].astype(int).astype(str) + "M",
                    "accn": ok["announcement_id"],
                    "filed": ok["known_at"],
                    "known_at": ok["known_at"],
                    "market": "PSX",
                }
            )
        )
    facts = pd.concat(frames, ignore_index=True)
    return facts[facts["value"].notna()].reset_index(drop=True)
