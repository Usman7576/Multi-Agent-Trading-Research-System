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
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------- text
MIN_PAGE_CHARS = 200  # a page with less text than this is treated as a scan
_EPS_ANYWHERE = re.compile(r"(?:earnings|loss|profit)(?:/\(?loss\)?)?per(?:ordinary)?share")


def _has_pnl(text: str) -> bool:
    """True once the text holds a complete profit and loss block (a profit line and an EPS line)."""
    low = re.sub(r"\s+", "", text.lower())
    has_profit = "aftertax" in low or re.search(r"(?:profit|loss|\))forthe(?:period|quarter|year)", low)
    return bool(has_profit and _EPS_ANYWHERE.search(low))


def pdf_text(pdf: bytes, max_pages: int = 15) -> tuple[str, str]:
    """Return (text, method) where method is 'text', 'ocr', 'mixed' or 'needs_ocr'.

    Each page is read on its own, because many reports are a typed letter with scanned statements attached.
    Reading stops at the first page that completes a profit and loss block, which keeps OCR of long annexed
    reports affordable.
    """
    import pdfplumber

    pages: list[str] = []
    used_text, used_ocr, ocr_missing = False, False, False
    with pdfplumber.open(io.BytesIO(pdf)) as doc:
        for i, page in enumerate(doc.pages[:max_pages]):
            text = page.extract_text() or ""
            if len(text.strip()) >= MIN_PAGE_CHARS:
                used_text = True
            else:
                try:
                    text = _ocr_page(pdf, i)
                    used_ocr = True
                except ImportError:
                    ocr_missing = True
            pages.append(text)
            if _has_pnl("\n".join(pages)):
                break
    method = (
        "needs_ocr" if ocr_missing else "mixed" if used_text and used_ocr else "ocr" if used_ocr else "text"
    )
    return "\n".join(pages), method


_ENGINE = None


def _ocr_page(pdf: bytes, index: int) -> str:
    global _ENGINE
    import pypdfium2 as pdfium
    from rapidocr_onnxruntime import RapidOCR

    if _ENGINE is None:
        _ENGINE = RapidOCR()
    page = pdfium.PdfDocument(pdf)[index]
    for scale in (2.0, 1.4):  # large scanned pages can exhaust memory at full resolution
        try:
            result, _ = _ENGINE(page.render(scale=scale).to_numpy())
            return "\n".join(group_ocr_lines(result or []))
        except Exception as exc:  # ONNX runtime raises its own error types (e.g. bad allocation)
            log.warning("OCR page %d at scale %.1f failed: %s", index + 1, scale, str(exc)[:120])
    return ""


def group_ocr_lines(boxes: list) -> list[str]:
    """Join OCR boxes into text lines. Boxes are on one line if their vertical centres are within about half
    a text height, so a table row stays together even when the scan is slightly tilted."""
    if not boxes:
        return []
    items = []
    for box, text, *_ in boxes:
        ys = [pt[1] for pt in box]
        items.append(((min(ys) + max(ys)) / 2, max(ys) - min(ys), min(pt[0] for pt in box), text))
    items.sort(key=lambda it: it[0])
    tolerance = 0.6 * sorted(it[1] for it in items)[len(items) // 2]
    lines, current, line_y = [], [], None
    for y, _h, x, text in items:
        if line_y is not None and abs(y - line_y) > tolerance:
            lines.append(current)
            current, line_y = [], None
        if line_y is None:
            line_y = y
        current.append((x, text))
    lines.append(current)
    return [" ".join(t for _x, t in sorted(line)) for line in lines]


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
        r"^(?:profit|\(?loss\)?|profit/\(loss\)|\(loss\)/profit)forthe(?:period|quarter|year)",
    ],
    "eps": [
        # 'earning', 'earnings', OCR's 'earming'; '(loss)/earnings'; 'earnings / (loss)'
        r"^(?:basic|diluted|and|&|-)*(?:\(?loss\)?/)?ear[nm]ings?(?:[/il|]\(?loss\)?)?per(?:ordinary)?share",
        r"^(?:basic|diluted|and|&|-)*\(?(?:loss|profit)\)?(?:/\(?(?:loss|profit)\)?)?per(?:ordinary)?share",
        r"^eps\b",
    ],
}
_NUMBER = re.compile(r"\(?-?[\d,]*\d(?:\.\d+)?\)?")
_UNITS = [
    (
        re.compile(
            r"(?:rupees?|rs\.?|\(r|pkr|amounts?)(?:are)?(?:in)?['\"`’‘.(]*(?:thousands?|0[0o]{1,2}|o[0o]{1,2})"
        ),
        "PKR_000",
    ),
    (re.compile(r"^\(?['\"`’‘]?000['\"`’‘]?\)?$"), "PKR_000"),
    (re.compile(r"(?:rupees?|rs\.?|pkr|amounts?)(?:are)?(?:in)?(?:millions?|mn\b)"), "PKR_mn"),
    (re.compile(r"(?:rupees?|rs\.?|pkr|amounts?)(?:are)?(?:in)?(?:billions?|bn\b)"), "PKR_bn"),
]
_MONTHS = [
    (re.compile(r"ninemonths?|9months?|thirdquarter"), 9),
    (re.compile(r"half-?year|sixmonths?|6months?|secondquarter"), 6),
    (re.compile(r"(?:year|annual)ended|twelvemonths?|12months?"), 12),
    (re.compile(r"qua?rterended|threemonths?|3months?|firstquarter"), 3),
]


def _squeeze(line: str) -> str:
    return re.sub(r"\s+", "", line.lower())


def _tokens(line: str, amounts: bool) -> list[str]:
    """Number tokens of a line, repairing how PDFs and OCR break them up: text PDFs split digits ('9 .82',
    '6 ,050,394,226', '( 120)') and OCR reads thousands separators in amounts as dots ('131.863.008')."""
    line = re.sub(r"(?<=\d)\s+(?=[,.]\d)", "", line)
    line = re.sub(r"\(\s+(?=\d)", "(", line)
    if amounts:
        line = re.sub(r"(?<=\d)\.(?=\d{3}(?:\D|$))", ",", line)
    return [t for t in _NUMBER.findall(line) if not re.fullmatch(r"(19|20)\d\d", t.strip("()-"))]


def _value(token: str) -> float:
    negative = (token.startswith("(") and token.endswith(")")) or token.startswith("-")
    number = float(token.strip("()-").replace(",", ""))
    return -number if negative else number


def _first_figure(line: str, is_eps: bool) -> float | None:
    """The first real figure on a line, skipping the statement's note-number column.

    Text PDFs also split a leading digit off a figure ('4 2,243,957 47,149,242' is 42,243,957). That looks
    just like note 4 followed by 2,243,957, so the reading closest in size to the next column is used.
    """
    tokens = _tokens(line, amounts=not is_eps)

    def real(tok: str) -> bool:
        return "." in tok if is_eps else abs(_value(tok)) >= 1000

    for k, tok in enumerate(tokens):
        if real(tok):
            value = _value(tok)
            if k > 0 and re.fullmatch(r"\d", tokens[k - 1]):
                later = next((j for j in range(k + 1, len(tokens)) if real(tokens[j])), None)
                merged = _value(tokens[k - 1] + tok.strip("()-"))
                if later is not None and value and _value(tokens[later]):
                    other = abs(_value(tokens[later]))
                    if re.fullmatch(r"\d", tokens[later - 1]) and later - 1 > k:
                        other = abs(_value(tokens[later - 1] + tokens[later].strip("()-")))
                    if abs(math.log(merged / other)) < abs(math.log(abs(value) / other)):
                        return -merged if value < 0 else merged
            return value
    return None


# Second line of a wrapped EPS label: 'and diluted (Rupees) 3.62', 'Basic 17.70', 'For continuing operations'
_EPS_CONTINUATION = re.compile(r"^(?:and|basic|diluted|(?:for)?continuing)")


def _figure_at(lines: list[str], i: int, is_eps: bool) -> float | None:
    """Figure on line ``i``, or on one of the next lines holding only numbers (statement labels wrap)."""
    value = _first_figure(lines[i], is_eps)
    if value is not None:
        return value
    for line in lines[i + 1 : i + 4]:
        wrapped = line.lstrip()[:1] in "(-" or (is_eps and _EPS_CONTINUATION.search(_squeeze(line)))
        if wrapped and (value := _first_figure(line, is_eps)) is not None:
            return value
        if re.search(r"[a-z]{4,}", line.lower()) and not re.fullmatch(
            r"[\W\d]*(?:rupees?|rs)[\W\d]*", line.lower()
        ):
            return None
        value = _first_figure(line, is_eps)
        if value is not None:
            return value
    return None


def _find(lines: list[str], field: str, is_eps: bool = False) -> list[tuple[int, float]]:
    patterns = LABELS[field]
    out = []
    for i, line in enumerate(lines):
        if any(re.search(p, _squeeze(line)) for p in patterns):
            value = _figure_at(lines, i, is_eps)
            if value is not None:
                out.append((i, value))
    return out


def _detect_unit(lines: list[str], anchor: int) -> str:
    """The unit header nearest above the figures, then anywhere in the document, else plain rupees.

    Prose such as 'Rs. 986 million' never matches, because a figure sits between the currency and the unit.
    """
    squeezed = [_squeeze(line) for line in lines]
    order = list(range(anchor, -1, -1)) + list(range(anchor + 1, len(lines)))
    for i in order:
        for pattern, name in _UNITS:
            if pattern.search(squeezed[i]):
                return name
    return "PKR"


def _detect_months(lines: list[str], anchor: int, title: str) -> int | None:
    """Period length of the first figure column.

    Column order varies (OGDC puts the quarter first, most others the cumulative period), so a header line
    naming two periods decides, using whichever comes first. Otherwise the nearest period label above the
    figures, then the title. Annual reports always follow the title.
    """
    title_months = next((m for pat, m in _MONTHS if pat.search(_squeeze(title))), None)
    if title_months == 12:
        return 12
    nearest = None
    for line in reversed(lines[max(0, anchor - 45) : anchor + 1]):
        low = _squeeze(line)
        hits = sorted((m.start(), months) for pat, months in _MONTHS for m in pat.finditer(low))
        if len({months for _pos, months in hits}) >= 2:
            return hits[0][1]
        if hits and nearest is None:
            nearest = hits[0][1]
    return nearest or title_months


def rule_extract(text: str, title: str = "") -> ResultsExtract:
    """Read one profit and loss block, anchored on its EPS line.

    Profit, before-tax profit and tax are the nearest matches above EPS, and revenue the first match in that
    block, so all figures come from the same statement (not the equity or cash-flow statement further on).
    """
    lines = text.splitlines()
    eps_hits = _find(lines, "eps", is_eps=True)
    found: dict[str, float] = {}
    if eps_hits:
        anchor, found["eps"] = eps_hits[0]
        start = max(0, anchor - 45)
    else:
        anchor, start = len(lines) - 1, 0
    for field, pick_last in (
        ("profit_after_tax", True),
        ("profit_before_tax", True),
        ("taxation", True),
        ("revenue", False),
        ("gross_profit", False),
    ):
        hits = [h for h in _find(lines, field) if start <= h[0] <= anchor]
        if hits:
            found[field] = hits[-1][1] if pick_last else hits[0][1]
    if "taxation" in found:
        found["taxation"] = abs(found["taxation"])
    return ResultsExtract(
        period_months=_detect_months(lines, anchor, title), unit=_detect_unit(lines, anchor), **found
    )


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
