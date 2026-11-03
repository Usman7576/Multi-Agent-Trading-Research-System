# PSX results-PDF extraction: measured accuracy

Measured 2026-11-04 with rules frozen at commit `f79f815`, rules only (no LLM).
**Status: when it answers, it is usually right, but it answers on only about half of reports. Extracted figures are
not used in any backtest yet.**

## How it is scored

Each company page lists its real share count, so profit after tax / EPS must come close to it (within 25%). A wrong
profit, EPS or unit shows up as a bad ratio. Correct reports usually land at 1.00. This test does not cover revenue or
period length, and it uses today's share count, so it is weaker for older reports (bonus issues change the count).

Reproduce with `scripts/eval_report_extraction.py`.

## Results

| Set | Reports | Correct | Flagged wrong | No result |
|---|---|---|---|---|
| Old rules, tuning set | 60 | 14 (23%) | 22 | 24 |
| New rules, tuning set (not an honest score: rules were fixed on these) | 60 | 54 (90%) | 4 | 2 |
| **Held-out A: random 2026 reports from 31 companies never looked at** | **40** | **22 (55%)** | **0** | **18** |
| **Held-out B: random 2019-2023 reports from those unseen companies** | **20** | **9 (45%)** | **3** | **8** |

Use the held-out rows, not the 90%. The gap between 90% and 55% shows how much the rules were fitted to the tuning set.

**Visual check:** 8 random reports scored correct in held-out A were compared with their page images, and all 8 match
on profit, EPS, unit and period length. This was checked by Claude, not by a person.

## What this means

- **Mostly silent failures:** in held-out A nothing was flagged wrong, so 22 of 22 answers passed. With a sample that
  small, the true error rate could still be up to about 13%.
- **Coverage is the main problem:** most no-result reports had no EPS found. The causes in the held-out sets have
  not been examined, so that the sets stay usable for one more honest comparison.
- **Older reports are harder:** of the 3 flagged in set B, BOP is a wrong unit (implied shares 800x too many). HINOON
  and PTC may be wrong or may reflect changed share counts; not checked.

## Changes that produced this

Board-meeting notices excluded; a failed download skips one report; note numbers no longer read as EPS; digits split by
PDF text rejoined; figures anchored on the EPS line, so they come from one statement; units from the nearest header;
period length from the first column; OCR for scanned pages inside text PDFs; OCR rows grouped by position.

## Known remaining issues

- ENGROH reports EPS by continuing and discontinued operations, and the continuing line is read as total EPS (flagged wrong).
- A single-digit note before a figure can still be merged by mistake when the next column is similar in size (SYS, tuning set).
- The local Qwen2.5-3B fallback was not re-tested on the new rules. Earlier it rescued 1 of 20 reports, so it is off by default.
