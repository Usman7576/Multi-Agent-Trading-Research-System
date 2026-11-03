# PSX results-PDF extraction: measured accuracy

Rules frozen at commit `8170633` before scoring, rules only (no LLM).
**Status: it answers on about 65% of unseen reports, and about 1 in 5 of those answers fails the check. Extracted
figures are not used in any backtest.**

## How it is scored

Each company page lists its real share count, so profit after tax / EPS must come close to it (within 25%). A wrong
profit, EPS or unit shows up as a bad ratio, and correct reports usually land at 1.00. Limits of this test:

- It does not cover revenue or period length, so those rest on the visual checks below.
- It uses today's share count. Bonus issues and restructurings change that over time, so on reports before 2025 a
  mismatch is counted as unverified, not wrong.
- A few companies fail it even when correct if the listed count excludes a share class. PTC may be one; not checked.

Reproduce with `scripts/eval_report_extraction.py`. Company sets are fixed in `configs/psx_extraction_sets.yaml`.

## Results on reports the rules were not tuned on

Held-out companies were chosen at random before any of their reports were read.

| Set | Reports | Answered | Correct | Wrong | No answer |
|---|---|---|---|---|---|
| **C: 2025-26, held-out companies, previous rules** | 40 | 19 (47.5%) | 16 (40%) | 3 | 21 |
| **C: 2025-26, held-out companies, current rules** | **40** | **26 (65%)** | **21 (52.5%)** | **5** | **14** |
| **D: 2019-24, held-out companies, current rules** | **20** | **13 (65%)** | 8 match today's shares | 5 unverified | **7** |

So this round's fixes added 5 correct answers and 2 wrong ones on the same 40 reports: a real but modest gain.

**Do not quote the tuning scores as accuracy.** On reports used to write the rules, the current rules answer 86% to 97%
and score up to 90%. The gap to 52.5% is how far the rules are fitted to those reports. An earlier held-out set (40
2026 reports) scored 22 correct, 0 wrong and 18 no answer on the previous rules. Its 0 wrong was partly luck of the draw:
set C on those same rules had 3 wrong.

## Visual checks

Reports were compared with their page images by Claude, not by a person.

| Sample | Result |
|---|---|
| 8 random correct reports from the earlier set | All 8 match on profit, EPS, unit and period length |
| 3 random answered reports from C | 2 match. POWER is wrong: note "14.1" was read as EPS (true EPS 0.80). The share test flagged it |
| 4 random answered reports from D | All 4 match, including SRVI, which the share test could not verify |

## Known remaining issues

- Decimal note numbers are read as EPS when the label is on its own line (POWER: note 14.1).
- Units are still missed sometimes (BWCL: rupees instead of thousands), or wrongly read as millions (FFL).
- EPS reported by continuing and discontinued operations is read as the continuing line (ENGROH).
- 14 of 40 reports get no answer; the causes in the held-out set have not been examined, so it stays unseen.
- The local Qwen2.5-3B fallback was not re-tested on these rules. Earlier it rescued 1 of 20 reports, so it is off by default.

## What would be needed before using this in a backtest

Reject answers that fail the share-count test, hand-check 30 reports as the plan requires, and extract the full history.
