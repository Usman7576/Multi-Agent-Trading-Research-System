# PSX results-PDF extraction: sample stats

Measured 2026-11-03. **Status: not reliable yet. These figures are not used in any backtest.**

Sample: results announcements dated 2026 for 20 large KSE-100 companies. The run selected 70 announcements and
stopped after 50 when a PDF download failed on a network error, so 6 companies (OGDC, PPL, PSO, SEARL, SYS, UBL)
are not covered.

| Measure | Count |
|---|---|
| Announcements processed | 50 |
| Board-meeting notices wrongly selected as results (contain no figures; all failed) | 10 |
| Real results reports | 40 (33 OCR, 7 text PDFs) |
| Passed the checks ("usable") | 21 of 40 (52.5%): 20 by rules, 1 by LLM fallback |
| Failed (incomplete or inconsistent, never used) | 19 of 40 |
| Usable but with a whole-number EPS, very likely a statement note number (e.g. MCB 36, BAHL 35, MEBL 34) | 8 of 21 |
| Usable but implying fewer than 10M shares, a likely missed unit (ENGROH, ILP) | 2 of 21 |
| Usable and not visibly wrong | 11 of 40 (27.5%) |
| Local LLM runs on real reports / failures it rescued | 20 / 1 |

**The 11 are not verified as correct.** No report was hand-checked except one LUCK report, and the only test is
that nothing looks implausible. Treat 27.5% as an upper bound on accuracy for this sample.

## What this shows

- The accounting checks are too weak: most usable filings could not be tested at all (13 of 21 lacked the figures
  for the PBT - tax = PAT check), so "passed" often only means "profit and EPS were found".
- The local Qwen2.5-3B model is not a useful fallback here, rescuing 1 of 20 reports at about 40 s each.

## Known bugs to fix before re-running

1. Board-meeting notices match the results-title filter.
2. EPS extraction takes the first number on the line, which can be the statement's note column.
3. Units are missed when the unit line is not recognised, so figures are scaled wrongly.
4. A failed PDF download stops the whole run instead of skipping that report.
