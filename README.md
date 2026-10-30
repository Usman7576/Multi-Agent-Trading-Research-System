# Multi-Agent Trading Research System

[![CI](https://github.com/Usman7576/Multi-Agent-Trading-Research-System/actions/workflows/ci.yml/badge.svg)](https://github.com/Usman7576/Multi-Agent-Trading-Research-System/actions/workflows/ci.yml)

A research and paper-trading system for the **Pakistan Stock Exchange (PSX)** and **US equities**, built
the way real quant systems are: a point-in-time data platform, a quant signal engine, an event-driven
backtester with realistic costs, a portfolio optimiser and a risk engine with veto power. Later milestones add
time-aware RAG over filings and news, and LangGraph agents that debate each trade (Bull vs. Bear, judged by
a Portfolio Manager). The LLM layer is one input among many and has to earn its place in ablations.

> Research and paper trading only. Not connected to real money. Not financial advice.

## Status

| Milestone | Scope | Status |
|---|---|---|
| **M1 Quant core** | Point-in-time store, US data, KSE-100 snapshots, features, LightGBM, HMM regimes, backtester, optimiser, risk engine, baselines | **Done** |
| **M2 PSX** | Own PSX scraper (prices, index, announcements, payouts, PDF reports with OCR), LDCP total-return adjustment, point-in-time liquidity universe, circuit breakers, PSX costs, PSX baselines | **Done** (report extraction run on a sample so far) |
| M3 Text layer | News scraper, time-aware hybrid RAG with evaluation, LLM signal extraction | Planned |
| M4 Agents | LangGraph analysts, Bull/Bear debate, Portfolio Manager, citation checks | Planned |
| M5 Product | Daily jobs, paper trading, FastAPI, Streamlit dashboard with Decision Cards | Planned |

Total LLM budget for the whole project: **$0 expected, $2 hard ceiling** (local models and free tiers).

## Milestone 2 results (PSX, out of sample 2019 to 2026, after costs)

Universe: the 60 most liquid PSX stocks on each date, chosen from data known that day, delisted names
included. Costs: 17 bps commission and fees + 20 bps slippage per side + square-root impact; orders into
stocks locked at the daily circuit limit do not fill; at most 10% of a day's traded value per order.

| Strategy | CAGR | Sharpe [95% CI] | Max DD | Avg gross | Cost drag/yr |
|---|---|---|---|---|---|
| 0. Buy & hold KSE-100 | 21.2% | 1.06 [0.21, 1.97] | 37.0% | 100% | 0.0% |
| 1. Equal-weight universe (monthly) | 19.3% | 0.81 [-0.03, 1.70] | 44.6% | 99% | 0.8% |
| 2. 12-1 momentum, top 10 (monthly) | 21.2% | 0.80 [0.03, 1.62] | 38.2% | 100% | 2.3% |
| 3. Ranked-factor composite + optimiser + risk | 6.5% | 0.64 [-0.23, 1.57] | 24.3% | 54% | 8.6% |
| 4. LightGBM + optimiser + risk | 9.8% | 0.90 [0.00, 1.84] | 24.2% | 54% | 6.4% |
| 4b. LightGBM, risk engine off | 9.3% | 0.75 [-0.11, 1.65] | 28.4% | 65% | 7.4% |

**What this shows, honestly:**

- **PSX has real, measurable cross-sectional signal.** LightGBM's out-of-sample IC is 0.071 (t = 6.3) and
  the simple factor composite's is 0.064 (t = 5.3), versus 0.013 in the US. The deflated Sharpe of strategy 4
  over 5 trials is 0.985, above the usual 0.95 bar.
- **The risk engine matters here:** with it, Sharpe is 0.90 vs 0.75 without (p = 0.021) and drawdown falls
  from 28.4% to 24.2%.
- **Costs decide the outcome.** Weekly rebalancing at 8.6x turnover costs 6.4% a year; at 2x costs the
  Sharpe drops to 0.33. With about half the market exposure, no strategy beat the KSE-100 during its strong
  2023-25 rally. The obvious next step is lower turnover (slower rebalancing, a larger turnover penalty).
- **Price features only:** results-PDF fundamentals are extracted for a sample so far, not the full history.

Full report: [`reports/psx_baselines.md`](reports/psx_baselines.md).

![PSX equity curves](reports/psx_equity.png)

## Milestone 1 results (US, out of sample 2019 to 2026, after costs)

| Strategy | CAGR | Sharpe [95% CI] | Max DD | Avg gross |
|---|---|---|---|---|
| 0. Buy & hold SPY | 17.2% | 0.93 [0.26, 1.68] | 33.5% | 100% |
| 1. Equal-weight universe (monthly) | 17.4% | 1.00 [0.32, 1.82] | 33.1% | 98% |
| 2. 12-1 momentum, top 10 (monthly) | 22.9% | 1.05 [0.45, 1.66] | 29.7% | 100% |
| 3. Ranked-factor composite + optimiser + risk | 6.7% | 0.80 [0.21, 1.51] | 13.6% | 66% |
| 4. LightGBM + optimiser + risk | 5.1% | 0.60 [-0.04, 1.34] | 17.9% | 65% |
| 4b. LightGBM, risk engine off | 5.9% | 0.52 [-0.19, 1.36] | 28.8% | 85% |

**What this shows, honestly:**

- **The ML model does not beat simple baselines.** LightGBM's out-of-sample IC is 0.013 (t = 1.2), not
  significant; its deflated Sharpe over 6 trials is 0.79, below the usual 0.95 bar. On 40 mega caps, price
  features alone carry little cross-sectional signal. This is the baseline the LLM layers will have to beat.
- **The risk engine does its job:** it cuts LightGBM's maximum drawdown from 28.8% to 17.9% and lifts Sharpe
  from 0.52 to 0.60, at the cost of lower exposure. The Sharpe difference itself is not significant (p = 0.20).
- **Classic 12-1 momentum** has the best Sharpe here, but its confidence interval overlaps every other
  strategy's; no strategy beats buy-and-hold SPY with statistical significance over this period.
- **Fundamentals are not in this run** (SEC EDGAR access needs a contact address in `.env`), so value and
  quality factors are missing. That is the next thing to add.

Full report with confidence intervals, statistical tests, signal quality, cost stress test and yearly
breakdown: [`reports/us_baselines.md`](reports/us_baselines.md). Every configuration that was run, including
superseded ones, is in [`reports/experiment_log.md`](reports/experiment_log.md).

![Equity curves](reports/us_equity.png)

## Architecture

```
SOURCES   yfinance | SEC EDGAR | PSX Data Portal (own scraper) | news (M3)
   |
[L0] Scrapers ............. polite HTTP client, raw archive, header-based parsers, drift detection
[L1] Point-in-time store .. Parquet + DuckDB; every row has known_at; reads filtered by as_of
[L2] Quant engine ......... technical + fundamental features, ranks, LightGBM (purged CV), HMM regime
[L3] Time-aware RAG ....... (M3)
[L4] LLM extraction ....... (M3)
[L5] LangGraph agents ..... (M4)
[L6] Portfolio ............ equal / inverse-vol / Ledoit-Wolf mean-variance (cvxpy), vol targeting
[L7] Risk engine .......... position, sector, top-5, liquidity, regime, drawdown kill-switch, CVaR; logs vetoes
[L8] Backtester ........... next-open fills, commission + slippage + square-root impact
   |
Evaluation: Sharpe CIs (block bootstrap), probabilistic and deflated Sharpe, IC, ablations
```

## How look-ahead bias is prevented

Look-ahead bias makes most backtests on GitHub meaningless. Here it is handled in code and checked by tests:

| Rule | Where | Test |
|---|---|---|
| Every row stores `known_at`: a price bar at its session close (real exchange calendar, early closes included); an EDGAR fact at the end of its **filing** day, not its period end | `data/ingest_us.py`, `data/edgar.py` | `test_bar_is_invisible_until_its_session_closes`, `test_fundamentals_visible_only_after_filing_day` |
| All reads go through `PITStore`, which returns only rows with `known_at <= as_of` and refuses queries after the attached clock's "now" | `data/pit_store.py` | `test_pit_no_lookahead_*` (property tests), `test_clock_guard_blocks_future_queries` |
| Research code cannot touch raw tables | | `test_no_direct_table_access` |
| Restatements are visible only once filed | `PITStore.fundamentals` | `test_restatement_replaces_value_only_after_it_is_filed` |
| Features on day D do not change when later data exists | `features/` | `test_feature_shift` |
| Labels carry the date they are revealed; the walk-forward model trains only on revealed labels | `features/labels.py`, `strategies/alphas.py` | `test_labels_are_forward_and_dated` |
| Cross-validation purges overlapping labels and embargoes the next 10 days | `models/cv.py` | `test_purged_kfold_never_overlaps` |
| The HMM regime uses filtered, not smoothed, probabilities | `models/regime.py` | `test_regime_filter_is_causal` |
| Orders fill at the next open, never on the bar that created them | `backtest/broker_sim.py` | `test_fill_next_open_only` |

## PSX data: built from scratch

There is no clean, free, point-in-time dataset for the Pakistan Stock Exchange, so the project builds one
from the PSX Data Portal.

| Dataset | How | Size |
|---|---|---|
| Daily bars, every listed symbol | Date-wise `/historical` requests (one per session, delisted names included); empty days are holidays, giving the calendar | 1.8M bars, 2015-2026 |
| KSE-100 index levels | Symbol-month requests | 2015-2026 |
| KSE-100 constituents | Daily snapshot (history is otherwise lost) | from Oct 2026 |
| Company announcements | Paginated per company, exact timestamps and PDF links | ~47k for the KSE-100 names |
| Dividends, bonus and right issues | Company payout tables | last ~2 years |
| Financial results | Results PDFs: text or OCR (RapidOCR), rule extraction, accounting checks, local-LLM fallback | sample of 2026 reports |

- **Endpoints and token:** the portal's own JavaScript loads tables through AJAX calls carrying an `X-Req-Id`
  token embedded in each page; the client reads it and sends it like the browser does.
- **Robustness:** 1 request/second, retries with backoff (including the plain `error` body the server sends
  under load), columns mapped by header text, drift detection, raw responses archived (gzip), per-date
  watermarks so every job resumes where it stopped.
- **Corporate actions from LDCP:** on an ex-date PSX sets LDCP (the reference price) to the ex-adjusted
  previous close, so `close / LDCP - 1` is the total return. Adjusted prices are compounded *forward*, so they
  never change retroactively. Checked against announcements: **97.1% of 313 announced dividends and bonus
  issues match a detected ex-date** (bonus issues 100%), and 90.8% agree with the announced amount within 0.5%.
- **Cleaning:** futures, odd lots and rights (symbols with a dash) are excluded; no-trade days are flat bars.

```bash
uv run python -m tra.scrapers.psx.run --dataset symbols
uv run python -m tra.scrapers.psx.run --dataset index --since 2015-01-01
uv run python -m tra.scrapers.psx.run --dataset prices --since 2015-01-01
uv run python -m tra.scrapers.psx.run --dataset adjust
uv run python -m tra.scrapers.psx.run --dataset announcements --symbols kse100
uv run python -m tra.scrapers.psx.run --dataset reports --llm --since 2026-01-01 --symbols kse100
uv run python scripts/run_baselines.py --market psx
```

The daily KSE-100 snapshot can be scheduled on weekdays at 17:30 with `scripts/register_psx_snapshot_task.ps1`.
Raw PSX data is never committed; only small parser fixtures are.

## Local LLM (no Ollama, $0)

`llama-cpp-python` runs Qwen2.5-3B-Instruct (4-bit GGUF, 2.1 GB, kept in `data/models/`) on the CPU. Output
is constrained to a Pydantic model's JSON schema, validated, retried once and cached. On noisy OCR text the 3B
model was less accurate than the rule-based reader, so it runs only as a checked fallback and on a 1-in-20
audit sample. Install with `uv sync --all-extras`.

## Run it locally (Windows, no Docker)

```bash
uv sync --all-groups                                   # Python 3.12 environment from uv.lock
cp .env.example .env                                   # add SEC_USER_AGENT="Your Name you@email.com"
uv run python -m tra.data.ingest_us --prices           # ~130k daily bars, quality-checked
uv run python -m tra.data.ingest_us --fundamentals     # SEC EDGAR (needs SEC_USER_AGENT)
uv run python scripts/run_baselines.py                 # writes reports/
uv run pytest                                          # offline test suite on synthetic fixtures
```

Experiments are YAML files merged as `configs/base.yaml` -> `configs/markets/<market>.yaml` ->
`configs/experiments/<name>.yaml`, validated with Pydantic and hashed; the hash is stored with every run.

## Repository layout

```
configs/            base settings, market universes, one file per experiment
src/tra/
  data/             Parquet lake, PITStore, calendars, quality checks, US ingestion, EDGAR parser,
                    PSX adjustment (LDCP), results-PDF extraction, point-in-time liquidity universe
  scrapers/         shared HTTP client, number parser, drift detection; psx/ token-aware client, prices,
                    index, constituents, announcements, payouts, reports, CLI
  llm/              local llama.cpp client with schema-constrained JSON
  features/         technical, point-in-time fundamentals, cross-sectional ranks, labels, pipeline
  models/           purged k-fold, LightGBM return model, Gaussian HMM regime model
  strategies/       alpha models for every baseline and walk-forward LightGBM
  portfolio/        construction and mean-variance optimiser
  risk/             risk engine and VaR/CVaR
  backtest/         event-driven engine, simulated broker, cost model
  eval/             performance metrics, IC analysis, bootstrap and deflated Sharpe
scripts/            run_baselines.py, Task Scheduler registration
reports/            results committed with the code that produced them
tests/              offline tests on synthetic fixtures and saved HTML
```

## Known limitations

- **Survivorship bias (US):** the universe is today's 40 large caps across all 11 sectors, including some
  later-weak names. Absolute returns are flattered for every strategy; the comparisons between strategies are
  the meaningful part. The PSX universe has no such bias (rebuilt daily from liquidity, delisted included).
- **PSX costs** are assumptions (commission, fees, slippage) to check against current broker rates; capital
  gains tax is not modelled. The circuit-breaker band uses today's rule (7.5%) for all years.
- **PSX sectors** come from today's symbol list, so delisted stocks are tagged "Unknown".
- **Free data:** yfinance prices are retroactively split- and dividend-adjusted. Returns and ratios are
  unaffected; market capitalisation is rebuilt from split events so share counts stay consistent.
- **Fundamentals timing:** EDGAR gives the filing date without a time, so a filing is treated as known only
  after its filing day ends (one day later than it may really have been public). This is conservative.
- **Sector labels** are current classifications, not historical ones.

## Engineering

Python 3.12, uv, pandas, DuckDB, LightGBM, hmmlearn, cvxpy, scikit-learn, pytest + hypothesis, ruff,
GitHub Actions. The offline test suite (synthetic fixtures and saved PSX pages) runs in about two minutes.

## License

MIT
