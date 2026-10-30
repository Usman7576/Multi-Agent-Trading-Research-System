# Experiment log

Every configuration that was run is listed here, including runs that were superseded. The deflated Sharpe
ratio in the results report counts all of them as trials (they are also stored in the run registry,
`data/state.sqlite`, table `runs`).

## 2026-10-30: Milestone 1, US baselines

Risk limits, costs, rebalance frequency and the LightGBM grid were written in `configs/base.yaml` before
any backtest was run.

| # | Change | Reason | Effect on results |
|---|---|---|---|
| 1 | First run of strategies 0-3 | Initial run | Strategy 3 held ~10% gross on average: CAGR 0.3% |
| 2 | Mean-variance budget fixed to fully invested; optimiser candidates raised from 10 to 20 | **Bug fix.** The optimiser had a free budget (`sum(w) <= 1`). With annualised covariance and Grinold-scaled alphas, the risk penalty was ~15x the expected alpha, so the optimiser parked most of the portfolio in cash. With 10 candidates and a 10% cap a fully invested optimiser has no freedom, so it now chooses among the top 20. Gross exposure is decided by vol targeting and the risk engine, as designed. | Strategy 3 CAGR 6.7%, vol 8.6%, Sharpe 0.80 |
| 3 | Full run including LightGBM started, then stopped before finishing | **Bug found in the log:** LightGBM retrained every 63 *rebalances* (about 15 months) instead of every 63 trading sessions, because the counter only advanced when the model was called. Fixed to count sessions; covered by `test_lightgbm_retrains_every_n_sessions_not_every_n_calls`. No results were recorded from the stopped run. | none |
| 4 | Full run including LightGBM (strategy 4) and the no-risk ablation (4b) | Planned | Strategy 4 Sharpe 0.60, 4b 0.52; see `us_baselines.md` |

The pre-fix run of strategy 3 stays in the trial count.

## 2026-10-31: Milestone 2, PSX baselines

Same strategy settings as the US run; PSX market file adds the point-in-time liquidity universe (top 60 by
60-day traded value), circuit-breaker locks, a 10% volume-participation cap and PSX cost assumptions
(17 bps commission and fees, 20 bps slippage, impact k = 0.2). Trials are counted separately per market.

| # | Change | Reason | Effect on results |
|---|---|---|---|
| 1 | First quick run (strategies 0-3) | Initial run | Invalid: NAV undefined (Sharpe nan) |
| 2 | Mark-to-market at the last traded price | **Bug fix.** Thin PSX stocks skip sessions; a held stock without a bar made NAV NaN. Orders still fill only on days with a real open. Test added. | Strategies 1-3 valid |
| 3 | Volume-participation cap applies to stocks only | **Bug fix.** The cap gave the KSE-100 index (no traded value) a zero order size, so the benchmark held cash. | Benchmark Sharpe 1.06 |
| 4 | Run registry records full experiment names | **Bug fix.** PSX runs had been filed under US names; 8 registry rows corrected by config hash. Undefined Sharpes excluded from trial counts. US results unaffected (produced earlier). | Trial counts per market correct |
| 5 | Full run including LightGBM crashed (out of memory) | OCR/LLM sample ran at the same time on a 16 GB machine | No results recorded |
| 6 | Full run, nothing else running | Planned | Strategy 4 Sharpe 0.90, DSR 0.985 (5 trials); see `psx_baselines.md` |
