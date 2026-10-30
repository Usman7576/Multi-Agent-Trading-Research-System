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
| 4 | Full run including LightGBM (strategy 4) and the no-risk ablation (4b) | Planned | See `us_baselines.md` |

The pre-fix run of strategy 3 stays in the trial count.
