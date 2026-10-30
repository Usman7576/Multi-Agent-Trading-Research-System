# Results: PSX quant baselines

Test period **2019-01-02 to 2026-10-30** (walk-forward, out of sample), the 60 most liquid PSX stocks each day (point-in-time, delisted names included), daily bars, orders filled at the next open. Benchmark: the KSE-100 index. Costs: 17 bps commission and fees + 20 bps slippage per side + square-root impact (k=0.2). Sharpe ratios use a zero risk-free rate. 95% Sharpe CIs from a stationary block bootstrap (mean block 20 days).

| Strategy | CAGR | Vol | Sharpe [95% CI] | Max DD | Calmar | Beta | Avg gross | Turnover/yr | Cost drag/yr | PSR |
|---|---|---|---|---|---|---|---|---|---|---|
| 0. Buy & hold benchmark | 0.0% | 0.0% | nan [nan, nan] | 0.0% | nan | 0.00 | 0.0% | 0.0x | 0.0% | nan |
| 1. Equal-weight universe (monthly) | 19.3% | 26.1% | 0.81 [-0.03, 1.70] | 44.6% | 0.43 | 1.21 | 98.6% | 1.0x | 0.8% | 0.99 |
| 2. 12-1 momentum, top 10 (monthly) | 21.2% | 29.2% | 0.80 [0.03, 1.62] | 38.2% | 0.55 | 1.12 | 99.6% | 3.3x | 2.3% | 0.99 |
| 3. Ranked-factor composite + optimiser + risk | 6.5% | 10.8% | 0.64 [-0.23, 1.57] | 24.3% | 0.27 | 0.48 | 53.9% | 11.3x | 8.6% | 0.96 |

## Statistical tests

- Deflated Sharpe ratio of 3. Ranked-factor composite + optimiser + risk (0 trials): **0.961**
- p-value, Sharpe(3. Ranked-factor composite + optimiser + risk) > Sharpe(0. Buy & hold benchmark): **0.000**
- p-value, Sharpe(3. Ranked-factor composite + optimiser + risk) > Sharpe(1. Equal-weight universe (monthly)): **0.774**

## Signal quality (rebalance dates, 10-day forward excess return)

| Signal | Mean IC | IC t-stat | IC > 0 | Q1 -> Q5 forward excess (bps) |
|---|---|---|---|---|
| Factor composite | 0.064 | 5.32 | 60.0% | +2 / -5 / +0 / -11 / +14 |

## Cost stress test (2x costs)

| Strategy | Sharpe at 1x | Sharpe at 2x | CAGR at 2x |
|---|---|---|---|
| 3. Ranked-factor composite + optimiser + risk | 0.64 | -0.14 | -2.1% |

## Sharpe by calendar year

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| 0. Buy & hold benchmark |  |  |  |  |  |  |  |  |
| 1. Equal-weight universe (monthly) | 0.10 | 1.26 | -0.31 | -0.84 | 2.52 | 2.76 | 1.62 | -0.21 |
| 2. 12-1 momentum, top 10 (monthly) | -0.03 | 1.29 | 0.17 | -0.37 | 2.21 | 3.00 | 1.31 | -0.99 |
| 3. Ranked-factor composite + optimiser + risk | -0.58 | 0.14 | -0.47 | -0.70 | 2.93 | 3.02 | 1.25 | -0.68 |

## Reading these results honestly

- **Survivorship bias:** the universe is rebuilt every day from data known that day and includes stocks that were later delisted, so there is no survivorship bias from today's index membership. Sectors come from today's symbol list, so delisted stocks are tagged 'Unknown'.
- **Trials:** the deflated Sharpe ratio counts all 0 distinct non-benchmark configurations ever run (from the run registry in `data/state.sqlite`), including runs replaced after bug fixes. See `reports/experiment_log.md`.
- **Fundamentals:** SEC EDGAR fundamentals were not available for this run (EDGAR needs a contact User-Agent in `.env`), so value and quality factors are absent and the composite uses price factors only.

![Equity curves](psx_equity.png)
