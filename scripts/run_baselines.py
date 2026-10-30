"""Run the Milestone 1 baselines on US data and write the results report.

    uv run python scripts/run_baselines.py            # all strategies
    uv run python scripts/run_baselines.py --quick    # skip LightGBM (fast check)

Outputs go to ``reports/``: a Markdown results table, JSON metrics, NAV series and an equity chart.
Every run is recorded in ``data/state.sqlite`` with its config hash.
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tra.backtest.engine import BacktestResult, run_backtest  # noqa: E402
from tra.config import REPO_ROOT, Settings, load_experiment  # noqa: E402
from tra.data.lake import ParquetLake  # noqa: E402
from tra.data.pit_store import PITStore  # noqa: E402
from tra.eval.metrics import performance_summary  # noqa: E402
from tra.eval.signal import daily_ic, ic_summary, quantile_returns  # noqa: E402
from tra.eval.significance import (  # noqa: E402
    bootstrap_sharpe_ci,
    bootstrap_sharpe_diff_pvalue,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from tra.features.pipeline import build_research_data  # noqa: E402
from tra.state import StateDB  # noqa: E402
from tra.strategies.alphas import FactorCompositeAlpha, LightGBMAlpha, make_alpha  # noqa: E402

log = logging.getLogger("run_baselines")
EXPERIMENTS = [
    "b0_benchmark",
    "b1_equal_weight",
    "b2_momentum",
    "b3_factor_composite",
    "b4_lightgbm",
    "b4_lightgbm_no_risk",
]
LABELS = {
    "b0_benchmark": "0. Buy & hold benchmark",
    "b1_equal_weight": "1. Equal-weight universe (monthly)",
    "b2_momentum": "2. 12-1 momentum, top 10 (monthly)",
    "b3_factor_composite": "3. Ranked-factor composite + optimiser + risk",
    "b4_lightgbm": "4. LightGBM + optimiser + risk",
    "b4_lightgbm_no_risk": "4b. LightGBM, risk engine off",
}
REPORTS = REPO_ROOT / "reports"
MARKETS = {
    "us": {
        "prefix": "",
        "title": "US quant baselines",
        "benchmark": "SPY",
        "universe": "40 US large caps",
        "survivorship": "the universe is today's large caps, so every strategy (including equal weight) "
        "benefits from knowing these companies survived. Absolute returns are flattered; comparisons "
        "between strategies on the same universe are fairer.",
    },
    "psx": {
        "prefix": "psx_",
        "title": "PSX quant baselines",
        "benchmark": "the KSE-100 index",
        "universe": "the 60 most liquid PSX stocks each day (point-in-time, delisted names included)",
        "survivorship": "the universe is rebuilt every day from data known that day and includes stocks "
        "that were later delisted, so there is no survivorship bias from today's index membership. "
        "Sectors come from today's symbol list, so delisted stocks are tagged 'Unknown'.",
    },
}
MK = MARKETS["us"]


def summarise(res: BacktestResult) -> dict[str, float]:
    costs = sum(f.cost for f in res.fills)
    m = performance_summary(res.nav, res.benchmark_nav, res.turnover, costs)
    lo, hi = bootstrap_sharpe_ci(res.returns)
    m.update(
        {
            "sharpe_ci_low": lo,
            "sharpe_ci_high": hi,
            "psr": probabilistic_sharpe_ratio(res.returns),
            "total_costs": costs,
            "n_trades": len(res.fills),
            "n_vetoes": len(res.risk.vetoes),
            "kill_switch_events": len(res.risk.events),
            "avg_gross": float(res.weights.sum(axis=1).mean()),
        }
    )
    return m


def signal_report(scores: pd.Series, labels: pd.DataFrame, horizon: int) -> dict[str, object]:
    realised = labels["fwd_excess"]
    ic = daily_ic(scores, realised)
    out: dict[str, object] = ic_summary(ic, horizon=1)  # scores are weekly, labels 10 days: little overlap
    out["quintile_fwd_excess_bps"] = (quantile_returns(scores, realised) * 1e4).round(1).to_dict()
    return out


def composite_scores(data, days) -> pd.Series:
    alpha = FactorCompositeAlpha()
    return pd.concat({d: alpha.scores(d, data) for d in days}, names=["date", "ticker"])


def markdown_table(df: pd.DataFrame) -> list[str]:
    cols = [str(c) for c in df.columns]
    out = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, row in df.iterrows():
        out.append(f"| {idx} | " + " | ".join("" if pd.isna(v) else f"{v:.2f}" for v in row) + " |")
    return out


def fmt_pct(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.1%}"


def write_report(
    rows: dict[str, dict],
    signals: dict[str, dict],
    tests: dict[str, float],
    stress: dict,
    subperiods: pd.DataFrame,
    cfg,
    data_end: pd.Timestamp,
    lgbm: LightGBMAlpha | None,
    has_fundamentals: bool,
    n_trials: int,
) -> None:
    lines = [
        f"# Results: {MK['title']}",
        "",
        f"Test period **{cfg.dates.test_start} to {data_end.date()}** (walk-forward, out of sample), "
        f"{MK['universe']}, daily bars, orders filled at the next open. Benchmark: {MK['benchmark']}. Costs: "
        f"{cfg.costs.commission_bps:g} bps commission and fees + {cfg.costs.slippage_bps:g} bps slippage per "
        f"side + square-root impact (k={cfg.costs.impact_k}). "
        "Sharpe ratios use a zero risk-free rate. 95% Sharpe CIs from a stationary block bootstrap "
        "(mean block 20 days).",
        "",
        "| Strategy | CAGR | Vol | Sharpe [95% CI] | Max DD | Calmar | Beta | Avg gross | Turnover/yr "
        "| Cost drag/yr | PSR |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in rows.items():
        lines.append(
            f"| {LABELS[name]} | {fmt_pct(m['cagr'])} | {fmt_pct(m['ann_vol'])} | {m['sharpe']:.2f} "
            f"[{m['sharpe_ci_low']:.2f}, {m['sharpe_ci_high']:.2f}] | {fmt_pct(m['max_drawdown'])} | "
            f"{m['calmar']:.2f} | {m['beta']:.2f} | {fmt_pct(m['avg_gross'])} | "
            f"{m.get('ann_turnover', float('nan')):.1f}x | "
            f"{fmt_pct(m['cost_drag_ann'])} | {m['psr']:.2f} |"
        )
    lines += ["", "## Statistical tests", ""]
    for k, v in tests.items():
        lines.append(f"- {k}: **{v:.3f}**")
    lines += [
        "",
        "## Signal quality (rebalance dates, 10-day forward excess return)",
        "",
        "| Signal | Mean IC | IC t-stat | IC > 0 | Q1 -> Q5 forward excess (bps) |",
        "|---|---|---|---|---|",
    ]
    for name, s in signals.items():
        q = s["quintile_fwd_excess_bps"]
        lines.append(
            f"| {name} | {s['mean_ic']:.3f} | {s['ic_t_stat']:.2f} | {fmt_pct(s['hit_rate'])} | "
            + " / ".join(f"{q[k]:+.0f}" for k in sorted(q))
            + " |"
        )
    lines += [
        "",
        "## Cost stress test (2x costs)",
        "",
        "| Strategy | Sharpe at 1x | Sharpe at 2x | CAGR at 2x |",
        "|---|---|---|---|",
    ]
    for name, m in stress.items():
        lines.append(
            f"| {LABELS[name]} | {rows[name]['sharpe']:.2f} | {m['sharpe']:.2f} | {fmt_pct(m['cagr'])} |"
        )
    lines += ["", "## Sharpe by calendar year", "", *markdown_table(subperiods), ""]
    if lgbm is not None and lgbm.train_log:
        last = lgbm.train_log[-1]
        lines += [
            "## LightGBM walk-forward",
            "",
            f"- Retrained {len(lgbm.train_log)} times (every {cfg.model.retrain_every} sessions) on labels "
            "already revealed at each retrain date; hyperparameters chosen by purged, embargoed CV.",
            f"- Last retrain {last['date'].date()}: {last['rows']:,} rows, params {last['params']}, "
            f"CV IC per grid point {last['cv_ic']}.",
            f"- Top features at last retrain (gain): {', '.join(last['top_features'])}.",
            "",
        ]
    lines += [
        "## Reading these results honestly",
        "",
        f"- **Survivorship bias:** {MK['survivorship']}",
        f"- **Trials:** the deflated Sharpe ratio counts all {n_trials} distinct non-benchmark "
        "configurations ever run (from the run registry in `data/state.sqlite`), including runs replaced "
        "after bug fixes. See `reports/experiment_log.md`.",
        "- **Fundamentals:** "
        + (
            "SEC EDGAR fundamentals are included."
            if has_fundamentals
            else "SEC EDGAR fundamentals were not available for this run (EDGAR needs a contact User-Agent "
            "in `.env`), so value and quality factors are absent and the composite uses price factors only."
        ),
        "",
        f"![Equity curves]({MK['out']}_equity.png)",
        "",
    ]
    (REPORTS / f"{MK['out']}_baselines.md").write_text("\n".join(lines), encoding="utf-8")


def plot(results: dict[str, BacktestResult]) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]}
    )
    for name, res in results.items():
        nav = res.nav / res.nav.iloc[0]
        ax1.plot(nav.index, nav, label=LABELS[name], lw=1.6 if name.startswith("b4_lightgbm") else 1.1)
        ax2.plot(nav.index, nav / nav.cummax() - 1, lw=0.9)
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of 1 (log)")
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(alpha=0.25)
    ax2.set_ylabel("Drawdown")
    ax2.grid(alpha=0.25)
    fig.suptitle(f"{MK['title']}, out of sample, after costs")
    fig.tight_layout()
    fig.savefig(REPORTS / f"{MK['out']}_equity.png", dpi=130)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip the LightGBM strategies")
    parser.add_argument("--market", choices=sorted(MARKETS), default="us")
    args = parser.parse_args()
    global MK
    MK = {**MARKETS[args.market], "out": args.market}
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    REPORTS.mkdir(exist_ok=True)
    settings = Settings()
    pit = PITStore(ParquetLake(settings.data_dir))
    state = StateDB(settings.data_dir / "state.sqlite")

    names = [n for n in EXPERIMENTS if not (args.quick and "lightgbm" in n)]
    cfgs = {n: load_experiment(MK["prefix"] + n) for n in names}
    base = cfgs[names[0]]
    as_of = pd.Timestamp.now(tz="UTC")
    data = build_research_data(pit, base.market, as_of, horizon=cfgs["b3_factor_composite"].model.horizon)
    log.info("Research data: %d rows, %d features", len(data.ranked), len(data.signal_columns))

    results: dict[str, BacktestResult] = {}
    rows: dict[str, dict] = {}
    lgbm: LightGBMAlpha | None = None
    for name, cfg in cfgs.items():
        started = pd.Timestamp.now(tz="UTC").isoformat()
        alpha = make_alpha(cfg.model, cfg.seed)
        res = run_backtest(cfg, data, alpha)
        results[name], rows[name] = res, summarise(res)
        if name == "b4_lightgbm":
            lgbm = alpha  # type: ignore[assignment]
        state.record_run(
            uuid.uuid4().hex[:12],
            cfg.name,  # full experiment name, e.g. psx_b4_lightgbm (trials are counted per market)
            cfg.config_hash(),
            started,
            pd.Timestamp.now(tz="UTC").isoformat(),
            "ok",
            rows[name],
        )
        log.info("%s: Sharpe %.2f CAGR %.1f%%", name, rows[name]["sharpe"], 100 * rows[name]["cagr"])

    trial_sharpes = state.trial_sharpes(exclude=(MK["prefix"] + "b0_benchmark",), prefix=MK["prefix"])
    for n, m in rows.items():
        m["deflated_sharpe"] = deflated_sharpe_ratio(results[n].returns, trial_sharpes)

    tests: dict[str, float] = {}
    best = "b4_lightgbm" if "b4_lightgbm" in rows else "b3_factor_composite"
    tests[f"Deflated Sharpe ratio of {LABELS[best]} ({len(trial_sharpes)} trials)"] = rows[best][
        "deflated_sharpe"
    ]
    pairs = [
        ("b3_factor_composite", "b0_benchmark"),
        ("b3_factor_composite", "b1_equal_weight"),
        ("b4_lightgbm", "b3_factor_composite"),
        ("b4_lightgbm", "b4_lightgbm_no_risk"),
    ]
    for a, b in pairs:
        if a in results and b in results:
            p = bootstrap_sharpe_diff_pvalue(results[a].returns, results[b].returns)
            tests[f"p-value, Sharpe({LABELS[a]}) > Sharpe({LABELS[b]})"] = p

    rebalance_days = [d.date for d in results["b3_factor_composite"].decisions]
    signals: dict[str, dict] = {
        "Factor composite": signal_report(composite_scores(data, rebalance_days), data.labels, 10)
    }
    if lgbm is not None:
        signals["LightGBM"] = signal_report(lgbm.prediction_panel(), data.labels, 10)

    stress = {}
    for name in ["b3_factor_composite", "b4_lightgbm"]:
        if name in cfgs:
            res = run_backtest(
                cfgs[name], data, make_alpha(cfgs[name].model, cfgs[name].seed), cost_multiplier=2.0
            )
            stress[name] = performance_summary(res.nav, res.benchmark_nav, res.turnover)

    yearly = pd.DataFrame(
        {
            LABELS[n]: r.returns.groupby(r.returns.index.year).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252)
            )
            for n, r in results.items()
        }
    ).T
    data_end = max(r.nav.index.max() for r in results.values())
    has_fund = "earnings_yield" in data.signal_columns
    write_report(rows, signals, tests, stress, yearly, base, data_end, lgbm, has_fund, len(trial_sharpes))
    plot(results)
    pd.DataFrame({n: r.nav for n, r in results.items()}).to_csv(
        REPORTS / f"{MK['out']}_nav.csv", float_format="%.2f"
    )
    payload = {
        "metrics": rows,
        "tests": tests,
        "signals": signals,
        "config_hashes": {n: c.config_hash() for n, c in cfgs.items()},
        "data_end": str(data_end.date()),
    }
    (REPORTS / f"{MK['out']}_baselines.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    if lgbm is not None:
        risk = results["b4_lightgbm"].risk
        vetoes = pd.DataFrame([{"date": d, **v.__dict__} for d, v in risk.vetoes])
        if not vetoes.empty:
            vetoes.groupby("rule").size().rename("count").to_csv(
                REPORTS / f"{MK['out']}_lightgbm_veto_counts.csv"
            )
    log.info("Wrote %s", REPORTS / f"{MK['out']}_baselines.md")


if __name__ == "__main__":
    main()
