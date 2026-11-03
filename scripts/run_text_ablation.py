"""Does news text add anything to the PSX quant model? Same strategy with and without text features.

    uv run python scripts/run_text_ablation.py

Results are split at the local model's training cutoff: before it, even a masked headline may be scored with
some memory of what followed, so only the later period is a clean test of the LLM.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid

import pandas as pd

from tra.backtest.engine import run_backtest
from tra.config import REPO_ROOT, Settings, load_experiment
from tra.data.lake import ParquetLake
from tra.data.pit_store import PITStore
from tra.eval.signal import daily_ic, ic_summary
from tra.eval.significance import bootstrap_sharpe_diff_pvalue, deflated_sharpe_ratio, sharpe
from tra.features.pipeline import build_research_data
from tra.state import StateDB
from tra.strategies.alphas import make_alpha

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_baselines import summarise  # noqa: E402

log = logging.getLogger("run_text_ablation")
EXPERIMENTS = {"psx_b4_lightgbm": None, "psx_b5_text_lm": "lm", "psx_b5_text_llm": "llm"}
LABELS = {
    "psx_b4_lightgbm": "Quant only",
    "psx_b5_text_lm": "Quant + word list",
    "psx_b5_text_llm": "Quant + LLM",
}
LLM_CUTOFF = pd.Timestamp("2024-01-01")  # Qwen2.5 knowledge ends around late 2023
REPORT = REPO_ROOT / "reports" / "psx_text_ablation.md"


def coverage(data, method: str | None) -> float:
    """Share of stock-days in the universe that have a sentiment value."""
    if method is None:
        return float("nan")
    return float(data.ranked[f"news_{method}_sentiment"].notna().mean())


def period(res, scores, labels) -> dict[str, dict[str, float]]:
    ic = daily_ic(scores, labels["fwd_excess"])
    out = {}
    for name, mask_r, mask_ic in [
        ("before", res.returns.index < LLM_CUTOFF, ic.index < LLM_CUTOFF),
        ("after", res.returns.index >= LLM_CUTOFF, ic.index >= LLM_CUTOFF),
    ]:
        out[name] = {
            "sharpe": sharpe(res.returns[mask_r].to_numpy()),
            "ic": ic_summary(ic[mask_ic], horizon=1)["mean_ic"],
        }
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    pit = PITStore(ParquetLake(settings.data_dir))
    state = StateDB(settings.data_dir / "state.sqlite")
    as_of = pd.Timestamp.now(tz="UTC")
    results, rows = {}, {}
    for name, method in EXPERIMENTS.items():
        cfg = load_experiment(name)
        data = build_research_data(pit, cfg.market, as_of, horizon=cfg.model.horizon, text_method=method)
        alpha = make_alpha(cfg.model, cfg.seed)
        started = pd.Timestamp.now(tz="UTC").isoformat()
        res = run_backtest(cfg, data, alpha)
        summary = summarise(res)
        state.record_run(
            uuid.uuid4().hex[:12],
            cfg.name,
            cfg.config_hash(),
            started,
            pd.Timestamp.now(tz="UTC").isoformat(),
            "ok",
            summary,
        )
        scores = alpha.prediction_panel()
        ic = ic_summary(daily_ic(scores, data.labels["fwd_excess"]), horizon=1)
        results[name] = res
        rows[name] = {
            **summary,
            "ic_mean": ic["mean_ic"],
            "ic_t": ic["ic_t_stat"],
            "coverage": coverage(data, method),
            "periods": period(res, scores, data.labels),
            "config_hash": cfg.config_hash(),
        }
        log.info("%s: Sharpe %.2f IC %.3f", name, summary["sharpe"], ic["mean_ic"])
    trials = state.trial_sharpes(exclude=("psx_b0_benchmark",), prefix="psx_")
    base = results["psx_b4_lightgbm"].returns
    lines = [
        "# Does text help? PSX quant vs quant + news",
        "",
        "Same LightGBM model, optimiser, risk engine and costs; only the added news features differ.",
        "",
        "| | Sharpe | CAGR | IC | IC t | Sharpe before 2024 | Sharpe from 2024 | IC from 2024 | Coverage |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in rows.items():
        p = r["periods"]
        cov = "n/a" if pd.isna(r["coverage"]) else f"{r['coverage']:.1%}"
        lines.append(
            f"| {LABELS[name]} | {r['sharpe']:.2f} | {r['cagr']:.1%} | {r['ic_mean']:.3f} | {r['ic_t']:.1f} "
            f"| {p['before']['sharpe']:.2f} | {p['after']['sharpe']:.2f} | {p['after']['ic']:.3f} | {cov} |"
        )
    lines += ["", f"{len(trials)} PSX trials counted so far.", ""]
    for name in ["psx_b5_text_lm", "psx_b5_text_llm"]:
        p_value = bootstrap_sharpe_diff_pvalue(results[name].returns, base)
        rows[name]["p_vs_quant"] = p_value
        rows[name]["deflated_sharpe"] = deflated_sharpe_ratio(results[name].returns, trials)
        lines.append(
            f"- {LABELS[name]}: p-value for beating quant only {p_value:.3f}, "
            f"deflated Sharpe {rows[name]['deflated_sharpe']:.3f}"
        )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    REPORT.with_suffix(".json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    log.info("Wrote %s", REPORT)


if __name__ == "__main__":
    main()
