"""Score results-PDF extraction against an independent fact: each company's real share count.

Profit after tax / EPS must be close to total shares, so a wrong profit, EPS or unit shows up as a bad ratio.
Uses archived PDFs and cached text only, so no network and no OCR are needed once reports were processed.

    uv run python scripts/eval_report_extraction.py --since 2026-01-01 --symbols LUCK,MCB
    uv run python scripts/eval_report_extraction.py --since 2025-01-01 --until 2025-12-31 --symbols kse100 -v

Held-out scoring: --sample picks reports at random before any is read, --exclude keeps out the companies the
rules were tuned on, and --process downloads and reads the chosen reports first.

    uv run --all-extras python scripts/eval_report_extraction.py --since 2026-01-01 --symbols kse100 \\
        --exclude LUCK,MCB --sample 40 --seed 1 --process
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from tra.config import CONFIG_DIR, Settings
from tra.data.lake import ParquetLake
from tra.data.psx_financials import UNIT_MULTIPLIER, accounting_checks, rule_extract
from tra.scrapers.common.http import HttpClient
from tra.scrapers.psx.client import USER_AGENT, PSXClient
from tra.scrapers.psx.fetch_reports import process_reports, result_announcements

RECENT_FROM = "2025-01-01"
TOLERANCE = 0.25  # shares change with bonus issues, and consolidated EPS excludes minority interests


def evaluate(
    lake: ParquetLake,
    raw_dir: Path,
    symbols: list[str],
    since: str,
    until: str | None,
    sample: int | None = None,
    seed: int = 1,
    process: bool = False,
    skip_ids: set[str] | None = None,
) -> pd.DataFrame:
    ann = result_announcements(lake, symbols, since)
    ann = ann[~ann["announcement_id"].astype(str).isin(skip_ids or set())]
    if until:
        ann = ann[ann["period_end"] <= pd.Timestamp(until)]
    if sample:
        ann = ann.sample(min(sample, len(ann)), random_state=seed).sort_values(["symbol", "known_at"])
    if process:
        http = HttpClient(USER_AGENT, requests_per_second=1.0, timeout_seconds=120)
        process_reports(PSXClient(http), lake, raw_dir, ann)
    prof = lake.read("company_profiles", "PSX").sort_values("snapshot_date").groupby("symbol").last()
    rows = []
    for a in ann.itertuples(index=False):
        cache = raw_dir / "reports" / a.symbol / f"{a.announcement_id}.text.json"
        if not cache.exists():
            continue
        saved = json.loads(cache.read_text(encoding="utf-8"))
        x = rule_extract(saved["text"], a.title)
        checks = accounting_checks(x)
        pat = None if x.profit_after_tax is None else x.profit_after_tax * UNIT_MULTIPLIER[x.unit]
        shares = prof["total_shares"].get(a.symbol)
        ratio = pat / x.eps / shares if pat and x.eps and shares else None
        rows.append(
            {
                "symbol": a.symbol,
                "id": str(a.announcement_id),
                "period_end": a.period_end.date(),
                "text": saved["method"],
                "unit": x.unit,
                "months": x.period_months,
                "pat_bn": None if pat is None else pat / 1e9,
                "eps": x.eps,
                "ratio": ratio,
                "pbt_tax": checks.pbt_tax_pat,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", required=True)
    parser.add_argument("--until", default=None)
    parser.add_argument("--symbols", required=True, help="Comma-separated, or 'kse100'")
    parser.add_argument("--exclude", default="", help="Comma-separated companies to leave out")
    parser.add_argument("--set", default=None, help="Company set from configs/psx_extraction_sets.yaml")
    parser.add_argument("--fresh", action="store_true", help="Leave out reports already used for scoring")
    parser.add_argument("--sample", type=int, default=None, help="Random sample size")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--process", action="store_true", help="Download and read the chosen reports first")
    parser.add_argument("-v", action="store_true", help="List every report")
    args = parser.parse_args()
    settings = Settings()
    lake = ParquetLake(settings.data_dir)
    if args.symbols == "kse100":
        snap = lake.read("index_constituents", "PSX")
        symbols = sorted(snap.loc[snap["snapshot_date"] == snap["snapshot_date"].max(), "symbol"])
    else:
        symbols = args.symbols.split(",")
    symbols = [s for s in symbols if s not in set(args.exclude.split(","))]
    sets = yaml.safe_load((CONFIG_DIR / "psx_extraction_sets.yaml").read_text(encoding="utf-8"))
    if args.set:
        symbols = [s for s in symbols if s in set(sets[args.set])]
    skip = set(sets["scored_report_ids"]) if args.fresh else set()
    df = evaluate(
        lake,
        settings.data_dir / "raw" / "psx",
        symbols,
        args.since,
        args.until,
        args.sample,
        args.seed,
        args.process,
        skip,
    )
    good = (df["ratio"] - 1).abs() <= TOLERANCE
    if args.v:
        pd.set_option("display.width", 200, "display.max_rows", 500)
        print(df.assign(ok=good).sort_values(["symbol", "period_end"]).round(3).to_string())
    print(summary(df, good))


def summary(df: pd.DataFrame, good: pd.Series) -> str:
    """Today's share count is only a fair test of recent reports: bonus issues change it over the years, so
    on older reports a mismatch may be a correct figure. Those are counted as unverified, not as wrong."""
    recent = pd.to_datetime(df["period_end"]) >= pd.Timestamp(RECENT_FROM)
    answered = df["ratio"].notna()
    lines = [
        f"reports {len(df)} | companies {df['symbol'].nunique()} "
        f"| answered {int(answered.sum())} ({answered.mean():.1%})"
    ]
    r, o = recent, ~recent
    if r.any():
        lines.append(
            f"recent ({int(r.sum())}): correct {int((good & r).sum())} ({(good & r).sum() / r.sum():.1%}) "
            f"| wrong {int((answered & ~good & r).sum())} | no answer {int((~answered & r).sum())}"
        )
    if o.any():
        lines.append(
            f"older ({int(o.sum())}): matches today's shares {int((good & o).sum())} "
            f"| unverified {int((answered & ~good & o).sum())} | no answer {int((~answered & o).sum())}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
