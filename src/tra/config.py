"""Configuration loading.

An experiment is described by YAML files merged in order:
``configs/base.yaml`` -> ``configs/markets/<market>.yaml`` -> ``configs/experiments/<name>.yaml``.
The merged config is validated with Pydantic and hashed so every result can be traced to its exact settings.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


class Settings(BaseSettings):
    """Machine-specific settings and secrets, read from environment variables or ``.env``."""

    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    data_dir: Path = REPO_ROOT / "data"
    sec_user_agent: str = ""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Instrument(_Strict):
    ticker: str
    sector: str


class MarketConfig(_Strict):
    name: Literal["US", "PSX"]
    timezone: str
    calendar: str
    close_time: str = Field(description="Local session close, HH:MM")
    benchmark: str
    universe: list[Instrument]

    @property
    def tickers(self) -> list[str]:
        return [i.ticker for i in self.universe]

    @property
    def sectors(self) -> dict[str, str]:
        return {i.ticker: i.sector for i in self.universe}


class DatesConfig(_Strict):
    start: date
    test_start: date
    end: date | None = None
    holdout_start: date | None = None

    @model_validator(mode="after")
    def _ordered(self) -> DatesConfig:
        if not self.start < self.test_start:
            raise ValueError("dates.start must be before dates.test_start")
        if self.end is not None and not self.test_start < self.end:
            raise ValueError("dates.test_start must be before dates.end")
        return self


class CostConfig(_Strict):
    commission_bps: float = 0.0
    slippage_bps: float = 5.0
    impact_k: float = 0.1
    min_trade_weight: float = Field(0.005, description="Skip trades smaller than this fraction of NAV")


class ModelConfig(_Strict):
    type: Literal["benchmark", "equal_weight", "momentum", "factor_composite", "lightgbm"]
    horizon: int = Field(10, description="Label horizon in trading days")
    retrain_every: int = Field(63, description="Trading days between retrains")
    embargo_days: int = 10
    min_train_days: int = 504
    cv_folds: int = 4
    param_grid: list[dict[str, Any]] = Field(default_factory=list)


class PortfolioConfig(_Strict):
    method: Literal["hold_benchmark", "equal", "inverse_vol", "mean_variance"]
    top_n: int | None = Field(10, description="Names held; None means the whole universe")
    target_vol: float | None = 0.11
    risk_aversion: float = 5.0
    turnover_penalty: float = 0.002
    cov_window: int = 126


class RiskConfig(_Strict):
    enabled: bool = True
    max_pos: float = 0.10
    max_sector: float = 0.30
    top5_max: float = 0.40
    gross_normal: float = 1.0
    gross_high_vol: float = 0.5
    high_vol_prob: float = 0.7
    dd_kill: float = 0.15
    dd_kill_gross: float = 0.30
    dd_recovery_window: int = 20
    adv_cap: float = 0.01
    adv_window: int = 20
    cvar_max: float = 0.03
    cvar_window: int = 252


class ExperimentConfig(_Strict):
    name: str
    description: str = ""
    seed: int = 7
    initial_nav: float = 1_000_000.0
    rebalance: Literal["never", "daily", "weekly", "monthly"]
    market: MarketConfig
    dates: DatesConfig
    model: ModelConfig
    portfolio: PortfolioConfig
    risk: RiskConfig
    costs: CostConfig

    def config_hash(self) -> str:
        """Stable short hash of the full config, saved with every run."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_experiment(experiment: str | Path, config_dir: Path = CONFIG_DIR) -> ExperimentConfig:
    """Load and validate an experiment by name (``b4_lightgbm``) or by path to its YAML file."""
    exp_path = Path(experiment)
    if not exp_path.suffix:
        exp_path = config_dir / "experiments" / f"{experiment}.yaml"
    exp = _read_yaml(exp_path)
    merged = _read_yaml(config_dir / "base.yaml")
    market_name = exp.get("market_name", merged.get("market_name", "US"))
    merged["market"] = _read_yaml(config_dir / "markets" / f"{market_name.lower()}.yaml")
    merged = _deep_merge(merged, exp)
    merged.pop("market_name", None)
    return ExperimentConfig.model_validate(merged)


def load_market(name: str, config_dir: Path = CONFIG_DIR) -> MarketConfig:
    return MarketConfig.model_validate(_read_yaml(config_dir / "markets" / f"{name.lower()}.yaml"))
