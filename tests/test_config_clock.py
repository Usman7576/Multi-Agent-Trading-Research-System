import pandas as pd
import pytest

from tra.clock import SimClock
from tra.config import load_experiment

EXPERIMENTS = ["b0_benchmark", "b1_equal_weight", "b2_momentum", "b3_factor_composite", "b4_lightgbm"]


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_experiments_load(name):
    cfg = load_experiment(name)
    assert cfg.name == name
    assert cfg.market.name == "US"
    assert len(cfg.market.tickers) == len(set(cfg.market.tickers))


def test_override_merges_deeply():
    cfg = load_experiment("b4_lightgbm_no_risk")
    assert cfg.risk.enabled is False
    assert cfg.risk.max_pos == 0.10  # untouched keys keep base values


def test_config_hash_changes_with_settings():
    base = load_experiment("b4_lightgbm").config_hash()
    assert base != load_experiment("b4_lightgbm_no_risk").config_hash()


def test_sim_clock_only_moves_forward():
    clock = SimClock("2020-01-02")
    clock.set("2020-01-03")
    assert clock.now() == pd.Timestamp("2020-01-03", tz="UTC")
    with pytest.raises(ValueError):
        clock.set("2020-01-01")
