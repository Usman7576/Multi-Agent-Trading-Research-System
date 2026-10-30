import numpy as np
import pandas as pd
import pytest

from tra.data.psx_adjust import adjust_prices, expected_factor, validate_against_payouts

DAYS = pd.bdate_range("2024-01-01", periods=6)


def _bars(closes, ldcps, ticker="XYZ"):
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": DAYS[: len(closes)],
            "close": closes,
            "ldcp": ldcps,
            "known_at": DAYS[: len(closes)].tz_localize("UTC"),
        }
    )


def test_bonus_issue_creates_no_artificial_drop():
    # 25% bonus on day 3: price falls from 125 to ~100 but LDCP is reset to 100 by the exchange
    bars, events = adjust_prices(_bars([120, 125, 101, 102], [np.nan, 120, 100, 101]))
    rets = bars["adj_close"].pct_change().dropna()
    assert rets.abs().max() < 0.05  # the bonus is not a -19% return
    assert len(events) == 1 and events["ex_date"].iloc[0] == DAYS[2]
    assert events["factor"].iloc[0] == pytest.approx(0.8)
    assert events["implied_bonus_pct"].iloc[0] == pytest.approx(25.0)


def test_cash_dividend_is_part_of_total_return():
    # Rs 5 dividend: previous close 100, LDCP 95, close 95 -> 0% total return that day
    bars, events = adjust_prices(_bars([100, 95], [99, 95]))
    assert bars["adj_close"].iloc[1] == pytest.approx(bars["adj_close"].iloc[0])
    assert events["implied_cash_dividend"].iloc[0] == pytest.approx(5.0)


def test_adjustment_is_forward_only():
    """Adjusted prices up to day D do not change when later events are added."""
    early, _ = adjust_prices(_bars([100, 101, 102], [99, 100, 101]))
    later, _ = adjust_prices(_bars([100, 101, 102, 80, 81], [99, 100, 101, 81.6, 80]))
    pd.testing.assert_series_equal(early["adj_close"], later["adj_close"].iloc[:3], check_names=False)


def test_rounding_differences_are_not_events():
    _, events = adjust_prices(_bars([100.0, 100.5], [99.9, 100.01]))
    assert events.empty


def test_expected_factors():
    assert expected_factor("dividend", 55, prev_close=110) == pytest.approx((110 - 5.5) / 110)
    assert expected_factor("bonus", 10, prev_close=50) == pytest.approx(1 / 1.1)


def test_validation_matches_payout_to_ex_date():
    _, events = adjust_prices(_bars([100, 101, 96.5, 97], [99, 100, 96, 96.5]))
    payouts = pd.DataFrame(
        {"symbol": ["XYZ"], "kind": ["dividend"], "percent": [50.0], "book_closure_from": [DAYS[4]]}
    )
    out = validate_against_payouts(events, payouts, DAYS)
    assert bool(out["matched"].iloc[0]) and out["ex_date"].iloc[0] == DAYS[2]
    assert abs(out["factor_error"].iloc[0]) < 0.001
