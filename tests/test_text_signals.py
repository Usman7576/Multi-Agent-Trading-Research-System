import numpy as np
import pandas as pd
import pytest

from tra.config import load_experiment
from tra.features.text import text_features
from tra.llm.extract import HeadlineSignal, filing_body, mask, quote_verified

DAYS = pd.bdate_range("2024-01-01", periods=60)
CLOSES = pd.Series(DAYS.tz_localize("UTC") + pd.Timedelta(hours=12), index=DAYS)


def _signals(rows):
    return pd.DataFrame(rows, columns=["doc_id", "symbol", "known_at", "lm_polarity", "llm_sentiment"])


def test_headline_after_the_close_counts_next_session():
    day = DAYS[10]
    s = _signals([("a", "LUCK", day + pd.Timedelta(hours=13), 1.0, 1.0)])
    s["known_at"] = pd.to_datetime(s["known_at"]).dt.tz_localize("UTC")
    f = text_features(s, CLOSES, "llm")
    assert (day, "LUCK") not in f.index and f.loc[(DAYS[11], "LUCK"), "news_llm_sentiment"] == pytest.approx(
        1.0
    )


def test_headline_before_the_close_counts_same_session():
    day = DAYS[10]
    s = _signals([("a", "LUCK", day + pd.Timedelta(hours=11), -1.0, -1.0)])
    s["known_at"] = pd.to_datetime(s["known_at"]).dt.tz_localize("UTC")
    assert text_features(s, CLOSES, "lm").loc[(day, "LUCK"), "news_lm_sentiment"] == pytest.approx(-1.0)


def test_text_features_do_not_change_when_later_news_exists():
    """The feature-shift test for news: adding later headlines leaves earlier values unchanged."""
    rng = np.random.default_rng(0)
    rows = [
        (
            f"d{i}",
            "LUCK" if i % 2 else "HBL",
            DAYS[rng.integers(0, 60)] + pd.Timedelta(hours=9),
            0.0,
            float(rng.uniform(-1, 1)),
        )
        for i in range(40)
    ]
    s = _signals(rows)
    s["known_at"] = pd.to_datetime(s["known_at"]).dt.tz_localize("UTC")
    cut = DAYS[30]
    early = s[s["known_at"] <= CLOSES[cut]]
    full = text_features(s, CLOSES, "llm")
    part = text_features(early, CLOSES, "llm")
    a = full[full.index.get_level_values("date") <= cut]
    b = part[part.index.get_level_values("date") <= cut]
    pd.testing.assert_frame_equal(a, b.reindex(a.index))


def test_old_news_fades_towards_new_news():
    s = _signals([("a", "LUCK", DAYS[5], 0.0, 1.0), ("b", "LUCK", DAYS[20], 0.0, -1.0)])
    s["known_at"] = pd.to_datetime(s["known_at"]).dt.tz_localize("UTC")
    f = text_features(s, CLOSES, "llm").xs("LUCK", level="ticker")["news_llm_sentiment"]
    assert f.loc[DAYS[10]] == pytest.approx(0.5) and f.loc[DAYS[20]] < 0 < f.loc[DAYS[19]]


def test_no_scored_news_gives_no_features():
    s = _signals([("a", "LUCK", DAYS[5], 0.0, None)])
    s["known_at"] = pd.to_datetime(s["known_at"]).dt.tz_localize("UTC")
    assert text_features(s, CLOSES, "llm").empty


def test_mask_hides_company_and_dates():
    masked = mask(
        "Lucky Cement Limited profit jumps on 3 March 2024; LUCK shares rise", "LUCK", "Lucky Cement Limited"
    )
    assert (
        "lucky" not in masked.lower()
        and "LUCK" not in masked
        and "2024" not in masked
        and "March" not in masked
    )
    assert "profit jumps" in masked


@pytest.mark.parametrize(
    ("quote", "ok"), [("profit jumps", True), ("PROFIT JUMPS", True), ("profits soared", False), ("", False)]
)
def test_quote_must_come_from_the_headline(quote, ok):
    signal = HeadlineSignal(sentiment=0.5, event="earnings", key_quote=quote)
    assert quote_verified(signal, "the company profit jumps") is ok


def test_sentiment_outside_range_is_rejected():
    with pytest.raises(ValueError):
        HeadlineSignal(sentiment=2.0, event="other", key_quote="x")


def test_adding_text_setting_leaves_existing_config_hashes_unchanged():
    """Trials are counted by config hash: the new setting must not turn old runs into new trials."""
    base = load_experiment("psx_b4_lightgbm")
    assert base.model.text_features is None
    assert base.config_hash() == "0054d620c535"
    assert load_experiment("psx_b5_text_llm").config_hash() != base.config_hash()


def test_quote_check_ignores_spaces_lost_in_pdf_text():
    signal = HeadlineSignal(sentiment=0.5, event="expansion", key_quote="Plant resumed operations")
    assert quote_verified(signal, "the Plantresumed operations on Monday")


def test_filing_body_skips_the_letterhead():
    text = "ACME LIMITED Karachi The General Manager PSX Dear Sir, we convey the following information: Plant resumed."
    assert filing_body(text) == "Plant resumed."
