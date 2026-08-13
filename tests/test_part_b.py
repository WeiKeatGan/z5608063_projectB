"""Unit tests for the Part B model code (portfolios, sentiment, fusion).

    python -m pytest -q   (from the repo root or this folder)
"""
import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import portfolios  # noqa: E402


# ---------------------------------------------------------------------------
# portfolios
# ---------------------------------------------------------------------------

def test_weights_sum_to_one_and_long_only():
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(252, 5)), columns=list("ABCDE"))
    for method in portfolios.METHODS:
        w = portfolios.compute_weights(rets, method)
        assert abs(w.sum() - 1.0) < 1e-9, method
        assert (w >= -1e-9).all(), method


def test_weights_differ_across_methods():
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0.001, 0.02, size=(252, 4)),
                        columns=list("ABCD"))
    ws = {m: portfolios.compute_weights(rets, m).to_numpy() for m in portfolios.METHODS}
    pairs = [("min_variance", "max_sharpe"), ("min_variance", "risk_parity"),
             ("risk_parity", "equal_weight"), ("equal_weight", "min_variance")]
    for a, b in pairs:
        assert np.abs(ws[a] - ws[b]).max() > 1e-6, (a, b)


def test_weights_start_next_day_no_lookahead():
    """Weights formed at rebalance date t must NOT be applied to date t itself."""
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    rets = pd.DataFrame({"A": np.zeros(6), "B": np.zeros(6)}, index=dates)
    rets.loc[dates[2], "B"] = 0.1          # realised on the rebalance date
    weights = pd.DataFrame({"A": [1.0, 1.0], "B": [0.0, 0.0]},
                           index=[dates[2], dates[4]])
    port, active = portfolios.evaluate_weights(rets, weights)
    assert dates[2] not in port.index            # date-2 return excluded
    assert port.index.min() == dates[3]          # first live date is t+1
    assert abs(port.loc[dates[3]]) < 1e-12       # B's 0.1 never enters


def test_oos_backtest_shapes():
    dates = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=300, freq="B"))
    rng = np.random.default_rng(2)
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(300, 4)),
                        index=dates, columns=list("ABCD"))
    bt = portfolios.oos_backtest(rets, method="min_variance", window=252,
                                 rebalance="monthly")
    assert bt["first_live_date"] == dates[253]
    assert len(bt["returns"]) == 47                 # days 253..299
    assert bt["weights"].index.min() == dates[252]
    assert abs(bt["growth"].iloc[-1] - (1.0 + bt["returns"]).prod()) < 1e-12


def test_performance_metrics_known_values():
    r = pd.Series(np.full(252, 0.001))
    m = portfolios.performance_metrics(r, periods_per_year=252)
    assert abs(m["annualized_return"] - (1.001 ** 252 - 1)) < 1e-9
    assert m["annualized_vol"] < 1e-12
    assert np.isnan(m["sharpe"])                   # zero-vol guard

    cum = pd.Series([1.0, 1.1, 1.05, 0.95, 1.0])
    r2 = cum.pct_change().dropna()
    m2 = portfolios.performance_metrics(r2, periods_per_year=4)
    assert abs(m2["max_drawdown"] - (0.95 / 1.1 - 1)) < 1e-9


# ---------------------------------------------------------------------------
# sentiment + fusion
# ---------------------------------------------------------------------------

def test_sentiment_scores_and_index():
    from src import sentiment
    panel = pd.DataFrame([
        {"ticker": "AAA", "date": pd.Timestamp("2020-01-02"), "sector": "S1",
         "n_headlines": 2, "headlines": ["company beats expectations",
                                         "stock surges on strong results"]},
        {"ticker": "BBB", "date": pd.Timestamp("2020-01-02"), "sector": "S1",
         "n_headlines": 1, "headlines": ["company announces massive losses"]},
        {"ticker": "CCC", "date": pd.Timestamp("2020-01-02"), "sector": "S2",
         "n_headlines": 0, "headlines": []},
    ])
    scores = sentiment.score_headlines(panel)
    assert list(scores.columns) == ["ticker", "date", "sector", "n_headlines", "vader_mean"]
    assert scores.loc[0, "vader_mean"] > 0
    assert scores.loc[1, "vader_mean"] < 0
    idx = sentiment.sector_sentiment_index(scores)
    assert "S1" in idx.columns and "S2" in idx.columns
    assert idx.loc[idx.index[0], "S1"] == 0.5 * (scores.loc[0, "vader_mean"]
                                                 + scores.loc[1, "vader_mean"])
    assert abs(idx.loc[idx.index[0], "S2"]) < 1e-12   # neutral fill for no news


def test_fusion_tilts_toward_positive_news():
    from src import fusion
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    weights = pd.DataFrame({"AAA": [0.5], "BBB": [0.5], "BTC-USD": [0.0]},
                           index=[dates[5]])
    scores = pd.DataFrame([
        {"ticker": "AAA", "date": dates[i], "vader_mean": 0.9,
         "sector": "S1", "n_headlines": 1} for i in range(1, 5)
    ] + [
        {"ticker": "BBB", "date": dates[i], "vader_mean": -0.9,
         "sector": "S1", "n_headlines": 1} for i in range(1, 5)
    ])
    fused = fusion.apply_sentiment(weights, scores, equity_tickers=["AAA", "BBB"],
                                   strength=0.25, lookback=5)
    assert fused.loc[dates[5], "AAA"] > 0.5          # tilted toward positive news
    assert fused.loc[dates[5], "BBB"] < 0.5
    assert abs(fused.loc[dates[5], ["AAA", "BBB"]].sum() - 1.0) < 1e-9
    assert fused.loc[dates[5], "BTC-USD"] == 0.0


if __name__ == "__main__":
    for fn in [test_weights_sum_to_one_and_long_only,
               test_weights_differ_across_methods,
               test_weights_start_next_day_no_lookahead,
               test_oos_backtest_shapes,
               test_performance_metrics_known_values,
               test_sentiment_scores_and_index,
               test_fusion_tilts_toward_positive_news]:
        fn()
    print("all tests OK")
