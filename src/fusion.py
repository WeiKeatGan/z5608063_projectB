"""Station 3 (extension) - fuse sentiment into the equity funds.

Tilt the fund's equity weights toward names with strong recent headline
sentiment, keep the equity/crypto split fixed, then re-run the backtest to see
whether the fusion adds value out-of-sample. An honest negative result,
explained, is good work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_sentiment(weights: pd.DataFrame,
                    scores: pd.DataFrame,
                    equity_tickers: list[str],
                    strength: float = 0.25,
                    lookback: int = 5) -> pd.DataFrame:
    """Tilt a weight schedule toward high recent-sentiment equity names.

    Parameters
    ----------
    weights : weight schedule from portfolios.oos_backtest (index = rebalance
        dates, columns = assets).
    scores : per-ticker-day sentiment [ticker, date, vader_mean].
    equity_tickers : the fund's equity assets (only these get tilted; crypto
        weights are untouched).
    strength : tilt intensity. Equity weight i is multiplied by
        1 + strength * z_i, where z_i is the cross-sectional z-score of recent
        sentiment among the fund's equity names.
    lookback : recent sentiment is the mean of the `lookback` trading days
        strictly BEFORE the rebalance date (never on it) - decision-safe.

    Returns
    -------
    Fused weight schedule (same index/columns as `weights`). The equity sleeve
    is renormalised to its original total so the equity/crypto split is kept.
    """
    wide = scores.pivot_table(index="date", columns="ticker",
                              values="vader_mean", dropna=False).fillna(0.0)
    wide = wide.sort_index()

    fused = weights.copy()
    for t in weights.index:
        hist = wide.loc[wide.index < t].tail(lookback)
        if hist.empty:
            continue
        recent = hist.mean(axis=0)
        eq = [c for c in weights.columns
              if c in equity_tickers and c in recent.index]
        if len(eq) < 2:
            continue
        vals = recent[eq].to_numpy()
        std = vals.std()
        z = (vals - vals.mean()) / std if std > 1e-12 else np.zeros_like(vals)
        base = fused.loc[t, eq].to_numpy()
        total = base.sum()
        if total <= 0:
            continue
        tilt = np.clip(base * (1.0 + strength * z), 0.0, None)
        fused.loc[t, eq] = tilt / tilt.sum() * total
    return fused
def sensitivity_grid(base_bt: dict, panel, scores, equity_cols,
                     strengths=(0.1, 0.25, 0.5), lookbacks=(3, 5, 10),
                     periods_per_year: int = 252) -> pd.DataFrame:
    """Sharpe of the fused fund across a grid of (strength, lookback) pairs,
    for robustness reporting only - NEVER used to select the headline
    strength/lookback values, which are fixed defaults set before any results
    were seen (see ai/ prompt log)."""
    from src.portfolios import evaluate_weights, performance_metrics

    rows = []
    for s in strengths:
        for lb in lookbacks:
            fused_w = apply_sentiment(base_bt["weights"], scores, equity_cols,
                                      strength=s, lookback=lb)
            fused_port, _ = evaluate_weights(panel, fused_w)
            m = performance_metrics(fused_port, periods_per_year=periods_per_year)
            rows.append({"strength": s, "lookback": lb, "sharpe": m["sharpe"],
                        "annualized_return": m["annualized_return"],
                        "max_drawdown": m["max_drawdown"]})
    return pd.DataFrame(rows)
