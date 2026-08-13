"""Station 3 - your sentiment model and sector index from news headlines.

Score each headline with VADER plus the finance lexicon extension, aggregate to
a daily per-ticker score (mean compound), then to an equal-weight sector index.
Headlines are a noisy proxy, so the signal is lagged at decision time (see
fusion.py): a headline aligned to trading day t is first usable for the
decision on day t+1 or later.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from src.finance_lexicon import FINANCE_LEXICON

_analyzer = None


def _get_analyzer():
    """Build the VADER analyzer once, with the finance lexicon applied.

    nltk's vader_lexicon must be downloaded once on any clean machine - that is
    a build step (scripts/run_part_b.py), NOT something the deployed app does.
    """
    global _analyzer
    if _analyzer is None:
        import nltk
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        _analyzer = SentimentIntensityAnalyzer()
        _analyzer.lexicon.update(FINANCE_LEXICON)
    return _analyzer


@functools.lru_cache(maxsize=2**20)
def _compound(title: str) -> float:
    """VADER compound score for one headline, cached per unique title."""
    return _get_analyzer().polarity_scores(title)["compound"]


def score_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """Score the assembled headline panel.

    Parameters
    ----------
    panel : per-ticker-day panel from features.assemble_headline_panel
        (columns ticker, date, sector, n_headlines, headlines).

    Returns
    -------
    DataFrame with one row per ticker-day: [ticker, date, sector,
    n_headlines, vader_mean], where vader_mean is the mean VADER compound
    across that day's headlines (0 if there are none).
    """
    rows = []
    for idx, row in panel.iterrows():
        titles = row["headlines"]
        scores = [_compound(t) for t in titles]
        rows.append({
            "ticker": row["ticker"],
            "date": row["date"],
            "sector": row["sector"],
            "n_headlines": len(titles),
            "vader_mean": float(np.mean(scores)) if scores else 0.0,
        })
    out = pd.DataFrame(rows)
    return out[["ticker", "date", "sector", "n_headlines", "vader_mean"]]


def sector_sentiment_index(scores: pd.DataFrame, no_news: str = "neutral") -> pd.DataFrame:
    """Daily equal-weight sentiment index per sector.

    Pivots per-ticker-day sentiment to a date x ticker grid, then averages the
    tickers within each sector. Ticker-days with no headlines are treated as
    neutral (0) - no news is no signal, so sparse sectors drift toward zero
    instead of the index jumping whenever headline coverage thins. (This is the
    justification documented in the report; 'drop' is available as an
    alternative.)

    Returns a DataFrame indexed by date with one column per sector.
    """
    if no_news not in ("neutral", "drop"):
        raise ValueError("no_news must be 'neutral' or 'drop'")
    sectors = scores[["ticker", "sector"]].drop_duplicates()
    ticker_to_sector = dict(zip(sectors["ticker"], sectors["sector"]))

    wide = scores.pivot_table(index="date", columns="ticker",
                              values="vader_mean", dropna=False).sort_index()
    if no_news == "neutral":
        wide = wide.fillna(0.0)

    out = {}
    for sector in sorted(sectors["sector"].unique()):
        tickers = [t for t, s in ticker_to_sector.items() if s == sector]
        cols = [t for t in tickers if t in wide.columns]
        out[sector] = wide[cols].mean(axis=1)
    return pd.DataFrame(out)


def ticker_sentiment_wide(scores: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker-day sentiment as a date x ticker grid (neutral-filled).

    Convenience for the fusion step, which needs recent per-ticker sentiment
    at each rebalance date.
    """
    wide = scores.pivot_table(index="date", columns="ticker",
                              values="vader_mean", dropna=False).fillna(0.0)
    return wide.sort_index()
