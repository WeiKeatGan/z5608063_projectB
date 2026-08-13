"""Station 2 - your features: return features and text assembly.

Build your return features here, and assemble the headlines into a daily text
panel. Scoring the text is the Station 3 sentiment model (see src/sentiment.py).
"""
import pandas as pd


def daily_returns(prices: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Simple daily returns per ticker, long format.

    Returns a DataFrame with columns [ticker, date, daily_return], one row per
    (ticker, date). Returns are computed within each ticker via pct_change, so
    each ticker's first observed date has a NaN return (no prior price to
    compare against) - this is expected, not a bug.
    """
    df = prices.sort_values(["ticker", "date"]).copy()
    df["daily_return"] = df.groupby("ticker")[price_col].pct_change()
    return df[["ticker", "date", "daily_return"]]


def assemble_headline_panel(headlines: pd.DataFrame) -> pd.DataFrame:
    """Assemble headlines into a daily panel per ticker and sector.

    Expects `headlines` to already be date-aligned to the equity trading
    calendar (see etl.load_clean_headlines). Groups to one row per
    (ticker, date), preserving:
      - n_headlines: count of headlines that day for that ticker
      - headlines: the RAW, unmodified headline text as a list (not stripped,
        not scored - VADER needs the full text, including stopwords, and
        scoring itself is Station 3, not here).

    Returns columns: [ticker, date, sector, n_headlines, headlines].
    """
    panel = (
        headlines
        .groupby(["ticker", "sector", "date"])["title"]
        .agg(n_headlines="count", headlines=list)
        .reset_index()
    )
    return panel[["ticker", "date", "sector", "n_headlines", "headlines"]]
