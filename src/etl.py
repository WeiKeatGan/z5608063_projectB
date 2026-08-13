"""Station 1 - ETL: load, clean, and quality-check the data.

Loads raw data through src.data_access, runs integrity checks (missing-date
audit, duplicate detection, outlier screening), and produces clean datasets
ready for Station 2 feature engineering.

Output:
    clean_equities  - equity prices with daily returns, deduplicated
    clean_crypto    - crypto prices capped at 2023-12-31 with daily returns
    clean_headlines - news headlines normalized and deduplicated on ticker+date+title
    integrity_report - dict of all checks and findings
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from src import data_access


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trading_calendar(equities: pd.DataFrame | None = None) -> pd.DatetimeIndex:
    """The REAL equity trading calendar: the actual dates the equity panel
    trades on (excludes weekends and market holidays automatically, since
    the data itself never has rows for closed days)."""
    if equities is None:
        equities = data_access.load_equity_prices()
    return pd.DatetimeIndex(sorted(equities["date"].unique()))


def _missing_dates(dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return calendar dates that are missing from the dataset."""
    return calendar.difference(dates)


# ---------------------------------------------------------------------------
# Equity cleaning
# ---------------------------------------------------------------------------

def load_clean_equities() -> tuple[pd.DataFrame, dict]:
    """Load equity prices, run integrity checks, compute daily returns.

    Returns
    -------
    df : pd.DataFrame
        Clean equity panel with columns [ticker, date, open, high, low, close,
        adjClose, volume, sector, daily_return].
    report : dict
        Integrity check findings for the equity dataset.
    """
    df = data_access.load_equity_prices()
    report: dict = {"dataset": "equity_prices", "raw_rows": len(df)}

    # --- duplicate check (ticker + date) ---
    dup_mask = df.duplicated(subset=["ticker", "date"], keep=False)
    n_dups = dup_mask.sum()
    report["duplicate_ticker_date"] = int(n_dups)
    if n_dups > 0:
        df = df.drop_duplicates(subset=["ticker", "date"], keep="first")

    # --- missing-date audit (PER TICKER against the panel's own calendar) ---
    # Compares each ticker's observed dates against the full set of dates the
    # PANEL trades on. A gap here means one ticker is missing days that the
    # rest of the panel has data for - a genuine per-ticker coverage issue,
    # not just "the panel doesn't trade on weekends."
    cal = _trading_calendar(df)
    ticker_gaps = {}
    for tk, g in df.groupby("ticker"):
        tk_dates = pd.DatetimeIndex(g["date"].unique())
        missing_tk = cal.difference(tk_dates)
        if len(missing_tk) > 0:
            ticker_gaps[tk] = len(missing_tk)
    report["tickers_with_gaps"] = len(ticker_gaps)
    report["total_missing_ticker_days"] = int(sum(ticker_gaps.values()))
    report["worst_gap_tickers"] = dict(sorted(ticker_gaps.items(), key=lambda x: -x[1])[:10])

    # --- compute daily returns (on equity's own calendar, before any merge) ---
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["daily_return"] = df.groupby("ticker")["adjClose"].pct_change()

    # --- outlier / extreme-value screen on returns (PER-TICKER z-score) ---
    # Pooled z-scores would judge a quiet stock against the whole cross-
    # section's volatility, so we standardise each ticker against its own
    # return distribution instead.
    grp = df.groupby("ticker")["daily_return"]
    z_series = ((df["daily_return"] - grp.transform("mean")) / grp.transform("std")).fillna(0)
    extreme_mask = z_series.abs() > 4
    n_extreme = extreme_mask.sum()
    report["outlier_returns_4sigma"] = int(n_extreme)
    if n_extreme > 0:
        extreme_rows = df.loc[extreme_mask, ["ticker", "date", "daily_return"]].copy()
        extreme_rows["z_score"] = z_series[extreme_mask].values
        report["outlier_details"] = extreme_rows.sort_values("z_score", key=abs, ascending=False).head(10).to_dict("records")
    else:
        report["outlier_details"] = []

    # --- value range check ---
    report["min_adj_close"] = float(df["adjClose"].min())
    report["max_adj_close"] = float(df["adjClose"].max())
    report["negative_volume_count"] = int((df["volume"] < 0).sum())

    report["clean_rows"] = len(df)
    report["tickers"] = sorted(df["ticker"].unique().tolist())
    report["date_range"] = [str(df["date"].min().date()), str(df["date"].max().date())]
    return df, report


# ---------------------------------------------------------------------------
# Crypto cleaning
# ---------------------------------------------------------------------------

def load_clean_crypto() -> tuple[pd.DataFrame, dict]:
    """Load crypto prices, cap at 2023-12-31, run checks, compute returns.

    Returns
    -------
    df : pd.DataFrame
        Clean crypto panel with daily returns computed on the 365-day calendar.
    report : dict
        Integrity check findings.
    """
    df = data_access.load_crypto_prices()
    report: dict = {"dataset": "crypto_prices", "raw_rows": len(df)}

    # --- cap sample at 2023-12-31 (10 stray rows dated 2024-01-01) ---
    pre_cap = len(df)
    df = df[df["date"] <= "2023-12-31"].copy()
    report["rows_removed_2024"] = pre_cap - len(df)

    # --- duplicate check (ticker + date) ---
    dup_mask = df.duplicated(subset=["ticker", "date"], keep=False)
    n_dups = dup_mask.sum()
    report["duplicate_ticker_date"] = int(n_dups)
    if n_dups > 0:
        df = df.drop_duplicates(subset=["ticker", "date"], keep="first")

    # --- missing-date audit (crypto trades 365 days/year) ---
    crypto_dates = df["date"].drop_duplicates().sort_values()
    crypto_cal = pd.date_range(
        crypto_dates.min(), crypto_dates.max(), freq="D"
    )
    missing = crypto_cal.difference(crypto_dates)
    report["missing_calendar_days"] = len(missing)

    # --- compute daily returns on crypto's own 365-day calendar ---
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["daily_return"] = df.groupby("ticker")["adjClose"].pct_change()

    # --- outlier / extreme-value screen on returns (PER-TICKER z-score) ---
    grp = df.groupby("ticker")["daily_return"]
    z_series = ((df["daily_return"] - grp.transform("mean")) / grp.transform("std")).fillna(0)
    extreme_mask = z_series.abs() > 4
    n_extreme = extreme_mask.sum()
    report["outlier_returns_4sigma"] = int(n_extreme)
    if n_extreme > 0:
        extreme_rows = df.loc[extreme_mask, ["ticker", "date", "daily_return"]].copy()
        extreme_rows["z_score"] = z_series[extreme_mask].values
        report["outlier_details"] = extreme_rows.sort_values("z_score", key=abs, ascending=False).head(10).to_dict("records")
    else:
        report["outlier_details"] = []

    report["clean_rows"] = len(df)
    report["tickers"] = sorted(df["ticker"].unique().tolist())
    report["date_range"] = [str(df["date"].min().date()), str(df["date"].max().date())]
    return df, report


# ---------------------------------------------------------------------------
# News headline cleaning
# ---------------------------------------------------------------------------

def load_clean_headlines(equities: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Load news headlines, normalise timezone, deduplicate, align to trading days.

    Returns
    -------
    df : pd.DataFrame
        Clean headline panel with date aligned to equity trading day.
    report : dict
        Integrity check findings.
    """
    df = data_access.load_news_headlines()
    report: dict = {"dataset": "news_headlines", "raw_rows": len(df)}

    # --- normalise timezone: UTC-aware -> tz-naive (prices are tz-naive) ---
    if pd.api.types.is_datetime64_any_dtype(df["date"]) and df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    # --- deduplicate on ticker + date + title (NOT just ticker + date) ---
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date", "title"], keep="first")
    report["exact_duplicate_rows"] = before - len(df)

    # --- align every headline to its equity trading day ---
    # A headline on a trading day stays on that day; a weekend/holiday headline
    # moves to the next trading day.  merge_asof keeps the LEFT key, so we
    # must write the matched RIGHT key back into the date column.
    cal = _trading_calendar(equities)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    cal_df = pd.DataFrame({"trading_day": pd.to_datetime(cal)})
    # Ensure both sides share the same datetime dtype before merge
    target_dtype = df["date"].dtype
    cal_df["trading_day"] = cal_df["trading_day"].astype(target_dtype)
    df = pd.merge_asof(
        df.sort_values("date"),
        cal_df.sort_values("trading_day"),
        left_on="date",
        right_on="trading_day",
        direction="forward",
    )
    # Replace original date with the aligned trading day
    n_unaligned = df["trading_day"].isna().sum()
    report["unaligned_headlines"] = int(n_unaligned)
    df["date"] = df["trading_day"]
    df = df.drop(columns=["trading_day"])
    df = df.dropna(subset=["date"])

    # --- summary stats ---
    report["clean_rows"] = len(df)
    report["unique_tickers"] = int(df["ticker"].nunique())
    report["unique_dates"] = int(df["date"].nunique())
    report["date_range"] = [str(df["date"].min().date()), str(df["date"].max().date())]
    report["headlines_per_ticker"] = (
        df.groupby("ticker").size().describe().to_dict()
    )
    return df, report


# ---------------------------------------------------------------------------
# Calendar merge: crypto returns onto equity trading calendar
# ---------------------------------------------------------------------------

def merge_panels(
    equities: pd.DataFrame,
    crypto: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Left-merge crypto daily returns onto the equity trading calendar.

    This intentionally excludes weekend-only crypto moves, which a fund trading
    on equity days could not act on. Returns are already computed within each
    panel before this merge.

    Returns
    -------
    combined : pd.DataFrame
        Wide panel with equity and crypto daily returns, indexed by
        (date, ticker).
    report : dict
        Merge statistics.
    """
    report: dict = {}

    eq_wide = equities.pivot_table(
        index="date", columns="ticker", values="daily_return", dropna=False
    )
    crypto_wide = crypto.pivot_table(
        index="date", columns="ticker", values="daily_return", dropna=False
    )
    crypto_aligned = crypto_wide.reindex(eq_wide.index)

    combined = pd.concat(
        [eq_wide.add_suffix("_EQ"), crypto_aligned.add_suffix("_CR")],
        axis=1,
    )

    report["equity_tickers"] = int(eq_wide.shape[1])
    report["crypto_tickers"] = int(crypto_wide.shape[1])
    report["combined_columns"] = int(combined.shape[1])
    report["calendar_days"] = int(combined.shape[0])
    report["date_range"] = [str(combined.index.min().date()), str(combined.index.max().date())]

    return combined, report
