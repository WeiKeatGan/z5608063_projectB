"""NoBearTrader - systematic multi-asset funds with news-sentiment analytics.

The app is a pure viewer over the precomputed results/ artifacts (written by
scripts/run_part_b.py). It never runs VADER and never recomputes backtests, so
it stays light enough for the Streamlit free tier. Raw data loads only via
src/data_access.py for the Data tab.

Run locally:   streamlit run streamlit_app.py
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src import data_access  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent

FT_BG = "#FFF1E5"
FT_BLUE = "#0F5499"
FT_PINK = "#E6007E"
FT_INK = "#33302E"
FT_MUTED = "#66605C"

st.set_page_config(page_title="NoBearTrader", layout="wide",
                   page_icon=":chart_with_upwards_trend:")

st.markdown(f"""
<style>
  html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background-color: {FT_BG};
  }}
  [data-testid="stSidebar"] {{
    background-color: #FBEBD8;
  }}
  h1, h2, h3 {{ color: {FT_INK}; }}
  .nb-hero {{
    color: {FT_BLUE}; font-size: 2.4rem; font-weight: 700; letter-spacing: -0.5px;
    padding-bottom: 0.25rem;
  }}
  .nb-tagline {{ color: {FT_MUTED}; font-size: 1rem; margin-top: -0.25rem; }}
  .nb-metric {{
    background: white; border-left: 4px solid {FT_BLUE}; border-radius: 6px;
    padding: 0.9rem 1.1rem; margin: 0.35rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}
  .nb-metric .k {{ font-size: 0.75rem; color: {FT_MUTED}; text-transform: uppercase;
                   letter-spacing: 0.05em; }}
  .nb-metric .v {{ font-size: 1.55rem; font-weight: 700; color: {FT_INK}; }}
  .nb-foot {{ color: {FT_MUTED}; font-size: 0.8rem; margin-top: 2rem; }}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="nb-hero">NoBearTrader</div>', unsafe_allow_html=True)
st.markdown('<div class="nb-tagline">Systematically managed multi-asset funds, '
            'backtested out-of-sample, with news-sentiment analytics.</div>',
            unsafe_allow_html=True)

PERIODS = 252  # funds trade on the equity calendar


# ---------------------------------------------------------------------------
# Precomputed artifacts (the app reads these; never recomputes the models)
# ---------------------------------------------------------------------------

def _load(name: str, **kwargs) -> pd.DataFrame:
    path = ROOT / "results" / "data" / name
    if not path.exists():
        st.error(f"Missing precomputed artifact {name}. Run "
                 "`python scripts/run_part_b.py` first.")
        st.stop()
    return pd.read_csv(path, **kwargs)


@st.cache_data(ttl=86_400, show_spinner=False)
def fund_returns() -> pd.DataFrame:
    df = _load("fund_returns.csv", parse_dates=["date"])
    return df.sort_values(["fund", "date"])


@st.cache_data(ttl=86_400, show_spinner=False)
def fund_weights() -> pd.DataFrame:
    return _load("fund_weights.csv", parse_dates=["date"])


@st.cache_data(ttl=86_400, show_spinner=False)
def sector_sentiment() -> pd.DataFrame:
    return _load("sector_sentiment_index.csv", parse_dates=["date"])


@st.cache_data(ttl=86_400, show_spinner=False)
def metrics() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "tables" / "performance_metrics.csv")


@st.cache_data(ttl=86_400, show_spinner=False)
def fusion_returns() -> pd.DataFrame:
    df = _load("fusion_returns.csv", parse_dates=["date"])
    return df.sort_values(["fund", "date"])


def _metrics_from_daily(r: pd.Series, periods_per_year: float = PERIODS) -> dict:
    r = r.dropna()
    n = len(r)
    if n == 0:
        return {"Return": 0.0, "Vol": 0.0, "Sharpe": 0.0, "Max DD": 0.0}
    total = float(np.prod(1.0 + r.to_numpy()))
    ann_ret = total ** (periods_per_year / n) - 1.0
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    cum = (1.0 + r).cumprod()
    max_dd = float((cum / cum.cummax() - 1.0).min())
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else np.nan
    return {"Return": ann_ret, "Vol": ann_vol, "Sharpe": sharpe, "Max DD": max_dd}


def _current_holdings(fund: str, top_n: int = 12) -> pd.DataFrame:
    w = fund_weights()
    fw = w[w["fund"] == fund]
    last = fw["date"].max()
    holdings = (fw[fw["date"] == last]
                .sort_values("weight", ascending=False)
                .head(top_n)
                .reset_index(drop=True))
    return holdings[["ticker", "weight"]]


def _format_pct(x: float, digits: int = 1) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:+.1f}%"


# ---------------------------------------------------------------------------
# Sidebar - product info
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### NoBearTrader")
    st.caption("All funds are walk-forward out-of-sample backtests on the "
               "FINS3645 dataset (2020-2023): weights formed from past data "
               "only, monthly rebalancing, long-only.")
    st.markdown("**Methodology**")
    st.markdown("· 3 asset families (Equity / Crypto / Combined)\n"
                "· 4 optimisation methods (min-variance, max-Sharpe, risk "
                "parity, equal weight)\n"
                "· Sentiment from VADER + a finance lexicon, lagged one "
                "trading day\n"
                "· Sharpe uses risk-free rate = 0")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_funds, tab_alloc, tab_sent, tab_data = st.tabs(
    ["Funds & fact sheets", "Build your allocation", "Sentiment",
     "Data inventory"])


# ============================================================ FUNDS
with tab_funds:
    st.subheader("Compare the funds")
    m = metrics().copy()
    display = m[["fund", "annualized_return", "annualized_vol", "sharpe",
                 "max_drawdown", "first_live_date", "last_date"]].copy()
    for c in ["annualized_return", "annualized_vol", "sharpe", "max_drawdown"]:
        display[c] = display[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "n/a")
    display.columns = ["Fund", "Ann. return", "Ann. vol", "Sharpe",
                       "Max drawdown", "Live from", "To"]
    st.dataframe(display, width="stretch", hide_index=True)

    growth_wide = fund_returns().pivot(index="date", columns="fund", values="growth")
    crypto_cols = [c for c in growth_wide.columns if c.startswith("Crypto ")]
    other_cols = [c for c in growth_wide.columns if c not in crypto_cols]

    st.caption(
        "Split into two charts: Crypto funds swing far more widely "
        "than Equity/Combined funds, so plotting all 12 together on "
        "one axis compresses the latter into an unreadable clump."
    )
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Equity & Combined funds**")
        st.line_chart(growth_wide[other_cols], height=360)
    with col_right:
        st.markdown("**Crypto funds**")
        st.line_chart(growth_wide[crypto_cols], height=360)

    st.caption("Sharpe ratio by fund (out-of-sample, rf = 0):")
    sharpe = m.set_index("fund")["sharpe"].sort_values()
    st.bar_chart(sharpe)

    st.markdown("---")
    st.subheader("Fund fact sheet")
    fund = st.selectbox("Choose a fund", sorted(m["fund"].tolist()))
    meta = m[m["fund"] == fund].iloc[0]
    fr = fund_returns()
    f_one = fr[fr["fund"] == fund].set_index("date")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="nb-metric"><div class="k">Growth of $1</div>'
                f'<div class="v">${f_one["growth"].iloc[-1]:.2f}</div></div>',
                unsafe_allow_html=True)
    c2.markdown(f'<div class="nb-metric"><div class="k">Annualised return</div>'
                f'<div class="v">{_format_pct(meta["annualized_return"])}</div></div>',
                unsafe_allow_html=True)
    c3.markdown(f'<div class="nb-metric"><div class="k">Sharpe (rf = 0)</div>'
                f'<div class="v">{meta["sharpe"]:.2f}</div></div>',
                unsafe_allow_html=True)
    c4.markdown(f'<div class="nb-metric"><div class="k">Max drawdown</div>'
                f'<div class="v">{_format_pct(meta["max_drawdown"])}</div></div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Growth of $1**")
        st.line_chart(f_one["growth"])
        st.markdown("**Drawdown**")
        st.area_chart(f_one["drawdown"])
    with col_b:
        st.markdown("**Current holdings** (most recent rebalance)")
        holdings = _current_holdings(fund)
        st.bar_chart(holdings.set_index("ticker")["weight"])
        st.dataframe(holdings.style.format({"weight": "{:.2%}"}),
                     width="stretch", hide_index=True)
    st.caption(f"Out-of-sample backtest from {meta['first_live_date']} to "
               f"{meta['last_date']} · annualisation sqrt(252) "
               "(Crypto funds sqrt(365))")


# ============================================================ ALLOCATION
with tab_alloc:
    st.subheader("Allocate across funds")
    st.caption("Pick funds and set your split. The combined portfolio is the "
               "weighted mix of the selected funds' out-of-sample daily "
               "returns - a what-if over historical performance.")
    fr = fund_returns()
    funds = sorted(fr["fund"].unique())
    chosen = st.multiselect("Select funds", funds,
                            default=["Combined Maximum-Sharpe",
                                     "Combined Minimum-Variance",
                                     "Crypto Minimum-Variance"])
    if not chosen:
        st.info("Select at least one fund to build an allocation.")
    else:
        weights = {}
        for f in chosen:
            weights[f] = st.slider(f"{f} (share)", 0, 100, int(100 // len(chosen)),
                                   step=5)
        total = sum(weights.values())
        st.caption(f"Total: {total}%"
                   + ("" if total == 100 else " - normalised to 100%"))
        w = {f: v / total for f, v in weights.items()}

        wide = fr.pivot(index="date", columns="fund", values="daily_return")
        # Use only dates where EVERY selected fund has a valid return - an
        # investor can only hold/rebalance a blended portfolio on days all
        # its constituents actually trade (e.g. mixing a crypto fund, which
        # trades 7 days/week, with an equity fund restricts the blend to
        # equity trading days only).
        sub = wide[chosen].dropna(how="any")
        dropped = len(wide[chosen]) - len(sub)
        if dropped > 0:
            st.caption(
                f"Note: {dropped} date(s) excluded where not all selected "
                "funds had data (e.g. crypto-only weekend dates when mixing "
                "with equity/combined funds)."
            )
        if sub.empty:
            st.warning("Selected funds share no common trading dates.")
            st.stop()
        port = sum(w[f] * sub[f] for f in chosen)
        # Annualise using the ACTUAL observed frequency of this blend, not a
        # hardcoded 252 - a crypto-only or crypto-heavy blend trades on more
        # days/year than an equity-only one, and this adapts automatically.
        span_years = (port.index.max() - port.index.min()).days / 365.25
        periods_per_year = len(port) / span_years if span_years > 0 else 252
        met = _metrics_from_daily(port, periods_per_year)
        growth = (1.0 + port).cumprod()

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="nb-metric"><div class="k">Growth of $1</div>'
                    f'<div class="v">${growth.iloc[-1]:.2f}</div></div>',
                    unsafe_allow_html=True)
        c2.markdown(f'<div class="nb-metric"><div class="k">Annualised return</div>'
                    f'<div class="v">{_format_pct(met["Return"])}</div></div>',
                    unsafe_allow_html=True)
        c3.markdown(f'<div class="nb-metric"><div class="k">Sharpe</div>'
                    f'<div class="v">{met["Sharpe"]:.2f}</div></div>',
                    unsafe_allow_html=True)
        c4.markdown(f'<div class="nb-metric"><div class="k">Max drawdown</div>'
                    f'<div class="v">{_format_pct(met["Max DD"])}</div></div>',
                    unsafe_allow_html=True)

        st.line_chart(pd.DataFrame({"Portfolio": growth,
                                    "Combined Maximum-Sharpe":
                                        (1 + wide["Combined Maximum-Sharpe"]).cumprod()},
                                   index=growth.index))

        alloc = pd.DataFrame({"Fund": list(w), "Share": list(w.values())})
        st.dataframe(alloc.style.format({"Share": "{:.1%}"}), width="stretch",
                     hide_index=True)


# ============================================================ SENTIMENT
with tab_sent:
    st.subheader("Sector news-sentiment index")
    st.caption("Daily equal-weight headline sentiment per sector (VADER + a "
               "finance lexicon extension). Ticker-days with no headlines are "
               "neutral; the signal is only usable for the next trading day's "
               "decision.")
    sent = sector_sentiment()
    sectors = sorted(sent["sector"].unique())
    chosen_s = st.multiselect("Sectors", sectors, default=sectors)
    smooth_days = st.slider(
        "Smoothing (rolling average, days)",
        min_value=1,
        max_value=30,
        value=21,
        step=1,
        help="Raw daily headline sentiment is noisy (few headlines per "
        "ticker-day); a rolling average reveals the underlying trend. "
        "Set to 1 for the unsmoothed daily signal.",
    )
    if chosen_s:
        wide = sent[sent["sector"].isin(chosen_s)].pivot(
            index="date", columns="sector", values="sentiment"
        )
        if smooth_days > 1:
            wide = wide.rolling(smooth_days, min_periods=1).mean()
        st.line_chart(wide, height=400)
        st.caption(
            f"{smooth_days}-day rolling average of daily equal-weight sector sentiment."
            if smooth_days > 1
            else "Raw daily sector sentiment (unsmoothed)."
        )
    else:
        st.info("Select at least one sector.")

    st.markdown("**Sentiment fusion** - base vs sentiment-tilted funds "
                "(out-of-sample):")
    st.caption("Equity weights are tilted toward names with strong recent "
               "headline sentiment; the equity/crypto split is preserved.")
    fuse = fusion_returns().pivot(index="date", columns="fund", values="growth")
    st.line_chart(fuse, height=360)


# ============================================================ DATA
with tab_data:
    st.subheader("Data inventory")
    st.caption("Raw data loads from the hosted bundle via src/data_access.py "
               "(cached); derived artifacts live in results/.")
    eq = data_access.load_equity_prices()
    cr = data_access.load_crypto_prices()
    news = data_access.load_news_headlines()
    inv = pd.DataFrame([
        {"dataset": "equity_prices", "rows": len(eq),
         "tickers": eq["ticker"].nunique(),
         "span": f"{eq['date'].min().date()} to {eq['date'].max().date()}"},
        {"dataset": "crypto_prices", "rows": len(cr),
         "tickers": cr["ticker"].nunique(),
         "span": f"{cr['date'].min().date()} to {cr['date'].max().date()}"},
        {"dataset": "news_headlines", "rows": len(news),
         "tickers": news["ticker"].nunique(),
         "span": f"{news['date'].min().date()} to {news['date'].max().date()}"},
    ])
    st.dataframe(inv, width="stretch", hide_index=True)
    with st.expander("Preview equity prices"):
        st.dataframe(eq.head(10), width="stretch")
    with st.expander("Preview news headlines"):
        st.dataframe(news[["date", "ticker", "sector", "title"]].head(10),
                     width="stretch")

st.markdown('<div class="nb-foot">NoBearTrader · FINS3645 project · all '
            'performance is out-of-sample on 2020-2023 data; past performance '
            'does not guarantee future results.</div>', unsafe_allow_html=True)
