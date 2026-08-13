"""Part B required exhibits: fund comparison figures and the sentiment chart.

Every figure is self-contained (title, labelled axes, source + sample-period
footer) using the NoBearTrader design system from src.style.
"""
import numpy as np
import pandas as pd

from src.style import (new_ft_figure, apply_ft_style, savefig_close,
                       FT_PALETTE_LONG, FT_BLUE, FT_PINK, FT_BG)


def plot_growth(fund_returns: pd.DataFrame, out_path) -> None:
    """Growth of $1 for every fund (out-of-sample), split into two panels:
    Crypto funds (wide scale swings) vs Equity/Combined funds (tighter
    range) - plotting all 12 on one axis compresses the latter into an
    unreadable clump beneath the crypto funds' much larger growth."""
    import matplotlib.pyplot as _plt
    import matplotlib.dates as mdates
    from src.style import FT_BG

    wide = fund_returns.pivot(index="date", columns="fund", values="growth")
    crypto_cols = [c for c in wide.columns if c.startswith("Crypto ")]
    other_cols = [c for c in wide.columns if c not in crypto_cols]

    fig, (ax1, ax2) = _plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor(FT_BG)

    for i, col in enumerate(other_cols):
        ax1.plot(wide.index, wide[col], label=col, linewidth=1.4,
                 color=FT_PALETTE_LONG[i % len(FT_PALETTE_LONG)])
    ax1.set_facecolor(FT_BG)
    apply_ft_style(ax1, "Growth of $1 - Equity & Combined funds", ylabel="Growth of $1")
    ax1.legend(fontsize=7, frameon=False, ncol=1, loc="upper left")

    for i, col in enumerate(crypto_cols):
        ax2.plot(wide.index, wide[col], label=col, linewidth=1.4,
                 color=FT_PALETTE_LONG[i % len(FT_PALETTE_LONG)])
    ax2.set_facecolor(FT_BG)
    apply_ft_style(ax2, "Growth of $1 - Crypto funds", ylabel="Growth of $1")
    ax2.legend(fontsize=7, frameon=False, ncol=1, loc="upper left")

    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    savefig_close(fig, out_path)


def plot_drawdown(fund_returns: pd.DataFrame, fund: str, out_path) -> None:
    """Drawdown of one fund as a filled area below zero."""
    dd = fund_returns[fund_returns["fund"] == fund].set_index("date")["drawdown"]
    fig, ax = new_ft_figure(figsize=(12, 5))
    ax.fill_between(dd.index, dd.to_numpy(), 0, color=FT_PINK, alpha=0.45)
    ax.plot(dd.index, dd.to_numpy(), color=FT_PINK, linewidth=1.0)
    apply_ft_style(ax, f"Drawdown - {fund}", ylabel="Drawdown")
    savefig_close(fig, out_path)


def plot_weights(fund_weights: pd.DataFrame, fund: str, out_path, top_n: int = 10) -> None:
    """Stacked area of the largest weight positions over time for one fund."""
    w = (fund_weights[fund_weights["fund"] == fund]
         .pivot(index="date", columns="ticker", values="weight").fillna(0.0))
    means = w.mean().sort_values(ascending=False)
    top = means.head(top_n).index
    data = w[top].copy()
    data["Other"] = w.loc[:, ~w.columns.isin(top)].sum(axis=1)
    colors = (FT_PALETTE_LONG * 2)[: len(data.columns)]

    fig, ax = new_ft_figure(figsize=(12, 6.5))
    ax.stackplot(data.index, *[data[c].to_numpy() for c in data.columns],
                 labels=data.columns, colors=colors, alpha=0.9)
    ax.set_ylim(0, 1)
    apply_ft_style(ax, f"Portfolio weights over time - {fund}", ylabel="Weight")
    ax.legend(fontsize=7,ncol=2,loc="upper left",frameon=True,facecolor=FT_BG,edgecolor="#33302E",framealpha=0.92,
    )
    savefig_close(fig, out_path)


def plot_sharpe(metrics: pd.DataFrame, out_path) -> None:
    """Horizontal barplot of the out-of-sample Sharpe ratio by fund."""
    m = metrics.sort_values("sharpe").reset_index(drop=True)
    fig, ax = new_ft_figure(figsize=(10, 7))
    colors = [FT_PINK if v < 0 else FT_BLUE for v in m["sharpe"]]
    ax.barh(m["fund"], m["sharpe"], color=colors)
    ax.axvline(0, color="#33302E", linewidth=0.8)
    apply_ft_style(ax, "Sharpe ratio by fund (out-of-sample, rf = 0)",
                   xlabel="Sharpe ratio")
    savefig_close(fig, out_path)


def plot_sector_sentiment_lines(sector_index: pd.DataFrame, out_path) -> None:
    """Sector sentiment index over time, one line per sector."""
    fig, ax = new_ft_figure(figsize=(12, 6.5))
    for i, col in enumerate(sector_index.columns):
        ax.plot(sector_index.index, sector_index[col], label=col, linewidth=1.0,
                color=FT_PALETTE_LONG[i % len(FT_PALETTE_LONG)])
    ax.axhline(0, color="#33302E", linewidth=0.8, linestyle="--")
    apply_ft_style(ax, "Sector sentiment index (daily, headline-aligned)",
                   ylabel="Sentiment (VADER compound, sector equal-weight)")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    savefig_close(fig, out_path)


def plot_sector_sentiment_heatmap(sector_index: pd.DataFrame, out_path,
                                  smooth: int = 21) -> None:
    """Heatmap of the sector sentiment index (smoothed) over time."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as _plt

    s = sector_index.rolling(smooth, min_periods=1).mean()
    fig, ax = _plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor(FT_BG)
    ax.set_facecolor(FT_BG)
    x0, x1 = mdates.date2num(s.index.min()), mdates.date2num(s.index.max())
    im = ax.imshow(s.to_numpy().T, aspect="auto", interpolation="nearest",
                   cmap="RdYlGn", extent=[x0, x1, 0, s.shape[1]],
                   vmin=-0.25, vmax=0.25)
    ax.set_yticks(np.arange(s.shape[1]) + 0.5, labels=s.columns)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.colorbar(im, ax=ax, label="Sentiment (VADER compound)")
    apply_ft_style(ax, f"Sector sentiment index ({smooth}-day average)")
    savefig_close(fig, out_path)


def plot_fusion(base_growth: pd.Series, fused_growth: pd.Series,
                fund_label: str, out_path) -> None:
    """Base vs sentiment-fused growth of $1 for one fund."""
    fig, ax = new_ft_figure(figsize=(12, 6))
    ax.plot(base_growth.index, base_growth.to_numpy(),
            label=f"{fund_label} (base)", color=FT_BLUE, linewidth=1.5)
    ax.plot(fused_growth.index, fused_growth.to_numpy(),
            label=f"{fund_label} (sentiment-fused)", color=FT_PINK, linewidth=1.5)
    apply_ft_style(ax, "Base vs sentiment-fused fund (out-of-sample)",
                   ylabel="Growth of $1")
    ax.legend(frameon=False)
    savefig_close(fig, out_path)
