"""Reproduce your Part B results. Run from the project root:

    python scripts/run_part_b.py

Stages: reuse the Part A foundation -> out-of-sample funds across
(equity, crypto, combined) x (min-variance, max-sharpe, risk parity, equal
weight) -> sentiment sector index (VADER + finance lexicon) -> sentiment
fusion extension. Writes app-readable CSVs to results/data/, report tables to
results/tables/, and figures to results/figures/.
"""
from __future__ import annotations

import sys
import pathlib
import json

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import etl, features, portfolios, sentiment, fusion, exhibits  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "results" / "data"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"

FAMILY_WINDOW = {"Combined": 252, "Equity": 252, "Crypto": 365}
FAMILY_PERIODS = {"Combined": 252, "Equity": 252, "Crypto": 365}
METHOD_LABELS = {
    "min_variance": "Minimum-Variance",
    "max_sharpe": "Maximum-Sharpe",
    "risk_parity": "Risk Parity",
    "equal_weight": "Equal Weight",
}
FUSION_STRENGTH = 0.25
FUSION_LOOKBACK = 5


def main() -> None:
    for d in (DATA_DIR, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("STATION 1-2 (reused from Part A) - clean data, returns, text panel")
    print("=" * 64)
    equities, eq_report = etl.load_clean_equities()
    crypto, cr_report = etl.load_clean_crypto()
    headlines, news_report = etl.load_clean_headlines(equities)
    combined, merge_report = etl.merge_panels(equities, crypto)
    eq_returns_long = features.daily_returns(equities)
    cr_returns_long = features.daily_returns(crypto)
    text_panel = features.assemble_headline_panel(headlines)
    print(f"  combined panel: {combined.shape} "
          f"({merge_report['calendar_days']} equity days, {merge_report['combined_columns']} assets)")
    print(f"  text panel: {text_panel.shape} ticker-days")

    eq_wide = combined[[c for c in combined.columns if c.endswith("_EQ")]].copy()
    eq_wide.columns = [c[:-3] for c in eq_wide.columns]
    cr_wide = combined[[c for c in combined.columns if c.endswith("_CR")]].copy()
    cr_wide.columns = [c[:-3] for c in cr_wide.columns]
    cr_wide_full = (cr_returns_long.pivot(index="date", columns="ticker",
                                          values="daily_return").sort_index())
    panels = {"Combined": combined, "Equity": eq_wide, "Crypto": cr_wide_full}

    print("\n" + "=" * 64)
    print("STATION 3a - walk-forward out-of-sample funds (no look-ahead)")
    print("=" * 64)
    fund_returns_rows, fund_weights_rows, metrics_rows = [], [], []
    backtests: dict[str, dict] = {}
    for family, panel in panels.items():
        window = FAMILY_WINDOW[family]
        for method, method_label in METHOD_LABELS.items():
            fund = f"{family} {method_label}"
            print(f"  backtesting {fund} (window={window}d, monthly rebalance)...")
            bt = portfolios.oos_backtest(panel, method=method, window=window,
                                         rebalance="monthly")
            backtests[fund] = bt
            m = portfolios.performance_metrics(bt["returns"],
                                               periods_per_year=FAMILY_PERIODS[family])
            growth = bt["growth"]
            dd = bt["drawdown"]
            fund_returns_rows.append(pd.DataFrame({
                "date": growth.index, "fund": fund,
                "daily_return": bt["returns"].to_numpy(),
                "growth": growth.to_numpy(), "drawdown": dd.to_numpy(),
            }))
            w = bt["weights"]
            weight_long = w.stack().reset_index()
            weight_long.columns = ["date", "ticker", "weight"]
            weight_long["fund"] = fund
            fund_weights_rows.append(weight_long)
            metrics_rows.append({
                "fund": fund, "family": family, "method": method,
                "annualized_return": m["annualized_return"],
                "annualized_vol": m["annualized_vol"],
                "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
                "periods_per_year": FAMILY_PERIODS[family],
                "first_live_date": bt["first_live_date"].strftime("%Y-%m-%d"),
                "last_date": bt["last_date"].strftime("%Y-%m-%d"),
                "n_days": m["n_days"], "n_rebalances": len(bt["rebalance_dates"]),
            })

    fund_returns = pd.concat(fund_returns_rows, ignore_index=True)
    fund_weights = pd.concat(fund_weights_rows, ignore_index=True)
    metrics = pd.DataFrame(metrics_rows)

    fund_returns.to_csv(DATA_DIR / "fund_returns.csv", index=False)
    fund_weights.to_csv(DATA_DIR / "fund_weights.csv", index=False)
    metrics.to_csv(TABLES_DIR / "performance_metrics.csv", index=False)
    print(f"  saved {DATA_DIR / 'fund_returns.csv'} "
          f"({fund_returns['fund'].nunique()} funds)")
    print(f"  saved {DATA_DIR / 'fund_weights.csv'}")
    print(f"  saved {TABLES_DIR / 'performance_metrics.csv'}")
    print("\n  Out-of-sample performance (rf = 0):")
    show = metrics[["fund", "annualized_return", "annualized_vol",
                    "sharpe", "max_drawdown"]].copy()
    for c in ["annualized_return", "annualized_vol", "sharpe", "max_drawdown"]:
        show[c] = show[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "n/a")
    print(show.to_string(index=False))

    # sanity check the brief's solver warning: weights must differ across methods
    first_rebal = backtests["Combined Minimum-Variance"]["weights"].iloc[0]
    print("\n  First-rebalance combined weights (top-5 per method) - sanity check")
    for method, label in METHOD_LABELS.items():
        fund = f"Combined {label}"
        top = backtests[fund]["weights"].iloc[0].sort_values(ascending=False).head(5)
        print(f"    {label:16s} " + ", ".join(f"{k}={v:.3f}" for k, v in top.items()))

    print("\n" + "=" * 64)
    print("STATION 3b - sentiment model + sector index (VADER + finance lexicon)")
    print("=" * 64)
    scores = sentiment.score_headlines(text_panel)
    print(f"  scored {len(scores)} ticker-days ({scores['vader_mean'].notna().sum()} with news)")
    sector_index = sentiment.sector_sentiment_index(scores, no_news="neutral")
    sector_long = sector_index.reset_index().melt(id_vars="date", var_name="sector",
                                                  value_name="sentiment")
    sector_long.to_csv(DATA_DIR / "sector_sentiment_index.csv", index=False)
    scores.to_csv(DATA_DIR / "ticker_sentiment.csv", index=False)
    print(f"  saved {DATA_DIR / 'sector_sentiment_index.csv'} "
          f"({sector_long['sector'].nunique()} sectors)")
    print(f"  saved {DATA_DIR / 'ticker_sentiment.csv'}")
    exhibits.plot_sector_sentiment_lines(sector_index, FIGURES_DIR / "sector_sentiment_index.png")
    exhibits.plot_sector_sentiment_heatmap(sector_index, FIGURES_DIR / "sector_sentiment_heatmap.png")
    print(f"  saved {FIGURES_DIR / 'sector_sentiment_index.png'} and heatmap")

    print("\n" + "=" * 64)
    print("STATION 3c - fusion: sentiment tilt into the equity funds")
    print("=" * 64)
    equity_tickers = list(eq_report["tickers"])
    # Combined-fund weight columns carry _EQ/_CR suffixes; sentiment scores use
    # plain tickers. Align the sentiment grid to each family's weight columns so
    # the tilt targets the right sleeve.
    scores_by_family = {"Equity": scores}
    combined_scores = scores.copy()
    combined_scores["ticker"] = combined_scores["ticker"] + "_EQ"
    scores_by_family["Combined"] = combined_scores
    equity_cols = {
        "Combined": [f"{t}_EQ" for t in equity_tickers],
        "Equity": equity_tickers,
        "Crypto": [],
    }
    fusion_rows, fusion_returns_rows = [], []
    for fund in ["Combined Maximum-Sharpe", "Combined Minimum-Variance",
                 "Equity Maximum-Sharpe"]:
        family = fund.split()[0]
        panel = panels[family]
        base_bt = backtests[fund]
        fused_weights = fusion.apply_sentiment(
            base_bt["weights"], scores_by_family[family], equity_cols[family],
            strength=FUSION_STRENGTH, lookback=FUSION_LOOKBACK)
        fused_port, _ = portfolios.evaluate_weights(panel, fused_weights)
        base_m = portfolios.performance_metrics(base_bt["returns"],
                                                periods_per_year=FAMILY_PERIODS[family])
        fused_m = portfolios.performance_metrics(fused_port,
                                                 periods_per_year=FAMILY_PERIODS[family])
        fused_growth = (1.0 + fused_port).cumprod()
        for tag, m in (("base", base_m), ("fused", fused_m)):
            fusion_rows.append({
                "fund": fund, "variant": tag,
                "annualized_return": m["annualized_return"],
                "annualized_vol": m["annualized_vol"], "sharpe": m["sharpe"],
                "max_drawdown": m["max_drawdown"],
            })
        fusion_returns_rows.append(pd.DataFrame({
            "date": base_bt["growth"].index, "fund": f"{fund} (base)",
            "growth": base_bt["growth"].to_numpy()}))
        fusion_returns_rows.append(pd.DataFrame({
            "date": fused_growth.index, "fund": f"{fund} (sentiment-fused)",
            "growth": fused_growth.to_numpy()}))
        exhibits.plot_fusion(base_bt["growth"], fused_growth, fund,
                             FIGURES_DIR / f"fusion_{fund.lower().replace(' ', '_').replace('-', '_')}.png")
        print(f"  {fund}: base Sharpe={base_m['sharpe']:.3f} "
              f"vs fused Sharpe={fused_m['sharpe']:.3f}")

    fusion_table = pd.DataFrame(fusion_rows)
    fusion_table.to_csv(TABLES_DIR / "fusion_before_after.csv", index=False)
    pd.concat(fusion_returns_rows, ignore_index=True).to_csv(
        DATA_DIR / "fusion_returns.csv", index=False)
    print(f"  saved {TABLES_DIR / 'fusion_before_after.csv'} "
          f"and {DATA_DIR / 'fusion_returns.csv'}")
    print(fusion_table.to_string(index=False))
    print("\n" + "=" * 64)
    print("STATION 3c (robustness) - fusion sensitivity grid, Combined Min-Variance")
    print("=" * 64)
    grid = fusion.sensitivity_grid(
        backtests["Combined Minimum-Variance"],
        panels["Combined"],
        scores_by_family["Combined"],
        equity_cols["Combined"],
        periods_per_year=FAMILY_PERIODS["Combined"],
    )
    grid.to_csv(TABLES_DIR / "fusion_sensitivity.csv", index=False)
    print(grid.to_string(index=False))
    print(f"  saved {TABLES_DIR / 'fusion_sensitivity.csv'}")
    print(
        f"  (reported headline config: strength={FUSION_STRENGTH}, lookback={FUSION_LOOKBACK}"
        f" — fixed defaults, NOT selected from this grid)"
    )
    print("\n" + "=" * 64)
    print("STATION 3c (robustness) - transaction-cost stress test (50bps/trade)")
    print("=" * 64)
    COST_BPS = 50.0
    cost_rows = []
    for fund in ["Combined Maximum-Sharpe", "Combined Minimum-Variance", "Equity Maximum-Sharpe"]:
        family = fund.split()[0]
        base_bt = backtests[fund]
        base_net, base_to = portfolios.apply_transaction_costs(
            base_bt["returns"], base_bt["weights"], cost_bps=COST_BPS
        )
        base_net_m = portfolios.performance_metrics(
            base_net, periods_per_year=FAMILY_PERIODS[family]
        )

        fused_weights = fusion.apply_sentiment(
            base_bt["weights"],
            scores_by_family[family],
            equity_cols[family],
            strength=FUSION_STRENGTH,
            lookback=FUSION_LOOKBACK,
        )
        fused_port, _ = portfolios.evaluate_weights(panels[family], fused_weights)
        fused_net, fused_to = portfolios.apply_transaction_costs(
            fused_port, fused_weights, cost_bps=COST_BPS
        )
        fused_net_m = portfolios.performance_metrics(
            fused_net, periods_per_year=FAMILY_PERIODS[family]
        )

        cost_rows.append(
            {
                "fund": fund,
                "base_sharpe_gross": portfolios.performance_metrics(
                    base_bt["returns"], periods_per_year=FAMILY_PERIODS[family]
                )["sharpe"],
                "base_sharpe_net": base_net_m["sharpe"],
                "base_avg_turnover": base_to["turnover"].mean(),
                "fused_sharpe_gross": portfolios.performance_metrics(
                    fused_port, periods_per_year=FAMILY_PERIODS[family]
                )["sharpe"],
                "fused_sharpe_net": fused_net_m["sharpe"],
                "fused_avg_turnover": fused_to["turnover"].mean(),
            }
        )
    cost_table = pd.DataFrame(cost_rows)
    cost_table.to_csv(TABLES_DIR / "fusion_cost_stress.csv", index=False)
    print(cost_table.to_string(index=False))
    print(f"  saved {TABLES_DIR / 'fusion_cost_stress.csv'}")
    print("\n" + "=" * 64)
    print("STATION 3 exhibits - comparison figures")
    print("=" * 64)
    exhibits.plot_growth(fund_returns, FIGURES_DIR / "growth_of_dollar_all_funds.png")
    exhibits.plot_drawdown(fund_returns, "Combined Minimum-Variance",
                           FIGURES_DIR / "drawdown_combined_min_variance.png")
    exhibits.plot_weights(fund_weights, "Combined Risk Parity",
                          FIGURES_DIR / "weights_combined_risk_parity.png")
    exhibits.plot_sharpe(metrics, FIGURES_DIR / "sharpe_by_fund.png")
    print("  saved growth, drawdown, weights, and sharpe figures")

    summary = {
        "product": "NoBearTrader",
        "n_funds": int(fund_returns["fund"].nunique()),
        "families": list(FAMILY_WINDOW),
        "methods": list(METHOD_LABELS.values()),
        "equity_days": int(merge_report["calendar_days"]),
        "first_live_date": metrics["first_live_date"].min(),
        "last_date": metrics["last_date"].max(),
        "annualisation": {f: f"sqrt({p})" for f, p in FAMILY_PERIODS.items()},
        "risk_free_rate": 0.0,
        "rebalance": "monthly (first trading day of the month)",
        "transaction_costs": 0.0,
        "sentiment": {
            "model": "VADER + finance lexicon extension",
            "no_news_treatment": "neutral (0)",
            "lag": "signal first usable the trading day after the aligned day",
            "ticker_days_scored": int(len(scores)),
        },
        "fusion": {
            "rule": "equity-weight tilt proportional to recent sector sentiment",
            "strength": FUSION_STRENGTH,
            "lookback_days": FUSION_LOOKBACK,
        },
    }
    lexicon = sentiment._get_analyzer()
    summary["sentiment"]["lexicon_terms_added"] = len(
        set(sentiment.FINANCE_LEXICON) & set(lexicon.lexicon))
    with open(TABLES_DIR / "part_b_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  saved {TABLES_DIR / 'part_b_summary.json'}")

    print("\n" + "=" * 64)
    print("Station 3 complete. Results saved under results/.")
    print("=" * 64)


if __name__ == "__main__":
    main()
