# AGENTS.md — NoBearTrader (FINS3645 Project B)

## Project context
This is Part B (Funds, Sentiment & App, Stations 3-4) of my FinTech systematic-fund
project. Read PROJECT_BRIEF.md before making changes. I reuse my own Part A
foundation (ETL + features, src/etl.py and src/features.py) — the data guide is the
same. The deliverable: out-of-sample funds across equity/crypto/combined families,
a sentiment sector index, a sentiment fusion extension, a Streamlit app, and the
report. Do NOT implement anything past the dataset: results stop at 2023-12-31.

## Environment
- Mac (PyCharm CE), Python 3.13 via repo-local .venv
- From inside this project folder, run everything with `../../.venv/bin/python`
- Run scripts from project root: `python scripts/run_part_b.py`
- Test: `python -m pytest -q` (or from repo root `./.venv/bin/python -m pytest -q fins2026/z5608063_projectB/tests`)

## Coding conventions
- Figures: use plt.savefig() + plt.close(), never plt.show()
- Figure style: FT-style — #FFF1E5 background, #0F5499 blue / #E6007E pink,
  sentence-case titles, source + sample-period footer (src/style.py). Reuse it in
  the app so figures and the app share one design system
- Before writing code that touches a DataFrame's columns, print(df.columns.tolist())
  first to confirm actual column names (suffixes matter: combined weights are
  TICKER_EQ / TICKER_CR)
- Never commit raw data or .parquet files — only derived artifacts under results/
- The deployed app must NOT import nltk and must NOT recompute backtests: it reads
  precomputed results/data/ CSVs. Keep nltk in requirements-dev.txt only

## Backtest rules (non-negotiable)
- Walk-forward, out-of-sample, NO look-ahead: weights at rebalance date t use
  returns up to and including t only, and are first applied to t+1
- Rebalance monthly (first trading day of each month) or less often
- Long-only weights, sum to 1. State constraints in the report
- Annualise with sqrt(252) for equity/combined funds, sqrt(365) for crypto-only
  (crypto backtest runs on crypto's own 365-day calendar)
- Risk-free rate = 0 for the Sharpe ratio (a stated choice)
- Sentiment signal lagged at least one trading day: headline aligned to day t is
  first usable for day t+1's decision

## Known pitfalls (learned on this project - see ai/ prompt logs)
- scipy optimisers on daily-return covariances stall silently (the brief warns
  about this). Risk parity is solved as true equal-risk-contribution via
  scipy.optimize.root with a log-parametrisation, NOT a loss-minimisation that
  sat at the equal-weight start. Always sanity-check that weights differ across
  methods.
- Fusion tilt needs the sentiment grid aligned to the fund's weight columns:
  sentiment uses plain tickers ("WMT") but combined-fund weights are "WMT_EQ".
  Build the aligned grid per family before fusing.
- pandas quirk: df[numpy_bool] selects ROWS; use df.loc[:, numpy_bool] to select
  columns (this crashed plot_weights).
- VADER needs a one-time nltk.download('vader_lexicon') — a build step in
  run_part_b.py, never in the app.
- Keep the raw headline text for VADER (casing, punctuation, stopwords).

## Workflow
- Build and verify one stage at a time; paste terminal output back after every run
- When fixing a bug, explain what was wrong and why the fix works
- Run scripts/run_part_b.py end-to-end and confirm the required outputs exist:
  results/data/fund_returns.csv, fund_weights.csv, sector_sentiment_index.csv,
  results/tables/performance_metrics.csv
- Run python scripts/check_handin.py before zipping/deploying

## Report-writing rules (report/report.docx -> report.pdf)
- I write my own analysis and interpretation; AI must never write finished report
  prose for direct submission (AI-generated reasoning submitted as my own is
  penalised)
- AI's role: ask pointed questions that make me articulate my reasoning, critique
  my drafts, flag factual inaccuracies (e.g. claiming a method the code doesn't
  implement), flag overclaims unsupported by the statistics, and flag phrasing too
  close to the brief's/rubric's wording
- Every number I cite must trace to a validated output in results/ — never invent
  or approximate a figure
- Required Part B exhibits: performance-metrics table, growth-of-$1, drawdown,
  weights-over-time, Sharpe barplot, sector sentiment-index time series, fusion
  before-vs-after table and figure
