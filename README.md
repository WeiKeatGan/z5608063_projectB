# NoBearTrader - FinTech Project Part B

Systematically managed multi-asset funds with news-sentiment analytics
(FINS3645, DFF Stations 3-4). This folder is also the public GitHub repository;
the app entrypoint is `streamlit_app.py` at the root.

## What is here

- **Funds** (`src/portfolios.py`): 12 walk-forward out-of-sample funds across
  three asset families (Equity, Crypto, Combined) x four methods
  (minimum-variance, maximum-Sharpe, risk parity, equal weight). No look-ahead:
  weights at rebalance date t use data up to t only, held from t+1. Monthly
  rebalancing; annualisation sqrt(252) equity/combined, sqrt(365) crypto;
  risk-free rate = 0 (stated choice).
- **Sentiment** (`src/sentiment.py` + `src/finance_lexicon.py`): VADER extended
  with a ~200-term finance lexicon, aggregated to a daily equal-weight sector
  index. Ticker-days with no headlines are neutral; the signal is lagged one
  trading day before it can drive a trade.
- **Fusion** (`src/fusion.py`): tilts equity weights toward names with strong
  recent headline sentiment (equity/crypto split preserved) and reports the
  before-vs-after effect.
- **App** (`streamlit_app.py`): compare funds, read each fund's fact sheet
  (growth of $1, drawdown, Sharpe, current holdings), set an allocation across
  funds, and explore the sector sentiment index. The app reads precomputed
  results/ artifacts - it never runs VADER or recomputes backtests.

## How to run

    pip install -r requirements.txt -r requirements-dev.txt   # dev adds nltk (VADER)
    python scripts/run_part_b.py            # reproduces all results into results/
    python -m pytest -q                     # unit tests (backtest no-look-ahead, metrics, sentiment, fusion)
    streamlit run streamlit_app.py          # runs the app locally
    python scripts/check_handin.py          # pre-hand-in checks

Raw data loads through `src/data_access.py` (see `context/DATA_GUIDE.md`); raw
data is never committed. The app's precomputed artifacts ARE committed under
`results/data/` (the deployed app reads them), so keep nltk out of
`requirements.txt`.

## Folder layout

- `streamlit_app.py`   the app entrypoint (repo root)
- `.streamlit/`        app config
- `src/`               code: etl, features (reused from Part A), portfolios,
  sentiment, finance_lexicon, fusion, exhibits, style
- `scripts/`           runnable scripts that reproduce your results
- `results/`           outputs: figures in `results/figures/`, tables in
  `results/tables/`, app data artifacts in `results/data/`
- `context/`           provided data guide and project context (do not edit)
- `report/`            your report - see `report/OUTLINE.md` (author in Word,
  submit `report.pdf`)
- `ai/`                your prompt logs and AI notes
- `tests/`             unit tests
- `AGENTS.md`          my own agent instructions (graded)

## Deploy + hand in

This folder is its own GitHub repo, independent of fins-agent. Commit your code
AND your precomputed artifacts under `results/` (the app reads them), then:

    python scripts/check_handin.py
    # git init in this folder, push the contents to a NEW private GitHub repo

Then connect the repo on share.streamlit.io (entrypoint `streamlit_app.py`). At
hand-in, make the repo PUBLIC and submit the live URL + repo link, and also zip
this whole folder for Moodle. See `PROJECT_BRIEF.md` Appendix D and
`docs/STUDENT_DEPLOY.md`.
