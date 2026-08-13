"""Station 3 - your funds: optimal portfolios + walk-forward out-of-sample backtest.

Fund families (equity, crypto, combined) x methods (minimum-variance,
maximum-Sharpe, risk parity, equal weight). Every fund is a walk-forward
out-of-sample backtest with NO look-ahead: weights at rebalance date t use
returns up to and including t only, and are first applied to the return on
date t+1. Rebalance monthly (first trading day of each month). Annualise with
the right calendar - 252 for equity/combined funds, 365 for crypto.

Solver robustness: the brief warns that optimisers on tiny daily-return
covariances can silently stall. We scale the return moments so the objective is
O(1) and add a small ridge to the covariance. Weights are long-only and sum to 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, root

METHODS = ("min_variance", "max_sharpe", "risk_parity", "equal_weight")


# ---------------------------------------------------------------------------
# Optimisation primitives (weights from a past returns window)
# ---------------------------------------------------------------------------

def _ridge_cov(mat: np.ndarray) -> np.ndarray:
    """Sample covariance plus a tiny ridge so the covariance stays
    positive-definite when the window is short relative to the asset count."""
    cov = np.cov(mat, rowvar=False, ddof=1)
    ridge = 1e-8 * np.trace(cov) / cov.shape[0]
    return cov + ridge * np.eye(cov.shape[0])


def _min_variance_weights(cov: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(cov)
    ones = np.ones(cov.shape[0])
    w = inv @ ones
    return w / w.sum()


def _max_sharpe_weights(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Long-only tangency portfolio (max Sharpe with risk-free rate 0)."""
    n = mu.size
    mu_s = mu * 100.0          # scale so the solver sees O(1) numbers;
    cov_s = cov * 1e4          # the Sharpe argmax is invariant to scaling

    def neg_sharpe(w):
        vol = np.sqrt(max(float(w @ cov_s @ w), 0.0))
        if vol <= 0:
            return 1e6
        return -float(w @ mu_s) / vol

    cons = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds = [(0.0, 1.0)] * n
    res = minimize(neg_sharpe, np.full(n, 1.0 / n), method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"ftol": 1e-10, "maxiter": 2000, "disp": False})
    w = np.clip(res.x, 0.0, 1.0)
    return w / w.sum()


def _risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    """Equal risk contribution (ERC) portfolio.

    Solves w_i (S w)_i = c for all i - every asset contributes the same
    amount of portfolio risk - using the log-parametrisation w = softmax(x),
    which keeps weights positive and summing to 1. The earlier loss-minimising
    formulation stalled at the equal-weight start on the 60-asset combined
    panel (the brief's solver-stall warning), so we solve the ERC system
    directly and fall back to inverse-volatility weighting if it fails.
    """
    n = cov.shape[0]

    def weights_from_x(x):
        y = np.append(x, 0.0)
        y = y - np.max(y)
        w = np.exp(y)
        return w / w.sum()

    def system(x):
        w = weights_from_x(x)
        rc = w * (cov @ w)
        return rc[:-1] - rc[:-1].mean()

    sol = root(system, np.zeros(n - 1), method="hybr")
    if sol.success:
        w = weights_from_x(sol.x)
    else:
        inv_vol = 1.0 / np.sqrt(np.maximum(np.diag(cov), 1e-12))
        w = inv_vol / inv_vol.sum()
    return w


def _equal_weight_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def compute_weights(returns: pd.DataFrame, method: str = "min_variance") -> pd.Series:
    """Optimal weights from a past returns window, over the FULL universe.

    Assets with no usable data in the window (more than 5% missing, or zero
    variance) get weight 0; the rest are optimised and the weights sum to 1.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
    x = returns.astype(float)
    usable = x.columns[x.isna().mean() <= 0.05]
    if len(usable) < 2:
        raise ValueError("fewer than 2 assets have usable data in the window")
    x = x[usable].dropna(how="any")
    if len(x) < 2:
        raise ValueError("fewer than 2 complete rows in the window")
    mat = x.to_numpy()
    std = mat.std(axis=0)
    keep = np.where(std > 0)[0]
    if keep.size < 2:
        raise ValueError("fewer than 2 assets have nonzero variance in the window")
    mat = mat[:, keep]
    cols_keep = [usable[i] for i in keep]

    cov = _ridge_cov(mat)
    if method == "min_variance":
        w = _min_variance_weights(cov)
    elif method == "max_sharpe":
        w = _max_sharpe_weights(mat.mean(axis=0), cov)
    elif method == "risk_parity":
        w = _risk_parity_weights(cov)
    else:
        w = _equal_weight_weights(mat.shape[1])

    full = pd.Series(0.0, index=returns.columns)
    full.loc[cols_keep] = w
    return full


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def _rebalance_dates(dates: pd.DatetimeIndex, start_idx: int,
                     mode: str, every: int) -> list:
    """Rebalance dates inside the out-of-sample region. 'monthly' = the first
    trading day of each calendar month; 'every_n' = every `every` trading days."""
    if mode == "every_n":
        return [dates[i] for i in range(start_idx, len(dates), every)]
    if mode != "monthly":
        raise ValueError(f"unknown rebalance mode {mode!r}")
    out, seen = [], set()
    for d in dates[start_idx:]:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def evaluate_weights(returns: pd.DataFrame, weights: pd.DataFrame):
    """Apply a weight schedule (index = rebalance dates) to a returns panel.

    Weights formed at rebalance date t are held from t+1 until the next
    rebalance, so the return realised ON t is never attributed to the new
    weights (no look-ahead). Returns (portfolio_daily_return, active_weights).
    """
    returns = returns.astype(float)
    active = weights.reindex(returns.index).ffill().shift(1)
    valid = active.notna().all(axis=1)
    active = active.loc[valid]
    port = (active * returns.loc[active.index]).sum(axis=1, min_count=1)
    return port.loc[port.notna()], active


def oos_backtest(returns: pd.DataFrame, method: str = "min_variance",
                 window: int = 252, rebalance: str = "monthly",
                 every: int = 21) -> dict:
    """Walk-forward out-of-sample backtest for one (method, fund) pair.

    Parameters
    ----------
    returns : wide DataFrame (index = trading dates, columns = assets)
    method : one of METHODS
    window : estimation window in trading days (252 for a year of equity days)
    rebalance : "monthly" (first trading day of each month) or "every_n"

    Returns
    -------
    dict with the OOS daily returns, growth of $1, drawdown, and the weight
    schedule (index = rebalance dates) used to produce them.
    """
    dates = pd.DatetimeIndex(returns.index)
    returns = returns.set_axis(dates)
    if len(dates) <= window:
        raise ValueError("returns must be longer than the estimation window")

    rebal_dates = _rebalance_dates(dates, window, rebalance, every)
    pairs = []
    for rb in rebal_dates:
        end = dates.get_loc(rb) + 1
        w_ret = returns.iloc[end - window:end]
        try:
            w = compute_weights(w_ret, method)
        except ValueError:
            continue
        pairs.append((rb, w))
    if not pairs:
        raise RuntimeError(f"no rebalance date produced weights for method {method!r}")

    weights = pd.DataFrame([w for _, w in pairs], index=[d for d, _ in pairs])
    weights = weights.loc[~weights.index.duplicated(keep="last")].sort_index()

    port, active = evaluate_weights(returns, weights)
    growth = (1.0 + port).cumprod()
    drawdown = growth / growth.cummax() - 1.0
    return {
        "method": method,
        "window": window,
        "rebalance": rebalance,
        "first_live_date": port.index.min(),
        "last_date": port.index.max(),
        "returns": port,
        "growth": growth,
        "drawdown": drawdown,
        "weights": weights,
        "active_weights": active,
        "rebalance_dates": list(weights.index),
    }


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def performance_metrics(daily_returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Annualised return, annualised volatility, Sharpe (rf = 0), max drawdown.

    Annualised return is geometric (the annual rate that compounds the OOS
    daily series to its observed cumulative growth). Volatility is the daily
    std scaled by sqrt(periods_per_year). Sharpe uses a risk-free rate of 0,
    a stated modelling choice.
    """
    r = daily_returns.dropna()
    n = len(r)
    if n == 0:
        return {"annualized_return": np.nan, "annualized_vol": np.nan,
                "sharpe": np.nan, "max_drawdown": np.nan, "n_days": 0}
    total_growth = float(np.prod(1.0 + r.to_numpy()))
    ann_return = total_growth ** (periods_per_year / n) - 1.0
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = ann_return / ann_vol if ann_vol > 1e-12 else np.nan
    cum = (1.0 + r).cumprod()
    max_dd = float((cum / cum.cummax() - 1.0).min())
    return {
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_days": int(n),
    }
def apply_transaction_costs(returns: pd.Series, weights: pd.DataFrame,
                            cost_bps: float = 50.0) -> tuple[pd.Series, pd.DataFrame]:
    """Deduct a simple turnover-based transaction cost from a backtest's daily
    return series.

    At each rebalance, turnover is the one-way sum of absolute weight changes
    versus the previously active weights (the first rebalance is priced from
    an all-cash start, so its turnover is the cost of building the initial
    portfolio). Cost = turnover * cost_bps / 10,000, deducted from the
    portfolio's return on the first day the new weights are active - the same
    day evaluate_weights' shift(1) makes them active, so this never
    introduces look-ahead.

    Returns
    -------
    (net_returns, turnover_table) - turnover_table has one row per rebalance
    with the turnover and dollar cost applied, for reporting.
    """
    w = weights.sort_index()
    prev = w.shift(1).fillna(0.0)
    turnover = (w - prev).abs().sum(axis=1)
    cost = turnover * (cost_bps / 10_000.0)

    dates = returns.index
    net = returns.copy()
    applied = []
    for rebal_date, c in cost.items():
        pos = dates.searchsorted(rebal_date)
        if pos + 1 < len(dates):
            active_day = dates[pos + 1]
            if active_day in net.index:
                net.loc[active_day] = net.loc[active_day] - c
                applied.append((rebal_date, active_day, turnover[rebal_date], c))

    turnover_table = pd.DataFrame(
        applied, columns=["rebalance_date", "cost_applied_date", "turnover", "cost"])
    return net, turnover_table
