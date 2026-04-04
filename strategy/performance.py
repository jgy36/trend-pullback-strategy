"""Performance analytics helpers for backtest outputs.

Functions accept either a trades DataFrame with a 'return' column or a returns
Series/DataFrame. All functions are vectorized and robust to NaNs and empty
inputs.

Public API:
- analyze_performance(data, return_col='return', trades_per_year=None, interpret=False)

"""
from __future__ import annotations

from typing import Optional, Dict

import numpy as np
import pandas as pd


def _extract_returns(data, return_col: str = "return") -> pd.Series:
    """Return a 1-D float Series of returns from either a DataFrame or Series.

    Drops NaNs. If DataFrame and the column is missing, raises ValueError.
    """
    if data is None:
        return pd.Series(dtype=float)

    if isinstance(data, pd.DataFrame):
        if return_col not in data.columns:
            raise ValueError(f"DataFrame missing return column: {return_col}")
        s = data[return_col]
    elif isinstance(data, (pd.Series, pd.Index)):
        s = pd.Series(data)
    else:
        # try to convert
        s = pd.Series(data)

    s = s.dropna().astype(float)
    s.index = s.index  # keep index for potential time-based inference
    return s


def _annualization_factor(returns: pd.Series, trades_per_year: Optional[float]) -> float:
    """Estimate the periods per year for annualizing Sharpe.

    If trades_per_year provided, use it. Otherwise infer from index:
    - If DatetimeIndex and median delta <= 3 days: assume daily -> 252
    - If DatetimeIndex: estimate trades_per_year = len / years_span
    - Otherwise fallback to len(returns) (treat as trades per year)
    """
    if trades_per_year is not None:
        return float(trades_per_year)

    if hasattr(returns.index, "is_all_dates") or isinstance(returns.index, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(returns.index)
        if len(idx) < 2:
            return max(1.0, float(len(returns)))
        diffs = idx.to_series().diff().dropna().dt.days.abs()
        median_diff = float(diffs.median())
        if median_diff <= 3:
            return 252.0
        # span-based estimate
        span_days = float((idx.max() - idx.min()).days)
        years = span_days / 365.25 if span_days > 0 else 1.0
        trades_per_year = len(returns) / years if years > 0 else len(returns)
        return max(1.0, float(trades_per_year))

    # Non-datetime index: assume these are trades; annualize by number of observations
    return max(1.0, float(len(returns)))


def sharpe_ratio(returns: pd.Series, trades_per_year: Optional[float] = None, risk_free: float = 0.0) -> float:
    """Compute Sharpe ratio (annualized). Risk-free rate assumed in same periodic units.

    Returns np.nan if not defined (e.g., zero variance, empty input).
    """
    r = returns.dropna().astype(float)
    if r.empty:
        return float("nan")

    mean = r.mean() - risk_free
    std = r.std(ddof=0)
    if std == 0 or np.isnan(std):
        return float("nan")

    periods_per_year = _annualization_factor(r, trades_per_year)
    return float(mean / std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown from returns (periodic returns).

    Uses cumulative wealth series starting at 1.0 and returns the maximum drawdown
    as a negative decimal (e.g., -0.25 for 25% drawdown).
    """
    r = returns.dropna().astype(float)
    if r.empty:
        return 0.0

    equity = (1.0 + r).cumprod()
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    return float(drawdown.min())


def profit_factor(returns: pd.Series) -> float:
    """Compute profit factor = sum(wins) / abs(sum(losses)).

    Returns np.inf if there are no losses and positive wins exist. Returns np.nan
    if there are neither wins nor losses.
    """
    r = returns.dropna().astype(float)
    if r.empty:
        return float("nan")

    wins = r[r > 0].sum()
    losses = r[r < 0].sum()  # negative or zero
    if losses == 0:
        if wins == 0:
            return float("nan")
        return float("inf")
    return float(wins / abs(losses))


def total_return(returns: pd.Series) -> float:
    """Compute compounded total return: prod(1 + r) - 1

    Returns 0.0 for empty input.
    """
    r = returns.dropna().astype(float)
    if r.empty:
        return 0.0
    return float((1.0 + r).prod() - 1.0)


def analyze_performance(data, return_col: str = "return", trades_per_year: Optional[float] = None, interpret: bool = False) -> Dict[str, float]:
    """Analyze performance and return a dict of metrics.

    data may be:
    - pd.Series of returns
    - pd.DataFrame with a returns column (default name: 'return')

    The returned dictionary contains:
      win_rate, avg_win, avg_loss, loss_rate, expectancy, sharpe,
      profit_factor, max_drawdown, total_return

    If interpret=True, prints a short interpretation of the key metrics.
    """
    returns = _extract_returns(data, return_col=return_col)

    # Basic win/loss statistics
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    win_rate = float(len(wins)) / len(returns) if len(returns) > 0 else 0.0
    loss_rate = float(len(losses)) / len(returns) if len(returns) > 0 else 0.0

    expectancy = win_rate * avg_win + loss_rate * avg_loss

    sharpe = sharpe_ratio(returns, trades_per_year=trades_per_year, risk_free=0.0)

    pf = profit_factor(returns)
    mdd = max_drawdown(returns)
    tot = total_return(returns)

    metrics = {
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "loss_rate": loss_rate,
        "expectancy": expectancy,
        "sharpe": sharpe,
        "profit_factor": pf,
        "max_drawdown": mdd,
        "total_return": tot,
    }

    if interpret:
        _print_interpretation(metrics)

    return metrics


def _print_interpretation(metrics: Dict[str, float]) -> None:
    """Print short interpretation of select metrics."""
    sharpe = metrics.get("sharpe")
    expectancy = metrics.get("expectancy")
    pf = metrics.get("profit_factor")

    def _fmt(x):
        if x is None:
            return "n/a"
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return str(x)
        return f"{x:.4f}"

    print("Performance interpretation:")
    if sharpe is None or np.isnan(sharpe):
        print(" - Sharpe: n/a")
    else:
        print(f" - Sharpe: {_fmt(sharpe)}", end="; ")
        if sharpe > 2:
            print("very strong")
        elif sharpe > 1:
            print("good")
        else:
            print("weak")

    print(f" - Expectancy: {_fmt(expectancy)}", end="; ")
    print("positive" if (expectancy is not None and expectancy > 0) else "not positive")

    if pf is None or (isinstance(pf, float) and np.isnan(pf)):
        print(" - Profit factor: n/a")
    else:
        print(f" - Profit factor: {_fmt(pf)}", end="; ")
        if pf > 1.5:
            print("solid")
        else:
            print("weak")
