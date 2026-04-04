from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    hold_window: int = 5,
    use_trailing_stop: bool = False,
    trailing_k: float = 2.0,
    hard_stop_pct: float = 0.03,
    starting_capital: float = 10000.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Simulate a realistic 5-day execution model for entry signals in `df`.

    Assumptions / requirements implemented:
    - Uses `entry_nonoverlap` column to open trades.
    - Max holding period = `hold_window` days (default 5).
    - ATR-based stop uses `stop_price_atr` if present; else fallback -hard_stop_pct from entry.
    - Entry executed at `close` on entry day; stop detection uses `low` on subsequent days.
    - Final exit (if no stop) uses `future_close_5` (close at day+hold_window) when available.
    - No lookahead bias: decisions only use data from entry day and subsequent days.

    Returns:
    - df_out: copy of input df with added columns: trade_id, entry_price, exit_price,
      exit_reason, hold_length, actual_return
    - results: Series of actual_return indexed by the original df index for entry rows
    """

    # Work on a copy
    df_out = df.copy()

    # Ensure deterministic ordering per ticker by date
    if 'date' in df_out.columns:
        df_out = df_out.sort_values(['ticker', 'date']).reset_index(drop=True)
    else:
        df_out = df_out.sort_values(['ticker']).reset_index(drop=True)

    # Prepare columns
    df_out['trade_id'] = pd.NA
    df_out['entry_price'] = pd.NA
    df_out['exit_price'] = pd.NA
    df_out['exit_reason'] = pd.NA
    df_out['hold_length'] = pd.NA
    df_out['actual_return'] = pd.NA

    trade_counter = 0

    # iterate per-ticker
    for ticker, grp in df_out.groupby('ticker'):
        idx = grp.index.to_numpy()
        closes = grp['close'].to_numpy()
        lows = grp['low'].to_numpy()
        # Optional columns
        stop_price_atr_series = grp['stop_price_atr'].to_numpy() if 'stop_price_atr' in grp.columns else np.array([np.nan] * len(grp))
        atr_series = grp['atr_14'].to_numpy() if 'atr_14' in grp.columns else np.array([np.nan] * len(grp))
        future_close = grp['future_close_5'].to_numpy() if 'future_close_5' in grp.columns else np.array([np.nan] * len(grp))
        entries = grp.get('entry_nonoverlap', grp.get('entry', pd.Series(False, index=grp.index))).to_numpy()

        n = len(idx)

        i = 0
        while i < n:
            if not entries[i]:
                i += 1
                continue

            # Start a trade
            trade_counter += 1
            entry_pos = i
            entry_idx = idx[entry_pos]
            entry_price = closes[entry_pos]

            # initial stop
            stop_price = stop_price_atr_series[entry_pos]
            if pd.isna(stop_price) or stop_price is None:
                # fallback hard stop
                stop_price = entry_price * (1.0 - hard_stop_pct)

            exit_price = np.nan
            exit_reason = None
            exit_hold = np.nan

            # scan next days (1..hold_window)
            for offset in range(1, hold_window + 1):
                j = entry_pos + offset
                if j >= n:
                    break

                # optional trailing stop update (only move stop up)
                if use_trailing_stop:
                    # compute ATR-based daily stop if atr available else use percent-based
                    daily_atr = atr_series[j]
                    if not pd.isna(daily_atr):
                        candidate = closes[j] - trailing_k * daily_atr
                    else:
                        candidate = closes[j] * (1.0 - hard_stop_pct)
                    # only move stop upward (i.e., closer to price)
                    if not pd.isna(candidate) and candidate > stop_price:
                        stop_price = candidate

                # check stop hit on this day's low
                low_j = lows[j]
                if (not pd.isna(low_j)) and (low_j <= stop_price):
                    # exited at stop price (assume we get stop_price)
                    exit_price = stop_price
                    exit_reason = 'stop'
                    exit_hold = offset
                    break

            # if no stop hit, exit at time limit
            if exit_reason is None:
                # prefer future_close at entry (which is the close at entry+hold_window)
                fc = future_close[entry_pos]
                if not pd.isna(fc):
                    exit_price = fc
                else:
                    # fallback: use last available close within window
                    last_pos = min(entry_pos + hold_window, n - 1)
                    exit_price = closes[last_pos]
                exit_reason = 'time'
                # compute actual hold length: if we have last_pos maybe less than hold_window
                # We want to know how many days elapsed until exit: if future_close used, it's hold_window
                exit_hold = min(hold_window, n - 1 - entry_pos)

            # Safety: ensure numeric
            if pd.isna(entry_price) or pd.isna(exit_price):
                actual_r = np.nan
            else:
                actual_r = float(exit_price) / float(entry_price) - 1.0

            # write back into df_out at entry row only (trade-level fields are stored on entry row)
            df_out.at[entry_idx, 'trade_id'] = trade_counter
            df_out.at[entry_idx, 'entry_price'] = float(entry_price) if not pd.isna(entry_price) else pd.NA
            df_out.at[entry_idx, 'exit_price'] = float(exit_price) if not pd.isna(exit_price) else pd.NA
            df_out.at[entry_idx, 'exit_reason'] = exit_reason
            df_out.at[entry_idx, 'hold_length'] = int(exit_hold) if not pd.isna(exit_hold) else pd.NA
            df_out.at[entry_idx, 'actual_return'] = actual_r

            # move pointer forward by 1 (entries are already non-overlapping by construction)
            i += 1

    # Results series per requirement
    results = df_out.loc[df_out.get('entry_nonoverlap', df_out.get('entry', False)), 'actual_return'].dropna()

    # Metrics and diagnostics
    num_trades = len(results)
    mean_return = results.mean() if num_trades > 0 else float('nan')
    win_rate = (results > 0).mean() if num_trades > 0 else float('nan')
    avg_hold = df_out.loc[df_out.get('entry_nonoverlap', df_out.get('entry', False)), 'hold_length'].dropna().astype(float).mean()

    # Equity curve (sequential full allocation per trade)
    equity = []
    capital = float(starting_capital)
    for r in results:
        capital *= 1.0 + float(r)
        equity.append(capital)

    final_capital = capital
    total_return = (final_capital / starting_capital - 1.0) if len(equity) > 0 else 0.0

    # max drawdown
    if len(equity) > 0:
        eq = np.array(equity)
        peak = np.maximum.accumulate(eq)
        drawdowns = (peak - eq) / peak
        max_dd = float(np.nanmax(drawdowns))
    else:
        max_dd = 0.0

    # Print metrics
    print('Backtest summary:')
    print(f'  Number of trades: {num_trades}')
    print(f'  Mean return: {mean_return:.6f}')
    print(f'  Win rate: {win_rate:.2%}')
    print(f'  Avg hold length: {avg_hold if not pd.isna(avg_hold) else "n/a"}')
    print(f'  Max drawdown: {max_dd:.2%}')
    print(f'  Final capital: ${final_capital:,.2f} (total return {total_return:.2%})')

    # Debugging / validation prints
    print('\nFirst 10 trades (entry rows):')
    display_cols = ['trade_id', 'ticker', 'date', 'entry_price', 'exit_price', 'hold_length', 'actual_return', 'exit_reason']
    try:
        # pandas display optional
        print(df_out.loc[df_out['trade_id'].notna(), display_cols].head(10).to_string(index=False))
    except Exception:
        print(df_out.loc[df_out['trade_id'].notna(), display_cols].head(10))

    # Validation checks
    if (results.isna().sum() > 0):
        print('Warning: some trades have NaN returns')
    if (df_out.loc[df_out['trade_id'].notna(), 'hold_length'].dropna().astype(float) > hold_window).any():
        print('Warning: some hold lengths exceed the hold window')

    return df_out, results
