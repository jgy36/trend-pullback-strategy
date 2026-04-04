"""Signal preparation utilities for the trend-pullback strategy.

This module extracts the preprocessing and signal logic from the notebook so it
can be imported and tested. The primary entrypoint is `prepare_df(df, **params)`
which returns a new DataFrame with additional feature and signal columns.

Design notes:
- The function is careful to sort by `ticker` and `date` if present so groupby-shift
  operations are deterministic.
- Default parameters mirror the notebook (ema spans, ATR window, etc.) but are
  configurable for testing and walk-forward runs.
- All shift() operations are performed within ticker groups to prevent cross-ticker
  data leakage.
"""
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
import os


def prepare_df(
    df: pd.DataFrame,
    ema_short: int = 20,
    ema_mid: int = 50,
    ema_long: int = 200,
    rsi_length: int = 14,
    atr_window: int = 14,
    atr_k: float = 2.0,
    volume_ma: int = 20,
    atr_ok_mult: float = 0.5,
    hold: int = 5,
    date_col: Optional[str] = 'date',
):
    """Return a copy of df enriched with strategy features and signals.

    Required input columns: ['open','high','low','close','volume','ticker']
    Optionally provide `date_col` to ensure sorting. The function will not
    mutate the original DataFrame (it returns a copy).
    """
    required = {'open', 'high', 'low', 'close', 'volume', 'ticker'}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    df = df.copy()

    # Sort for deterministic groupby/shift behavior
    if date_col and date_col in df.columns:
        df = df.sort_values(['ticker', date_col]).reset_index(drop=True)
    else:
        df = df.sort_values(['ticker']).reset_index(drop=True)

    # --- EMAs ---
    df[f'ema_{ema_long}'] = df.groupby('ticker')['close'].transform(
        lambda x: x.ewm(span=ema_long).mean()
    )
    df[f'ema_{ema_mid}'] = df.groupby('ticker')['close'].transform(
        lambda x: x.ewm(span=ema_mid).mean()
    )
    df[f'ema_{ema_short}'] = df.groupby('ticker')['close'].transform(
        lambda x: x.ewm(span=ema_short).mean()
    )

    # --- RSI ---
    df['rsi'] = df.groupby('ticker')['close'].transform(
        lambda x: ta.rsi(x, length=rsi_length)
    )

    # --- True Range (grouped shift to prevent cross-ticker leakage) ---
    high = df['high']
    low = df['low']
    close = df['close']

    # FIX: use grouped shift so the first row of each ticker doesn't bleed
    # into the last row of the previous ticker
    prev_close = df.groupby('ticker')['close'].shift(1)

    df['tr'] = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    df['atr_14'] = df.groupby('ticker')['tr'].transform(
        lambda x: x.rolling(atr_window, min_periods=1).mean()
    )

    # --- Volume MA for liquidity filter ---
    df['volume_ma'] = df.groupby('ticker')['volume'].transform(
        lambda x: x.rolling(volume_ma, min_periods=1).mean()
    )

    # --- Strategy signals ---
    df['trend'] = (
        (df['close'] > df[f'ema_{ema_long}']) &
        (df[f'ema_{ema_short}'] > df[f'ema_{ema_mid}']) &
        (df[f'ema_{ema_mid}'] > df[f'ema_{ema_long}'])
    )

    df['pullback'] = (
        (df['close'] < df[f'ema_{ema_short}']) &
        (df['close'] > df[f'ema_{ema_long}'])
    )
    df['oversold'] = df['rsi'] < 45

    # FIX: use grouped shift so RSI lag doesn't cross ticker boundaries
    rsi_prev = df.groupby('ticker')['rsi'].shift(1)
    df['trigger'] = (df['rsi'] > rsi_prev) & (rsi_prev < 40)

    # --- Trend slope ---
    df['trend_slope'] = df.groupby('ticker')[f'ema_{ema_long}'].transform(
        lambda x: x.diff(20)
    )
    df['trend_slope_flag'] = df['trend_slope'] > 0

    # --- Price strength (momentum confirmation) ---
    # FIX: grouped shift for consistency
    close_prev = df.groupby('ticker')['close'].shift(1)
    df['price_strength'] = (df['close'] > close_prev).fillna(False)

    # --- Volume and ATR filters ---
    df['volume_ok'] = df['volume'] > df['volume_ma']
    df['atr_median'] = df.groupby('ticker')['atr_14'].transform('median')
    df['atr_ok'] = df['atr_14'] > (atr_ok_mult * df['atr_median'])

    # --- Entry signal ---
    df['entry_base'] = (
        df['trend'] &
        df['pullback'] &
        df['oversold'] &
        df['trigger'] &
        df['trend_slope_flag'] &
        df['price_strength'] &
        df['volume_ok'] &
        df['atr_ok']
    )
    df['entry'] = df['entry_base'].astype(bool)

    # --- Forward return / aligned stops ---
    df['min_low_next_5'] = df.groupby('ticker')['low'].transform(
        lambda s: s.shift(-1).rolling(hold, min_periods=1).min()
    )
    df['stop_return'] = (df['min_low_next_5'] - df['close']) / df['close']

    # ATR-based stop price and forward return
    df['stop_price_atr'] = df['close'] - atr_k * df['atr_14']
    df['stop_return_atr'] = (df['stop_price_atr'] - df['close']) / df['close']

    df['future_close_5'] = df.groupby('ticker')['close'].shift(-hold)
    df['forward_return_5'] = df['future_close_5'] / df['close'] - 1

    return df


def download_price_data(
    tickers: List[str],
    start: str,
    end: str,
    group_by: str = 'ticker',
    threads: bool = True,
) -> pd.DataFrame:
    # Defensive normalization of tickers: strip whitespace and uppercase
    tickers = [t.strip().upper() for t in tickers]

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        group_by=group_by,
        threads=threads,
        progress=False,
    )

    if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
        raise RuntimeError(
            "yfinance returned no data for the requested tickers/date range. "
            "Check tickers, network, or date range."
        )

    out_frames = []
    full_idx = pd.date_range(start=start, end=end, freq='B')

    if isinstance(raw.columns, pd.MultiIndex):
        # Modern yfinance: MultiIndex is (Ticker, Price) — e.g. ('AAPL', 'Open')
        # Detect level order by checking which level contains known price fields
        price_fields = {'Open', 'High', 'Low', 'Close', 'Volume'}
        level0_vals = set(raw.columns.get_level_values(0))
        level1_vals = set(raw.columns.get_level_values(1))

        if price_fields & level1_vals:
            # (Ticker, Price) layout — modern yfinance
            ticker_level, price_level = 0, 1
        else:
            # (Price, Ticker) layout — older yfinance
            ticker_level, price_level = 1, 0

        ticker_labels = raw.columns.get_level_values(ticker_level).unique()

        for t in ticker_labels:
            if t not in tickers:
                continue
            try:
                if ticker_level == 0:
                    sub = raw.loc[:, (t, slice(None))]
                    sub.columns = sub.columns.droplevel(0)
                else:
                    sub = raw.loc[:, (slice(None), t)]
                    sub.columns = sub.columns.droplevel(1)
            except Exception:
                continue

            # Skip all-NaN tickers (delisted / not yet listed)
            if sub.isna().all().all():
                continue

            sub = sub.rename(columns=lambda c: str(c).lower())
            sub.index.name = 'date'
            sub = sub.reset_index()

            # Normalize date column name across yfinance versions
            sub = sub.rename(columns={'Date': 'date', 'Datetime': 'date', 'index': 'date'})
            sub['ticker'] = t

            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col not in sub.columns:
                    sub[col] = np.nan

            sub = sub[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
            s2 = (
                sub.set_index('date')
                .reindex(full_idx)
                .reset_index()
                .rename(columns={'index': 'date'})
            )
            s2['ticker'] = t
            out_frames.append(s2)

    else:
        # Single-ticker download — flat columns
        single = raw.reset_index()
        single = single.rename(columns={'Date': 'date', 'Datetime': 'date'})
        single = single.rename(columns=lambda c: str(c).lower())
        single['ticker'] = tickers[0]

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in single.columns:
                single[col] = np.nan

        single = single[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
        s2 = (
            single.set_index('date')
            .reindex(full_idx)
            .reset_index()
            .rename(columns={'index': 'date'})
        )
        s2['ticker'] = tickers[0]
        out_frames.append(s2)

    if not out_frames:
        raise RuntimeError(
            "After processing, no per-ticker frames available. "
            "Check raw download contents and ticker symbols."
        )

    combined = pd.concat(out_frames, ignore_index=True, sort=False)
    combined[['open', 'high', 'low', 'close', 'volume']] = (
        combined[['open', 'high', 'low', 'close', 'volume']]
        .apply(pd.to_numeric, errors='coerce')
    )

    return combined


def select_top_by_dollar_volume(
    df: pd.DataFrame,
    top_n: int = 200,
    window_days: int = 60,
    end_date: Optional[str] = None,
) -> List[str]:
    """Return the top_n tickers by average daily dollar volume over the last
    `window_days` business days ending at `end_date`.
    """
    if end_date is None:
        end = df['date'].max()
    else:
        end = pd.to_datetime(end_date)

    start = pd.to_datetime(end) - pd.Timedelta(days=int(window_days * 1.5))
    recent = df[(df['date'] <= end) & (df['date'] >= start)].copy()
    recent = recent.dropna(subset=['close', 'volume'])

    if recent.empty:
        return []

    recent['dollar_vol'] = recent['close'] * recent['volume']
    avg = recent.groupby('ticker')['dollar_vol'].mean()
    top = avg.nlargest(top_n).index.tolist()
    return top


def prepare_universe(
    tickers: Optional[List[str]] = None,
    sp500_csv: Optional[str] = None,
    dynamic: bool = True,
    top_n: int = 200,
    start: str = '2020-01-01',
    end: str = '2024-01-01',
    vol_window: int = 60,
    **prepare_kwargs,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build and prepare a universe for backtesting.

    Returns (prepared_df, tickers_used). Prints summary statistics about the
    universe and basic trade metrics.

    - If `dynamic` is True and `tickers` is None, attempts to load candidates from
      `sp500_csv` (if provided) and selects top_n by dollar volume over the last
      `vol_window` days.
    - If `tickers` is provided, uses that list directly.
    """
    candidates = []
    if tickers:
        candidates = list(tickers)
    elif sp500_csv and os.path.exists(sp500_csv):
        cand_df = pd.read_csv(sp500_csv, header=None)
        # Normalize candidates: strip and uppercase
        candidates = [t.strip().upper() for t in cand_df.iloc[:, 0].astype(str).tolist()]
    else:
        raise ValueError(
            "No tickers provided and sp500_csv not found. "
            "Provide tickers or a path to a CSV."
        )

    raw = download_price_data(candidates, start=start, end=end)

    if dynamic:
        tops = select_top_by_dollar_volume(raw, top_n=top_n, window_days=vol_window, end_date=end)
        if not tops:
            raise RuntimeError(
                "Dynamic selection returned no tickers; "
                "check data availability and dates."
            )
        tickers_used = tops
        raw = raw[raw['ticker'].isin(tickers_used)].copy()
    else:
        tickers_used = candidates

    prepared = prepare_df(raw, **prepare_kwargs)

    n_tickers = len(set(prepared['ticker'].dropna()))
    n_trades = int(prepared['entry'].sum()) if 'entry' in prepared.columns else 0

    if 'actual_return' in prepared.columns:
        perf_series = prepared.loc[prepared['entry'], 'actual_return']
    else:
        perf_series = prepared.loc[prepared['entry'], 'forward_return_5']
    perf_series = perf_series.dropna()

    avg_return = perf_series.mean() if len(perf_series) else float('nan')
    win_rate = (perf_series > 0).mean() if len(perf_series) else float('nan')

    print(f"Tickers used:              {n_tickers}")
    print(f"Candidate trades (entries): {n_trades}")
    print(f"Avg return:                {avg_return:.4f}" if not np.isnan(avg_return) else "Avg return: nan")
    print(f"Win rate:                  {win_rate:.4f}" if not np.isnan(win_rate) else "Win rate: nan")

    return prepared, tickers_used
