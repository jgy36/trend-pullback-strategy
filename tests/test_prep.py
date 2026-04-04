import pandas as pd
import numpy as np

from strategy.prep import prepare_df


def make_sample_df(days=30):
    # Two tickers, deterministic increasing close for ticker A, mixed for B
    dates = pd.date_range('2021-01-01', periods=days, freq='D')
    rows = []
    for t in ['AAA', 'BBB']:
        for i, d in enumerate(dates):
            if t == 'AAA':
                close = 100 + i * 0.5  # steady upward drift
                low = close - 0.2
                high = close + 0.2
                openp = close - 0.1
                vol = 1000 + i * 10
            else:
                # B has some noise
                close = 50 + ((-1) ** i) * (i % 5) * 0.2
                low = close - 0.3
                high = close + 0.3
                openp = close - 0.05
                vol = 500 + (i % 3) * 50
            rows.append({
                'date': d,
                'ticker': t,
                'open': openp,
                'high': high,
                'low': low,
                'close': close,
                'volume': vol,
            })
    return pd.DataFrame(rows)


def test_prepare_df_creates_columns():
    df = make_sample_df(days=30)
    # Use small EMAs and ATR windows so the test is stable
    out = prepare_df(df, ema_short=3, ema_mid=5, ema_long=10, rsi_length=5, atr_window=5, volume_ma=3, hold=5)

    # Check a selection of expected columns
    expected_cols = [
        'ema_10', 'ema_5', 'ema_3', 'rsi', 'atr_14', 'volume_ma',
        'trend', 'pullback', 'oversold', 'trigger', 'trend_slope', 'trend_slope_flag',
        'price_strength', 'volume_ok', 'atr_ok', 'entry_base', 'entry',
        'min_low_next_5', 'stop_price_atr', 'future_close_5', 'forward_return_5'
    ]
    for c in expected_cols:
        assert c in out.columns, f"Missing column {c}"

    # Price strength: ensure the deterministic ticker 'AAA' has a rising second row
    g_aaa = out[out['ticker'] == 'AAA'].reset_index(drop=True)
    assert len(g_aaa) >= 2
    assert bool(g_aaa['price_strength'].iloc[1]) is True

    # forward_return_5 should equal close.shift(-5)/close - 1 where available
    sample = out[(out['ticker'] == 'AAA')].reset_index(drop=True)
    # manually compute for index 0
    if len(sample) > 5:
        expected = sample.loc[5, 'close'] / sample.loc[0, 'close'] - 1
        assert np.isclose(sample.loc[0, 'forward_return_5'], expected, atol=1e-12)
