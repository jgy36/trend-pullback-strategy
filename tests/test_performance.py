import math

import pandas as pd

from strategy.performance import analyze_performance


def test_analyze_from_series():
    # mix of wins and losses
    s = pd.Series([0.02, -0.01, 0.05, -0.02, 0.0, 0.03])
    metrics = analyze_performance(s)

    # basic keys
    for k in [
        "win_rate",
        "avg_win",
        "avg_loss",
        "expectancy",
        "sharpe",
        "profit_factor",
        "max_drawdown",
        "total_return",
    ]:
        assert k in metrics

    # win_rate should be > 0
    assert metrics["win_rate"] > 0
    # avg_win positive, avg_loss negative or zero
    assert metrics["avg_win"] >= 0
    assert metrics["avg_loss"] <= 0


def test_analyze_from_dataframe():
    df = pd.DataFrame({"return": [0.01, 0.01, -0.005, -0.02, 0.03]})
    metrics = analyze_performance(df, return_col="return")

    # profit factor should be finite positive
    pf = metrics["profit_factor"]
    assert not (isinstance(pf, float) and math.isnan(pf))

    # total_return should equal product(1+r)-1
    expected_total = (1 + df["return"]).prod() - 1
    assert abs(metrics["total_return"] - expected_total) < 1e-12
