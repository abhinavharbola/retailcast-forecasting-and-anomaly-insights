import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.metrics import mape, mase, wape


def test_mape_perfect_forecast():
    assert mape([10, 20, 30], [10, 20, 30]) == 0


def test_mape_ignores_zero_actuals():
    assert mape([0, 10], [5, 12]) == mape([10], [12])


def test_wape_basic():
    assert wape([10, 10], [12, 8]) == 20.0


def test_mase_naive_baseline_equals_one():
    train = list(range(1, 100))
    actual = [50, 51, 52]
    forecast = [43, 44, 45]
    assert abs(mase(actual, forecast, train) - 1.0) < 1e-9


def test_mase_perfect_forecast_is_zero():
    train = list(range(1, 30))
    assert mase([5, 5, 5], [5, 5, 5], train) == 0