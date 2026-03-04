import math

from utils.multi_seed import summarize


def test_summarize_single_value():
    s = summarize([0.5])
    assert s.n == 1
    assert s.mean == 0.5
    assert s.std == 0.0
    assert math.isnan(s.ci95)


def test_summarize_two_values_ci_positive():
    s = summarize([0.4, 0.6])
    assert s.n == 2
    assert abs(s.mean - 0.5) < 1e-12
    assert s.std > 0
    assert s.ci95 > 0

