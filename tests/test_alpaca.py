import numpy as np

from trajectory_risk.alpaca import ALPaCAConfig, ALPaCAForecaster
from trajectory_risk.conformal_adapter import ARForecaster, ConformalIntervalAdapter, _half_width
from trajectory_risk.history import ActionMode

from test_end_to_end import config


def series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return np.sin(np.arange(n)/7) + 0.3*rng.standard_normal(n)


def test_alpaca_shapes_positive_scale_and_no_lookahead():
    y = series()
    f = ALPaCAForecaster(ALPaCAConfig(iterations=50, n_context=8), horizon=3).fit(y[:200])
    origins = np.arange(250, 260)
    mean, scale = f.predict_all(y, origins, 3)
    assert mean.shape == scale.shape == (10, 3)
    assert np.all(np.isfinite(mean)) and np.all(scale > 0)
    future = y.copy()
    future[origins[0]+1:] += 50  # Everything after the first origin changes.
    m2, s2 = f.predict_all(future, origins[:1], 3)
    np.testing.assert_allclose(m2, mean[:1])
    np.testing.assert_allclose(s2, scale[:1])


def test_conformal_level_matches_issued_interval_for_both_forecasters():
    c = config(t_max=20)
    y = series()
    for forecaster in (ARForecaster(), ALPaCAForecaster(ALPaCAConfig(iterations=30, n_context=5), 3)):
        adapter = ConformalIntervalAdapter(c, y, n_train=150, window=30, forecaster=forecaster)
        for t in range(c.t_max):
            adapter.observe(t)
            adapter.prepare(t, type("G", (), {"values": np.linspace(.05, .3, 4)})())
            adapter.issue(t, ActionMode.ORDINARY, np.array([.05, .1, .2]))
        for k in range(c.t_max - 3):
            r = adapter.records[k]
            inside = np.abs(r["y"] - r["yhat"]) <= r["half_width"]
            np.testing.assert_array_equal(inside, r["level"] >= r["alpha"])
            np.testing.assert_array_equal(~inside, r["viol"])
            np.testing.assert_allclose(r["half_width"], [r["scale"][h]*_half_width(r["sdesc"][h], r["alpha"][h])
                                                         for h in range(3)])
