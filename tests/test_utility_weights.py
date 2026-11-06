from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajectory_risk.configs import WeightConfig
from trajectory_risk.history import ActionMode, AlphaGrid, UpdateError, add_origin, init_history, resolve_horizon, resolve_trajectory
from trajectory_risk.utility_weights import (
    bounded_simplex_projection, build_weight_updater, smooth_and_cap,
)


def accept(result):
    assert int(result.error) == 0
    return result.state


def fresh(dtype=jnp.float64):
    return init_history(AlphaGrid.create(jnp.array([0.05, 0.1, 0.2], dtype)), 3, 8, 0.05, 0.2)


def curves(history, efficiency=(1, 3, 8)):
    return 10 - jnp.asarray(efficiency, history.grid.values.dtype)[:, None]*history.grid.values


def project_reference(raw, lower, upper):
    lo, hi = np.min(raw)-upper, np.max(raw)-lower
    for _ in range(150):
        mid = (lo+hi)/2
        if np.clip(raw-mid, lower, upper).sum() > 1:
            lo = mid
        else:
            hi = mid
    return np.clip(raw-(lo+hi)/2, lower, upper)


@pytest.mark.parametrize("raw", [[0.01, 0.02, 0.03], [1, 10, 100], [-1, 0, 3], [0, 0, 0]])
def test_bounded_projection_against_numpy_reference(raw):
    expected = project_reference(np.array(raw, float), 0.05, 0.8)
    actual = jax.jit(bounded_simplex_projection)(jnp.array(raw), 0.05, 0.8)
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    np.testing.assert_allclose(np.sum(actual), 1)
    assert np.asarray((actual >= 0.05) & (actual <= 0.8)).all()


@pytest.mark.parametrize("kind", ["static", "marginal_efficiency", "failure_aware"])
def test_static_application_burn_in_and_missing_size_fallback(kind):
    history = fresh()
    cfg = WeightConfig(kind=kind, application_weights=(1, 2, 3), failure_window=2)
    updater = build_weight_updater(cfg)
    state = updater.init_state(history)
    expected = np.array([1, 2, 3])/6
    missing = updater.update(state, history, jnp.full(3, 0.1), 0)
    state = accept(missing)
    np.testing.assert_allclose(state.values, expected)
    assert bool(missing.used_static_fallback) == (kind != "static")
    result = jax.jit(updater.update)(state, history, jnp.full(3, 0.1), 1, curves(history))
    require_fallback = kind == "failure_aware"
    assert bool(result.used_static_fallback) == require_fallback
    if require_fallback or kind == "static":
        np.testing.assert_allclose(result.state.values, expected)


def test_efficiency_uses_current_curve_and_raw_projection_before_smoothing():
    history = fresh()
    cfg = WeightConfig(kind="marginal_efficiency", smoothing=0.5, max_step_norm=0.05,
                       min_weight=0.05, max_weight=0.8)
    updater = build_weight_updater(cfg)
    state = updater.init_state(history)
    # Stored geometry is deliberately different from the current predictable curve.
    history = accept(add_origin(history, 0, ActionMode.ORDINARY, jnp.full(3, 0.1), curves(history, (9, 9, 9))))
    result = jax.jit(updater.update)(state, history, jnp.full(3, 0.1), 0, curves(history))
    raw = np.array([1, 3, 8])/3*0.1
    expected_target = project_reference(raw/raw.sum(), 0.05, 0.8)  # normalized first (eq. 17)
    np.testing.assert_allclose(result.marginal_efficiency, [1, 3, 8])
    np.testing.assert_allclose(result.target, expected_target)
    difference = 0.5*(expected_target-np.array([1/3]*3))
    expected = np.array([1/3]*3) + min(1, 0.05/(np.linalg.norm(difference)+np.finfo(float).eps))*difference
    np.testing.assert_allclose(result.state.values, expected)
    assert np.linalg.norm(result.state.values-state.values) <= 0.05 + 1e-14


def completed_history():
    history = fresh()
    # Two origins fail at horizon 1; one succeeds (H sentinel must not count).
    for origin, failed in [(0, True), (1, True), (2, False)]:
        history = accept(add_origin(history, origin, ActionMode.ORDINARY, jnp.full(3, 0.1)))
        for h in [1, 2, 3]:
            actual = failed and h == 1
            history = accept(resolve_horizon(history, origin, h, origin+h, actual,
                                              0.01 if actual else 0.2, True))
            outcome = resolve_trajectory(history, origin, origin+h)
            if outcome.accepted:
                history = outcome.state
    return history


def test_failure_aware_reduces_fragile_horizon_using_mature_cohort():
    history = completed_history()
    cfg = WeightConfig(kind="failure_aware", failure_window=3, eps_failure=0.1,
                       smoothing=1, max_step_norm=None, min_weight=0.05, max_weight=0.8)
    updater = build_weight_updater(cfg)
    state = updater.init_state(history)
    for t in range(5):
        result = updater.update(state, history, jnp.full(3, 0.1), t, curves(history, (3, 3, 3)))
        state = accept(result)
        assert not bool(result.ready)
    result = jax.jit(updater.update)(state, history, jnp.full(3, 0.1), 5, curves(history, (3, 3, 3)))
    state = accept(result)
    assert bool(result.ready)
    np.testing.assert_allclose(result.first_failure_mass, [2/3, 0, 0])
    raw = (0.1+1e-6)/(0.1 + np.array([2/3, 0, 0]))
    np.testing.assert_allclose(result.target, project_reference(raw/raw.sum(), 0.05, 0.8))
    assert float(state.values[0]) < float(state.values[1])
    np.testing.assert_allclose(np.sum(state.values), 1)


@pytest.mark.parametrize("kind", ["static", "marginal_efficiency", "failure_aware"])
def test_weight_invariants_rejections_scan_and_independent_batches(kind):
    history = fresh()
    cfg = WeightConfig(kind=kind, failure_window=3, max_step_norm=0.02)
    updater = build_weight_updater(cfg)
    initial = updater.init_state(history)
    bad = updater.update(initial, history, jnp.full(3, 0.1), 1, curves(history))
    assert int(bad.error) == UpdateError.OUT_OF_ORDER
    np.testing.assert_array_equal(bad.state.values, initial.values)

    def step(state, t):
        slopes = jnp.array([1.0, 3.0, 8.0])*(1 + jnp.sin(t))
        current = 10-slopes[:, None]*history.grid.values
        result = updater.update(state, history, jnp.full(3, 0.1), t, current)
        return result.state, (result.error, result.state.values, jnp.linalg.norm(result.state.values-state.values))

    eager = jax.lax.scan(step, initial, jnp.arange(20, dtype=jnp.int32))
    compiled = jax.jit(lambda: jax.lax.scan(step, initial, jnp.arange(20, dtype=jnp.int32)))()
    np.testing.assert_allclose(eager[1][1], compiled[1][1], atol=1e-13)
    np.testing.assert_array_equal(compiled[1][0], 0)
    values = np.asarray(compiled[1][1])
    assert np.all(values > 0)
    assert np.all(values >= cfg.min_weight-1e-13) and np.all(values <= cfg.max_weight+1e-13)
    np.testing.assert_allclose(values.sum(axis=-1), 1, atol=1e-13)
    assert np.max(compiled[1][2]) <= 0.02+1e-13
    duplicate = updater.update(compiled[0], history, jnp.full(3, 0.1), 19, curves(history))
    assert int(duplicate.error) == UpdateError.DUPLICATE
    stacked = jax.tree.map(lambda value: jnp.stack([value, value]), initial)
    result = jax.jit(jax.vmap(updater.update, in_axes=(0, None, None, None, 0)))(
        stacked, history, jnp.full(3, 0.1), 0, jnp.stack([curves(history), curves(history, (8, 3, 1))]))
    np.testing.assert_array_equal(result.error, 0)


def test_invalid_reference_rejected_and_partial_curves_are_not_zero_efficiency_claims():
    history = fresh()
    updater = build_weight_updater(WeightConfig(kind="marginal_efficiency"))
    state = updater.init_state(history)
    bad = updater.update(state, history, jnp.full(3, jnp.nan), 0, curves(history))
    assert int(bad.error) == UpdateError.INVALID_INPUT
    np.testing.assert_array_equal(bad.state.values, state.values)
    fallback = updater.update(state, history, jnp.full(3, 0.1), 0,
                              curves(history), jnp.array([True, False, True]))
    assert bool(fallback.used_static_fallback)
    np.testing.assert_array_equal(fallback.state.values, state.application)


def test_zero_step_cap_and_float32():
    history = fresh(jnp.float32)
    updater = build_weight_updater(WeightConfig(kind="marginal_efficiency", max_step_norm=0))
    state = updater.init_state(history)
    result = jax.jit(updater.update)(state, history, jnp.full(3, 0.1), 0, curves(history))
    state = accept(result)
    np.testing.assert_array_equal(state.values, state.application)
    assert state.values.dtype == jnp.float32
    np.testing.assert_allclose(np.sum(result.target), 1, atol=1e-6)
