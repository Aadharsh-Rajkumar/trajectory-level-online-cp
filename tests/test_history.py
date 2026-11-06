import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajectory_risk.history import (
    ActionMode, AlphaGrid, UpdateError, add_origin, edge_critical_levels,
    first_failure_histogram, init_history, lower_orthant_indicators,
    marginal_set_efficiency, recent_completed_origins, resolve_horizon,
    resolve_trajectory, set_size_curve,
)
from trajectory_risk.runner import ForecastInputs, RunnerState, record_issuance, require_accepted


def fresh(horizon=2, capacity=6):
    return init_history(AlphaGrid.create([0.05, 0.1, 0.2]), horizon, capacity, 0.05, 0.2)


def accept(update):
    require_accepted(update)
    return update.state


def assert_tree_equal(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("values", [[0.1], [0.1, 0.1], [0, 0.2], [0.2, 1], [0.1, np.nan]])
def test_grid_validation(values):
    with pytest.raises(ValueError):
        AlphaGrid.create(values)


def test_strict_grid_ties_and_batched_indicators():
    grid = fresh().grid
    pairs = jnp.array([[0.1, 0.1], [0.05, 0.2], [-jnp.inf, jnp.inf]])
    expected = (np.asarray(pairs)[:, 0, None, None] < np.asarray(grid.values)[None, :, None]) & (
        np.asarray(pairs)[:, 1, None, None] < np.asarray(grid.values)[None, None, :])
    eager = lower_orthant_indicators(pairs, grid)
    np.testing.assert_array_equal(eager, expected)
    np.testing.assert_array_equal(jax.jit(lower_orthant_indicators)(pairs, grid), expected)
    np.testing.assert_array_equal(jax.vmap(lambda pair: lower_orthant_indicators(pair, grid))(pairs), expected)


@pytest.mark.parametrize("compiled", [False, True])
def test_actual_tie_is_containment_and_updates_match(compiled):
    add = jax.jit(add_origin) if compiled else add_origin
    resolve = jax.jit(resolve_horizon) if compiled else resolve_horizon
    terminal = jax.jit(resolve_trajectory) if compiled else resolve_trajectory
    state = accept(add(fresh(), 0, ActionMode.ORDINARY, jnp.array([0.1, 0.1])))
    state = accept(resolve(state, 0, 1, 1, False, 0.1, True))
    pending = terminal(state, 0, 1)
    assert int(pending.error) == UpdateError.NOT_YET_RESOLVED
    state = accept(resolve(state, 0, 2, 2, False, 0.1, True))
    result = terminal(state, 0, 2)
    assert bool(result.newly_resolved)
    state = accept(result)
    assert int(state.trajectory_label[0]) == 0
    assert int(state.first_failure[0]) == 2
    repeat = terminal(state, 0, 2)
    assert int(repeat.error) == UpdateError.DUPLICATE
    assert not bool(repeat.newly_resolved)
    assert_tree_equal(state, repeat.state)


def test_eager_and_jit_state_equality():
    state = fresh()
    args = (0, ActionMode.ORDINARY, jnp.array([0.1, 0.1]))
    eager = add_origin(state, *args)
    compiled = jax.jit(add_origin)(state, *args)
    assert_tree_equal(eager, compiled)
    eager_h = resolve_horizon(eager.state, 0, 1, 1, True, 0.05, True)
    compiled_h = jax.jit(resolve_horizon)(compiled.state, 0, 1, 1, True, 0.05, True)
    assert_tree_equal(eager_h, compiled_h)
    assert_tree_equal(resolve_trajectory(eager_h.state, 0, 1),
                      jax.jit(resolve_trajectory)(compiled_h.state, 0, 1))


def test_rejected_observations_leave_state_unchanged():
    state = accept(add_origin(fresh(), 0, ActionMode.ORDINARY, jnp.array([0.1, 0.1])))
    attempts = [
        (resolve_horizon(state, 0, 1, 0, False, 0.2, True), UpdateError.WRONG_REVEAL_TIME),
        (resolve_horizon(state, 0, 1, 2, False, 0.2, True), UpdateError.WRONG_REVEAL_TIME),
        (resolve_horizon(state, 0, 2, 2, False, 0.2, True), UpdateError.OUT_OF_ORDER),
        (resolve_horizon(state, 0, 0, 0, False, 0.2, True), UpdateError.INVALID_INPUT),
        (resolve_horizon(state, 7, 1, 8, False, 0.2, True), UpdateError.UNKNOWN_ORIGIN),
        (resolve_horizon(state, 0, 1, 1, True, 0.1, True), UpdateError.INVALID_INPUT),
        (resolve_horizon(state, 0, 1, 1, False, np.nan, True), UpdateError.INVALID_INPUT),
    ]
    for result, error in attempts:
        assert int(result.error) == error
        assert_tree_equal(result.state, state)
    state = accept(resolve_horizon(state, 0, 1, 1, False, 0.2, True))
    duplicate = jax.jit(resolve_horizon)(state, 0, 1, 1, False, 0.2, True)
    assert int(duplicate.error) == UpdateError.DUPLICATE
    assert_tree_equal(duplicate.state, state)


@pytest.mark.parametrize("mode,violation,ordinary_level,label", [
    (ActionMode.CONSERVATIVE_EXTREME, False, -np.inf, 0),
    (ActionMode.AGGRESSIVE_EXTREME, True, np.inf, 1),
])
def test_extreme_labels_do_not_come_from_counterfactual_levels(mode, violation, ordinary_level, label):
    state = accept(jax.jit(add_origin)(fresh(), 0, mode))
    assert np.isnan(state.issued_alpha[0]).all()
    state = accept(jax.jit(resolve_horizon)(state, 0, 1, 1, violation, ordinary_level, True))
    if mode == ActionMode.CONSERVATIVE_EXTREME:
        pending = resolve_trajectory(state, 0, 1)
        assert int(pending.error) == UpdateError.NOT_YET_RESOLVED
        state = accept(resolve_horizon(state, 0, 2, 2, False, ordinary_level, True))
        state = accept(resolve_trajectory(state, 0, 2))
    else:
        state = accept(resolve_trajectory(state, 0, 1))
        state = accept(resolve_horizon(state, 0, 2, 2, False, ordinary_level, True))
        repeat = resolve_trajectory(state, 0, 2)
        assert int(repeat.error) == UpdateError.DUPLICATE
    assert int(state.trajectory_label[0]) == label
    pair = edge_critical_levels(state, jnp.array([0]), jnp.array([[1, 2]]), 2)
    assert bool(pair.mask[0, 0])
    assert np.all(np.asarray(pair.levels) == ordinary_level)


def test_extreme_inconsistent_membership_and_fabricated_alpha_rejected():
    state = fresh()
    for mode in [ActionMode.CONSERVATIVE_EXTREME, ActionMode.AGGRESSIVE_EXTREME]:
        result = add_origin(state, 0, mode, jnp.array([0.1, 0.1]))
        assert int(result.error) == UpdateError.INVALID_INPUT
    state = accept(add_origin(state, 0, ActionMode.CONSERVATIVE_EXTREME))
    assert int(resolve_horizon(state, 0, 1, 1, True).error) == UpdateError.INVALID_INPUT


def stream_state(steps=12, capacity=5):
    """Origin 3 succeeds, others fail at horizon 1; critical pairs still complete."""
    state = fresh(capacity=capacity)
    add = jax.jit(add_origin)
    observe = jax.jit(resolve_horizon)
    terminal = jax.jit(resolve_trajectory)
    for t in range(steps):
        for h in [1, 2]:
            origin = t - h
            if origin < 0:
                continue
            fail = h == 1 and origin != 3
            state = accept(observe(state, origin, h, t, fail, 0.05 if fail else 0.2, True))
            result = terminal(state, origin, t)
            if result.accepted:
                state = result.state
        state = accept(add(state, t, ActionMode.ORDINARY, jnp.array([0.1, 0.1])))
    return state


def test_ring_wraparound_and_origin_cohort_excludes_early_failures():
    state = stream_state()
    cohort = jax.jit(recent_completed_origins, static_argnames=("window",))(state, 11, 3)
    np.testing.assert_array_equal(cohort.origin_id, [7, 8, 9])
    assert bool(cohort.ready)
    np.testing.assert_array_equal(state.origin_id[cohort.slots], cohort.origin_id)
    assert bool(state.trajectory_resolved[10 % 5])  # Early failure exists but is outside cohort.
    assert_tree_equal(cohort, recent_completed_origins(state, 11, 3))
    stale = edge_critical_levels(state, jnp.array([1]), jnp.array([[1, 2]]), 11)
    assert not bool(stale.mask[0, 0])
    assert bool(stale.missing_mask[0, 0])


def test_incomplete_cohort_and_no_reveal_order_backfill():
    state = stream_state(steps=4, capacity=6)
    cohort = recent_completed_origins(state, 3, 3)
    np.testing.assert_array_equal(cohort.origin_id, [-1, 0, 1])
    np.testing.assert_array_equal(cohort.mask, [False, True, True])
    assert not bool(cohort.ready)
    early = edge_critical_levels(state, jnp.array([2]), jnp.array([[1, 2]]), 3)
    assert bool(early.pending_mask[0, 0])
    assert not bool(early.mask[0, 0])


def test_missing_critical_levels_are_explicit_and_never_create_fake_events():
    state = accept(add_origin(fresh(), 0, ActionMode.ORDINARY, jnp.array([0.1, 0.1])))
    state = accept(resolve_horizon(state, 0, 1, 1, False, 0.2, True))
    state = accept(resolve_horizon(state, 0, 2, 2, True, critical_available=False))
    state = accept(resolve_trajectory(state, 0, 2))
    assert int(state.trajectory_label[0]) == 1
    np.testing.assert_array_equal(state.missing_critical_count, [0, 1])
    assert bool(state.critical_missing[0, 1])
    pair = jax.jit(edge_critical_levels)(state, jnp.array([0]), jnp.array([[1, 2]]), 2)
    assert not bool(pair.mask[0, 0])
    assert bool(pair.missing_mask[0, 0])
    assert np.isnan(pair.levels).all()


def test_queries_do_not_reveal_future_data_from_later_state():
    state = stream_state(steps=5, capacity=6)
    pair = edge_critical_levels(state, jnp.array([2]), jnp.array([[1, 2]]), 3)
    assert not bool(pair.mask[0, 0])
    curve, mask = set_size_curve(state, 4, 3)
    assert not np.asarray(mask).any()
    assert np.isnan(curve).all()


def test_success_sentinel_not_counted_as_last_horizon_failure():
    state = fresh()
    for origin, last_fail in [(0, False), (1, True)]:
        state = accept(add_origin(state, origin, ActionMode.ORDINARY, jnp.array([0.1, 0.1])))
        state = accept(resolve_horizon(state, origin, 1, origin + 1, False, 0.2, True))
        state = accept(resolve_horizon(state, origin, 2, origin + 2, last_fail,
                                       0.05 if last_fail else 0.2, True))
        state = accept(resolve_trajectory(state, origin, origin + 2))
    hist, ready = jax.jit(first_failure_histogram, static_argnames=("window",))(state, 3, 2)
    assert bool(ready)
    np.testing.assert_array_equal(hist, [0, 0.5])


def test_unsafe_overwrite_rejected_even_after_early_failure():
    state = fresh(capacity=3)
    for origin in range(3):
        state = accept(add_origin(state, origin, ActionMode.ORDINARY, jnp.array([0.1, 0.1])))
    state = accept(resolve_horizon(state, 0, 1, 1, True, 0.05, True))
    state = accept(resolve_trajectory(state, 0, 1))
    result = jax.jit(add_origin)(state, 3, ActionMode.ORDINARY, jnp.array([0.1, 0.1]))
    assert int(result.error) == UpdateError.UNSAFE_OVERWRITE
    assert_tree_equal(state, result.state)


def test_size_queries_and_current_curve_efficiency():
    state = fresh()
    grid = state.grid
    curves = 2.0 - jnp.array([2.0, 5.0])[:, None] * grid.values
    state = accept(add_origin(state, 0, ActionMode.ORDINARY, jnp.array([0.1, 0.1]), curves))
    retrieved, mask = jax.jit(set_size_curve)(state, 0, 0)
    np.testing.assert_allclose(retrieved, curves)
    np.testing.assert_array_equal(mask, [True, True])
    current = 3.0 - jnp.array([1.0, 3.0])[:, None] * grid.values
    efficiency, available = jax.jit(marginal_set_efficiency)(grid, current, jnp.array([0.05, 0.2]))
    np.testing.assert_allclose(efficiency, [1.0, 3.0])
    assert np.asarray(available).all()
    invalid = current.at[0, 0].set(jnp.nan)
    efficiency, available = marginal_set_efficiency(grid, invalid, jnp.array([0.1, 0.1]))
    np.testing.assert_allclose(efficiency, [0, 3])
    np.testing.assert_array_equal(available, [False, True])


def test_extremes_preserve_reference_and_explicit_runner_queue():
    history = fresh()
    state = RunnerState(history, jnp.array([0.05, 0.05]), jnp.asarray(0.123), jnp.asarray(-1))
    inputs = ForecastInputs(jnp.full((2, 3), jnp.nan), jnp.zeros(2, bool), jnp.empty(0))
    state = accept(jax.jit(record_issuance)(state, 0, ActionMode.ORDINARY, inputs, jnp.array([0.1, 0.2])))
    state = accept(jax.jit(record_issuance)(state, 1, ActionMode.CONSERVATIVE_EXTREME, inputs))
    state = accept(jax.jit(record_issuance)(state, 2, ActionMode.AGGRESSIVE_EXTREME, inputs))
    np.testing.assert_array_equal(state.alpha_reference, [0.1, 0.2])
    assert float(state.queued_issue_increment) == 0.123
    assert np.isnan(state.history.issued_alpha[1:3]).all()


def test_compiled_scan_matches_origin_stream_reference():
    def step(state, t):
        def observe(index, carry):
            history, ok = carry
            h = index + 1
            origin = t - h

            def active(carry):
                history, ok = carry
                fail = (h == 1) & (origin != 3)
                update = resolve_horizon(history, origin, h, t, fail,
                                          jnp.where(fail, 0.05, 0.2), True)
                terminal = resolve_trajectory(update.state, origin, t)
                expected = ((terminal.error == UpdateError.OK) |
                            (terminal.error == UpdateError.DUPLICATE) |
                            (terminal.error == UpdateError.NOT_YET_RESOLVED))
                return terminal.state, ok & update.accepted & expected

            return jax.lax.cond(origin >= 0, active, lambda c: c, carry)

        history, ok = jax.lax.fori_loop(0, 2, observe, (state, jnp.asarray(True)))
        issued = add_origin(history, t, ActionMode.ORDINARY, jnp.array([0.1, 0.1]))
        return issued.state, ok & issued.accepted

    scan = jax.jit(lambda initial: jax.lax.scan(step, initial, jnp.arange(12, dtype=jnp.int32)))
    state, accepted = scan(fresh(capacity=5))
    assert np.asarray(accepted).all()
    assert_tree_equal(state, stream_state(steps=12, capacity=5))


def test_history_update_batches_under_vmap():
    state = fresh()
    # Two independent histories, one valid issuance and one rejected origin jump.
    states = jax.tree.map(lambda value: jnp.stack([value, value]), state)
    updates = jax.jit(jax.vmap(add_origin))(
        states, jnp.array([0, 2]), jnp.array([0, 0]), jnp.array([[0.1, 0.1], [0.1, 0.1]]))
    np.testing.assert_array_equal(updates.error, [UpdateError.OK, UpdateError.OUT_OF_ORDER])
    rejected = jax.tree.map(lambda value: value[1], updates.state)
    assert_tree_equal(rejected, state)


def test_missing_counts_survive_ring_eviction():
    state = fresh(capacity=3)
    for t in range(6):
        for h in [1, 2]:
            origin = t - h
            if origin < 0:
                continue
            available = not (origin == 0 and h == 2)
            state = accept(resolve_horizon(state, origin, h, t, False, 0.2, available))
            terminal = resolve_trajectory(state, origin, t)
            if terminal.accepted:
                state = terminal.state
        state = accept(add_origin(state, t, ActionMode.ORDINARY, jnp.array([0.1, 0.1])))
    assert not np.any(np.asarray(state.origin_id) == 0)
    np.testing.assert_array_equal(state.missing_critical_count, [0, 1])
