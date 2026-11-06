from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajectory_risk.configs import DependenceConfig, RiskMode
from trajectory_risk.dependence import (
    build_dependence_estimator, build_overlap_credit_selector, fixed_share_update,
)
from trajectory_risk.history import (
    ActionMode, AlphaGrid, UpdateError, add_origin, init_history, resolve_horizon, resolve_trajectory,
)


def accept(result):
    assert int(result.error) == UpdateError.OK
    return result.state


def equal_trees(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        tolerance = (5e-6, 5e-7) if a.dtype == jnp.float32 else (2e-12, 2e-12)
        np.testing.assert_allclose(a, b, rtol=tolerance[0], atol=tolerance[1], equal_nan=True)


def grid_history(horizon=3, window=3):
    return init_history(AlphaGrid.create([0.05, 0.1, 0.2]), horizon,
                        max(horizon + window, horizon + 1), 0.05, 0.2)


def reference_experts(levels, missing, edges, t, windows):
    grid = np.array([0.05, 0.1, 0.2])
    experts = np.zeros((len(edges), len(windows), 3, 3))
    count = np.zeros((len(edges), len(windows)), int)
    gaps = np.zeros_like(count)
    for e, (s, r) in enumerate(edges):
        for j, window in enumerate(windows):
            origins = [k for k in range(t-r-window+1, t-r+1) if 0 <= k < len(levels)]
            valid = [k for k in origins if not (missing[k, s-1] or missing[k, r-1])]
            count[e, j] = len(valid)
            gaps[e, j] = len(origins) - len(valid)
            if valid:
                pairs = levels[valid][:, [s-1, r-1]]
                surfaces = (pairs[:, 0, None, None] < grid[None, :, None]) & (
                    pairs[:, 1, None, None] < grid[None, None, :])
                experts[e, j] = surfaces.mean(axis=0)
    return experts, count, gaps


def run_reference_comparison(kind, missing=None, assumption="stable_certified"):
    levels = np.array([[0.05, 0.1, 0.2], [0.0, 0.0, 0.0], [0.2, 0.05, 0.1],
                       [0.1, 0.2, 0.05], [0.0, 0.2, 0.0], [0.2, 0.0, 0.2]] * 2)
    missing = np.zeros_like(levels, bool) if missing is None else missing
    cfg = DependenceConfig(kind=kind, window=3, windows=(1, 3), meta_eta=0.3, delta=0.1,
                           assumption=assumption)
    estimator = build_dependence_estimator(cfg, 30)
    history = grid_history()
    state = estimator.init_state(history)
    edges = np.asarray(state.edges)
    weights = np.ones((3, 3, 2))
    pending = {}
    lifetime_missing = np.zeros(3, int)
    sigma = 1 / np.ceil(30 / edges[:, 1])
    snapshots, loss_log = [], []
    update, estimate = jax.jit(estimator.update), jax.jit(estimator.estimate)
    for t in range(len(levels)):
        for h in [1, 2, 3]:
            origin = t-h
            if origin < 0:
                continue
            history = accept(resolve_horizon(history, origin, h, t, levels[origin, h-1] < 0.1,
                                              levels[origin, h-1], not missing[origin, h-1]))
            terminal = resolve_trajectory(history, origin, t)
            if terminal.accepted:
                history = terminal.state
        before = state
        state = accept(update(state, history, t))
        result = estimate(state, history, t)
        assert int(result.error) == 0
        if kind == "adaptive":
            for e, (s, r) in enumerate(edges):
                origin = t-r
                if origin >= 0:
                    unavailable = missing[origin, s-1] or missing[origin, r-1]
                    lifetime_missing[e] += unavailable
                    if (origin, e) in pending and not unavailable:
                        prediction, pi = pending[origin, e]
                        pair = levels[origin, [s-1, r-1]]
                        x = (pair[0] < np.array([0.05, 0.1, 0.2]))[:, None] & (
                            pair[1] < np.array([0.05, 0.1, 0.2]))[None, :]
                        losses = ((prediction-x) ** 2).mean(axis=(-2, -1))
                        mixture_loss = ((np.sum(pi[:, None, None] * prediction, axis=0)-x)**2).mean()
                        np.testing.assert_allclose(result.expert_losses[e], losses)
                        np.testing.assert_allclose(result.mixture_losses[e], mixture_loss)
                        residue = origin % r
                        multiplied = weights[e, residue] * np.exp(-0.3 * losses)
                        shared = (1-sigma[e])*multiplied + sigma[e]*multiplied.sum()/2
                        weights[e, residue] = shared/shared.sum()
                        assert bool(result.loss_mask[e])
                    else:
                        assert not bool(result.loss_mask[e])
                        assert np.isnan(result.expert_losses[e]).all()
            np.testing.assert_allclose(state.weights, weights, atol=1e-13)
            experts, count, gaps = reference_experts(levels, missing, edges, t, (1, 3))
            pi = np.array([weights[e, t % r]/weights[e, t % r].sum() for e, (_, r) in enumerate(edges)])
            point = np.sum(pi[:, :, None, None] * experts, axis=1)
            ready = (count == [1, 3]).all(axis=-1)
            for e in range(3):
                if ready[e]:
                    pending[t, e] = (experts[e].copy(), pi[e].copy())
            np.testing.assert_allclose(result.expert_weights, pi)
            np.testing.assert_allclose(result.expert_surfaces, experts)
            np.testing.assert_array_equal(result.expert_sample_count, count)
            np.testing.assert_array_equal(result.lifetime_missing_count, lifetime_missing)
            # Only origin t-d_e's residue can change.
            for e, (_, delay) in enumerate(edges):
                for residue in range(3):
                    if residue != (t-delay) % delay:
                        np.testing.assert_array_equal(state.weights[e, residue], before.weights[e, residue])
            loss_log.append((np.asarray(result.expert_losses), np.asarray(result.mixture_losses), np.asarray(result.loss_mask)))
        elif kind == "fixed_window":
            experts, count, gaps = reference_experts(levels, missing, edges, t, (3,))
            point, ready = experts[:, 0], count[:, 0] == 3
            np.testing.assert_array_equal(result.sample_count, count[:, 0])
            np.testing.assert_array_equal(result.missing_count, gaps[:, 0])
        else:
            point = np.zeros((3, 3, 3))
            counts = np.zeros(3, int)
            missing_counts = np.zeros(3, int)
            for e, (s, r) in enumerate(edges):
                valid = [k for k in range(max(t-r+1, 0)) if not (missing[k, s-1] or missing[k, r-1])]
                counts[e] = len(valid)
                missing_counts[e] = max(t-r+1, 0)-len(valid)
                if valid:
                    pairs = levels[valid][:, [s-1, r-1]]
                    point[e] = ((pairs[:, 0, None, None] < np.array([0.05, 0.1, 0.2])[None, :, None]) & (
                        pairs[:, 1, None, None] < np.array([0.05, 0.1, 0.2])[None, None, :])).mean(axis=0)
            ready = counts > 0
            np.testing.assert_array_equal(result.sample_count, counts)
            np.testing.assert_array_equal(result.missing_count, missing_counts)
        np.testing.assert_allclose(result.point_surface, point, atol=1e-13)
        np.testing.assert_array_equal(result.ready, ready)
        if assumption == "unassumed":
            assert not np.asarray(result.certified).any()
            np.testing.assert_array_equal(result.lower_surface, 0)
        else:
            np.testing.assert_allclose(result.lower_surface,
                np.where(np.asarray(result.certified)[:, None, None], np.maximum(point - np.asarray(result.radius)[:, None, None], 0), 0))
        assert np.all(np.diff(np.asarray(result.point_surface), axis=-1) >= -1e-14)
        assert np.all(np.diff(np.asarray(result.point_surface), axis=-2) >= -1e-14)
        duplicate = update(state, history, t)
        assert int(duplicate.error) == UpdateError.DUPLICATE
        equal_trees(duplicate.state, state)
        history = accept(add_origin(history, t, ActionMode.ORDINARY, jnp.full(3, 0.1)))
        snapshots.append(result)
    equal_trees(result, estimator.estimate(state, history, t))
    return estimator, state, history, snapshots, loss_log


@pytest.mark.parametrize("kind", ["full_history", "fixed_window", "adaptive"])
def test_estimators_against_complete_numpy_references(kind):
    run_reference_comparison(kind)


@pytest.mark.parametrize("kind", ["full_history", "fixed_window", "adaptive"])
def test_missing_pairs_never_backfill_or_invent_brier_losses(kind):
    missing = np.zeros((12, 3), bool)
    missing[4, 1] = True
    _, state, _, snapshots, _ = run_reference_comparison(kind, missing)
    assert int(state.missing_count[0]) == 1
    assert not bool(snapshots[6].certified[0])
    if kind == "full_history":
        assert not bool(snapshots[-1].certified[0])
    else:
        assert bool(snapshots[-1].certified[0])  # Clean exact-origin window recovers.
    if kind == "adaptive":
        assert not bool(state.adaptive_regret_valid[0])


@pytest.mark.parametrize("kind", ["full_history", "fixed_window", "adaptive"])
def test_uncertified_point_estimates_and_credit_policy(kind):
    _, _, _, snapshots, _ = run_reference_comparison(kind, assumption="unassumed")
    result = snapshots[-1]
    certified = jax.jit(build_overlap_credit_selector(RiskMode.CERTIFIED))(result)
    plugin = jax.jit(build_overlap_credit_selector(RiskMode.EXPERIMENTAL_PLUGIN))(result)
    np.testing.assert_array_equal(certified.surface, 0)
    assert bool(certified.certified)
    assert np.asarray(certified.fallback_mask).all()
    np.testing.assert_allclose(plugin.surface, result.point_surface)
    assert not bool(plugin.certified)


def test_fixed_share_equations_exactly_and_no_normalization_before_sharing():
    weights = jnp.array([[1.0, 1.0, 1.0], [2.0, 0.1, 0.7]])
    losses = jnp.array([[0.0, 0.3, 1.0], [1.0, 0.2, 0.0]])
    shares = jnp.array([0.1, 0.25])
    multiplied = np.asarray(weights) * np.exp(-0.3 * np.asarray(losses))
    expected = (1-np.asarray(shares)[:, None])*multiplied + np.asarray(shares)[:, None]*multiplied.sum(axis=-1, keepdims=True)/3
    np.testing.assert_allclose(jax.jit(fixed_share_update)(weights, losses, 0.3, shares), expected, atol=1e-14)
    np.testing.assert_allclose(fixed_share_update(weights, losses, 0.3, 0.1),
        0.9*multiplied + 0.1*multiplied.sum(axis=-1, keepdims=True)/3)


def test_global_radii_edge_specific_shares_and_delayed_adaptive_regret():
    _, state, _, snapshots, log = run_reference_comparison("adaptive")
    np.testing.assert_array_equal(state.n_max, [15, 10, 10])
    np.testing.assert_allclose(state.share, [1/15, 1/10, 1/10])
    np.testing.assert_array_equal(state.residue_mask, [[True, True, False], [True, True, True], [True, True, True]])
    result = snapshots[-1]
    complexity = 9 * (2 + 3 + 3)
    radii = np.sqrt(np.array([2, 3, 3])[:, None] / (2*np.array([1, 3])[None, :]) * np.log(complexity*30**2/0.1))
    np.testing.assert_allclose(result.radius, np.sum(np.asarray(result.expert_weights)*radii, axis=-1))
    for e in range(3):
        active = [entry for entry in log if entry[2][e]]
        for start in range(len(active)):
            losses = np.array([entry[0][e] for entry in active[start:]])
            mixture = np.array([entry[1][e] for entry in active[start:]])
            penalty = state.delays[e] * (np.log(state.n_max[e]*2)+2)/0.3
            assert 0.7*mixture.sum() <= losses.sum(axis=0).min() + penalty + 1e-13


def test_full_history_radius_after_eviction():
    _, state, _, snapshots, _ = run_reference_comparison("full_history")
    n = np.asarray(state.sample_count)
    expected = np.sqrt(np.asarray(state.delays)/(2*n)*np.log(np.pi**2*72*n**2/(3*0.1)))
    np.testing.assert_allclose(snapshots[-1].radius, expected)
    assert n[0] > 6  # Ring capacity is 6; full-history counts are not truncated.


def test_no_lookahead_and_stale_estimate_rejection():
    history = grid_history()
    estimator = build_dependence_estimator(DependenceConfig(kind="full_history"), 30)
    state = accept(estimator.update(estimator.init_state(history), history, 0))
    history = accept(add_origin(history, 0, ActionMode.ORDINARY, jnp.full(3, 0.1)))
    history = accept(resolve_horizon(history, 0, 1, 1, True, 0.01, True))
    history = accept(resolve_horizon(history, 0, 2, 2, True, 0.01, True))
    state = accept(jax.jit(estimator.update)(state, history, 1))
    assert not np.asarray(state.sample_count).any()
    state = accept(estimator.update(state, history, 2))
    assert int(state.sample_count[0]) == 1
    stale = estimator.estimate(state, history, 1)
    assert int(stale.error) == UpdateError.OUT_OF_ORDER
    np.testing.assert_array_equal(stale.point_surface, 0)
    np.testing.assert_array_equal(stale.expert_weights, 0)
    assert not np.asarray(stale.certified).any()


@pytest.mark.parametrize("kind", ["full_history", "fixed_window", "adaptive"])
def test_single_horizon_empty_edge_set(kind):
    history = grid_history(horizon=1)
    estimator = build_dependence_estimator(DependenceConfig(kind=kind, window=2, windows=(1, 2)), 10)
    state = accept(jax.jit(estimator.update)(estimator.init_state(history), history, 0))
    result = jax.jit(estimator.estimate)(state, history, 0)
    assert result.point_surface.shape == (0, 3, 3)
    assert result.expert_weights.shape[0] == 0


def test_late_window_update_rejects_overwritten_outgoing_origin():
    history = grid_history(horizon=2, window=2)
    estimator = build_dependence_estimator(DependenceConfig(kind="fixed_window", window=2), 10)
    state = estimator.init_state(history)
    for t in range(5):
        for h in [1, 2]:
            origin = t-h
            if origin < 0:
                continue
            history = accept(resolve_horizon(history, origin, h, t, False, 0.2, True))
            outcome = resolve_trajectory(history, origin, t)
            if outcome.accepted:
                history = outcome.state
        if t == 4:
            history = accept(add_origin(history, t, ActionMode.ORDINARY, jnp.full(2, 0.1)))
            result = estimator.update(state, history, t)
            assert int(result.error) == UpdateError.UNSAFE_OVERWRITE
            equal_trees(result.state, state)
        else:
            state = accept(estimator.update(state, history, t))
            history = accept(add_origin(history, t, ActionMode.ORDINARY, jnp.full(2, 0.1)))


@pytest.mark.parametrize("kind", ["full_history", "fixed_window", "adaptive"])
def test_compiled_scan_nonzero_certificate_and_float32(kind):
    history = init_history(AlphaGrid.create(jnp.array([0.05, 0.1, 0.2], jnp.float32)),
                           2, 34, 0.05, 0.2)
    estimator = build_dependence_estimator(DependenceConfig(kind=kind, window=32, windows=(16, 32), delta=0.1), 50)
    initial = estimator.init_state(history)

    def step(carry, t):
        history, numerical = carry

        def resolve(index, history):
            h = index + 1
            origin = t-h

            def active(history):
                observed = resolve_horizon(history, origin, h, t, True, 0.0, True)
                return resolve_trajectory(observed.state, origin, t).state

            return jax.lax.cond(origin >= 0, active, lambda h: h, history)

        history = jax.lax.fori_loop(0, 2, resolve, history)
        update = estimator.update(numerical, history, t)
        estimate = estimator.estimate(update.state, history, t)
        issued = add_origin(history, t, ActionMode.ORDINARY, jnp.full(2, 0.1))
        return (issued.state, update.state), (update.error, estimate)

    run = lambda: jax.lax.scan(step, (history, initial), jnp.arange(40, dtype=jnp.int32))
    eager, compiled = run(), jax.jit(run)()
    equal_trees(eager, compiled)
    np.testing.assert_array_equal(compiled[1][0], 0)
    estimate = jax.tree.map(lambda value: value[-1], compiled[1][1])
    assert bool(estimate.certified[0])
    assert np.asarray(estimate.lower_surface).min() > 0
    assert estimate.point_surface.dtype == jnp.float32
    assert estimate.radius.dtype == jnp.float32
    np.testing.assert_array_equal(estimate.point_surface, 1)
