from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajectory_risk.configs import ExperimentConfig, SignalConfig
from trajectory_risk.history import (
    ActionMode, UpdateError, add_origin, resolve_horizon, resolve_trajectory,
)
from trajectory_risk.runner import (
    ForecastInputs, build_feedback_kernels, init_runner_state, record_issuance, require_accepted,
)
from trajectory_risk.signals import (
    CompletedWindowProbabilitySignal, LearnedProbabilitySignal, PendingValidSignal,
    RevealOnlySignal, SignalEvent, build_signal, logistic_probability, train_logistic,
)


def config(kind="learned", refresh=True):
    cfg = ExperimentConfig(t_max=30, beta=0.2)
    cfg.history.horizon = 3
    cfg.history.grid_size = 4
    cfg.history.grid_values = (0.05, 0.1, 0.2, 0.3)
    cfg.history.alpha_min = 0.05
    cfg.history.alpha_max = 0.2
    cfg.history.capacity = 8
    cfg.history.covariate_size = 1
    cfg.dependence.kind = "full_history"
    cfg.weights.kind = "static"
    cfg.signal = SignalConfig(kind=kind, window=3, refresh_pending=refresh,
                              prior_probability=0.4, learning_rate=0.5)
    return cfg


def accept(result):
    require_accepted(result)
    return result.state


def assert_tree_equal(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)


def inputs(history, origin):
    return ForecastInputs(jnp.full(history.set_size_grid.shape[1:], jnp.nan),
                          jnp.zeros(history.issued_alpha.shape[1], bool),
                          jnp.array([origin / 10], history.grid.values.dtype))


@pytest.mark.parametrize("kind,refresh", [
    ("reveal_only", True), ("pending_valid", True),
    ("completed_window", True), ("completed_window", False),
    ("learned", True), ("learned", False),
])
def test_each_signal_telescopes_per_origin_after_wraparound_and_drain(kind, refresh):
    cfg = config(kind, refresh)
    state = init_runner_state(cfg)
    kernels = build_feedback_kernels(cfg)
    observe, issue = jax.jit(kernels.observe), jax.jit(kernels.issue)
    totals = np.zeros(12)
    labels = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0])
    first_failure = [0, 1, 2, 3, 0, 0, 1, 2, 0, 3, 1, 0]
    modes = [ActionMode.ORDINARY] * 12
    modes[5] = ActionMode.CONSERVATIVE_EXTREME
    modes[6] = ActionMode.AGGRESSIVE_EXTREME
    expected_resolved, expected_q = 0, cfg.safety.q_init
    all_pre = []

    def accumulate(packet):
        assert packet.increments.shape == (4,)
        for origin, increment, active in zip(packet.origin_id, packet.increments, packet.event_mask):
            if active:
                totals[int(origin)] += float(increment)

    for t in range(15):
        for h in [1, 2, 3]:
            origin = t - h
            if not 0 <= origin < 12:
                continue
            actual = first_failure[origin] == h
            # Extremes deliberately have opposite counterfactual ordinary outcomes.
            level = 0.3 if modes[origin] == ActionMode.AGGRESSIVE_EXTREME else (
                    0.01 if modes[origin] == ActionMode.CONSERVATIVE_EXTREME else (0.01 if actual else 0.3))
            state = replace(state, history=accept(resolve_horizon(
                state.history, origin, h, t, actual, level, critical_available=origin % 3 != 0)))
            result = resolve_trajectory(state.history, origin, t)
            if result.accepted:
                state = replace(state, history=result.state)
                expected_resolved += 1
                expected_q += cfg.safety.eta_q * (cfg.beta - labels[origin])
        old_queue = float(state.queued_issue_increment)
        result = observe(state, t)
        state = accept(result)
        accumulate(result.packet)
        np.testing.assert_allclose(result.pre_issuance_increment, result.packet.total + old_queue)
        assert float(state.queued_issue_increment) == 0
        np.testing.assert_allclose(state.accounting.q, expected_q, atol=1e-14)
        assert int(state.accounting.resolved_count) == expected_resolved
        assert np.count_nonzero(state.signal.pending) <= 3
        all_pre.append(float(result.pre_issuance_increment))
        duplicate = observe(state, t)
        assert int(duplicate.error) == UpdateError.DUPLICATE
        assert_tree_equal(duplicate.state, state)
        assert float(duplicate.pre_issuance_increment) == 0

        if t < 12:
            alpha = jnp.array([0.1, 0.13, 0.17]) if modes[t] == ActionMode.ORDINARY else None
            state = accept(record_issuance(state, t, modes[t], inputs(state.history, t), alpha))
            result = issue(state, t)
            state = accept(result)
            accumulate(result.packet)
            assert float(result.pre_issuance_increment) == 0
            np.testing.assert_allclose(state.queued_issue_increment, result.packet.total)
            if modes[t] != ActionMode.ORDINARY:
                np.testing.assert_allclose(result.packet.total, cfg.beta - labels[t])
                np.testing.assert_array_equal(state.signal.features[t % 8], 0)

    np.testing.assert_allclose(totals, cfg.beta - labels, atol=1e-13)
    np.testing.assert_allclose(state.signal.cumulative_increment, np.sum(cfg.beta - labels), atol=1e-13)
    np.testing.assert_allclose(np.sum(all_pre), np.sum(cfg.beta - labels), atol=1e-13)
    np.testing.assert_allclose(state.accounting.q - cfg.safety.q_init,
                               cfg.safety.eta_q * np.sum(cfg.beta - labels), atol=1e-14)
    assert not np.asarray(state.signal.pending).any()
    assert int(state.signal.trained_examples) == (10 if kind == "learned" else 0)


@pytest.mark.parametrize("signal,expected", [(RevealOnlySignal(0.2), 0.0), (PendingValidSignal(0.2), 0.2)])
def test_issue_baselines_and_terminal_failure_sign(signal, expected):
    state = init_runner_state(config()).history
    numerical = signal.init_state(state)
    numerical = accept(signal.observe(numerical, state, 0))
    state = accept(add_origin(state, 0, ActionMode.ORDINARY, jnp.array([0.1] * 3),
                              covariates=jnp.zeros(1)))
    issued = signal.issue(numerical, state, 0)
    numerical = accept(issued)
    np.testing.assert_allclose(issued.packet.total, expected)
    assert bool(issued.packet.event_mask[-1]) == (expected != 0)
    state = accept(resolve_horizon(state, 0, 1, 1, True, 0.05, True))
    state = accept(resolve_trajectory(state, 0, 1))
    terminal = signal.observe(numerical, state, 1)
    np.testing.assert_allclose(terminal.packet.total, 0.2 - 1 - expected)


@pytest.mark.parametrize("refresh", [True, False])
def test_logistic_terminal_then_train_then_pending_refresh(refresh):
    cfg = config("learned", refresh)
    state = init_runner_state(cfg)
    signal = build_signal(cfg.signal, cfg.beta)
    for t in [0, 1]:
        if t == 1:
            state = replace(state, history=accept(resolve_horizon(state.history, 0, 1, 1, False, 0.2, True)))
        state = replace(state, signal=accept(signal.observe(state.signal, state.history, t)))
        state = accept(record_issuance(state, t, ActionMode.ORDINARY, inputs(state.history, t), jnp.array([0.1] * 3)))
        state = replace(state, signal=accept(signal.issue(state.signal, state.history, t)))
    old = state.signal
    state = replace(state, history=accept(resolve_horizon(state.history, 0, 2, 2, True, 0.05, True)))
    state = replace(state, history=accept(resolve_trajectory(state.history, 0, 2)))
    state = replace(state, history=accept(resolve_horizon(state.history, 1, 1, 2, False, 0.2, True)))
    outcome = jax.jit(signal.observe)(old, state.history, 2)
    require_accepted(outcome)
    x = np.asarray(old.features[0])
    probability = 1 / (1 + np.exp(-x @ np.asarray(old.theta)))
    expected_theta = np.asarray(old.theta) - 0.5 * (probability - 1) * x
    np.testing.assert_allclose(outcome.state.theta, expected_theta, atol=1e-14)
    origins = np.asarray(outcome.packet.origin_id)
    terminal_index = np.flatnonzero(origins == 0)[0]
    pending_index = np.flatnonzero(origins == 1)[0]
    np.testing.assert_allclose(outcome.packet.increments[terminal_index], old.provisional[0] - 1)
    expected_prediction = 1 / (1 + np.exp(-np.asarray(old.features[1]) @ expected_theta))
    expected_revision = old.provisional[1] - expected_prediction if refresh else 0
    np.testing.assert_allclose(outcome.packet.increments[pending_index], expected_revision)
    assert not bool(outcome.state.pending[0])
    assert float(outcome.state.provisional[0]) == 1
    assert int(outcome.state.trained_examples) == 1
    assert int(outcome.packet.event_kind[terminal_index]) == SignalEvent.TERMINAL
    if refresh:
        assert int(outcome.packet.event_kind[pending_index]) == SignalEvent.REFRESH
    else:
        np.testing.assert_array_equal(outcome.state.provisional[1], old.provisional[1])
    assert_tree_equal(outcome, signal.observe(old, state.history, 2))


def test_mean_batch_logistic_gradient_matches_numpy_and_masks_unrevealed_labels():
    theta = jnp.array([-0.4, 0.3])
    features = jnp.array([[1.0, 0.2], [1.0, 0.7], [1.0, 0.9]])
    labels = jnp.array([1, 0, 1])
    mask = jnp.array([True, True, False])
    probabilities = 1 / (1 + np.exp(-np.asarray(features[:2]) @ np.asarray(theta)))
    expected = np.asarray(theta) - 0.2 * np.asarray(features[:2]).T @ (probabilities - np.array([1, 0])) / 2
    updated, count = jax.jit(train_logistic)(theta, features, labels, mask, 0.2)
    np.testing.assert_allclose(updated, expected)
    assert int(count) == 2
    alternate, _ = train_logistic(theta, features, labels.at[2].set(0), mask, 0.2)
    np.testing.assert_array_equal(updated, alternate)
    unchanged, count = train_logistic(theta, features, labels, jnp.zeros(3, bool), 0.2)
    np.testing.assert_array_equal(unchanged, theta)
    assert int(count) == 0


def test_completed_window_ignores_early_failures_outside_mature_origin_cohort():
    cfg = config("completed_window")
    cfg.signal.window = 1
    signal = build_signal(cfg.signal, cfg.beta)
    history = init_runner_state(cfg).history
    numerical = signal.init_state(history)
    for t in range(4):
        for h in range(1, 4):
            origin = t - h
            if origin < 0:
                continue
            fail = origin == 1 and h == 1
            history = accept(resolve_horizon(history, origin, h, t, fail, 0.05 if fail else 0.2, True))
            terminal = resolve_trajectory(history, origin, t)
            if terminal.accepted:
                history = terminal.state
        numerical = accept(signal.observe(numerical, history, t))
        history = accept(add_origin(history, t, ActionMode.ORDINARY, jnp.array([0.1] * 3), covariates=jnp.zeros(1)))
        numerical = accept(signal.issue(numerical, history, t))
        expected = 0.4 if t < 3 else 0.0  # At t=3, ONLY origin 0 is mature.
        np.testing.assert_allclose(numerical.provisional[t % 8], expected)
    assert int(history.trajectory_label[1]) == 1


def test_no_lookahead_even_when_history_contains_future_terminal_label():
    cfg = config()
    runner = init_runner_state(cfg)
    signal = build_signal(cfg.signal, cfg.beta)
    numerical = accept(signal.observe(runner.signal, runner.history, 0))
    history = accept(add_origin(runner.history, 0, ActionMode.ORDINARY, jnp.array([0.1] * 3), covariates=jnp.zeros(1)))
    numerical = accept(signal.issue(numerical, history, 0))
    history = accept(resolve_horizon(history, 0, 1, 1, False, 0.2, True))
    history = accept(resolve_horizon(history, 0, 2, 2, True, 0.05, True))
    history = accept(resolve_trajectory(history, 0, 2))
    result = jax.jit(signal.observe)(numerical, history, 1)
    require_accepted(result)
    np.testing.assert_array_equal(result.state.theta, numerical.theta)
    assert not np.asarray(result.packet.resolved_mask).any()
    assert bool(result.state.pending[0])
    assert int(result.state.trained_examples) == 0


def test_model_features_are_frozen_at_issue_time():
    cfg = config()
    state = init_runner_state(cfg)
    signal = build_signal(cfg.signal, cfg.beta)
    numerical = accept(signal.observe(state.signal, state.history, 0))
    history = accept(add_origin(state.history, 0, ActionMode.ORDINARY, jnp.array([0.1] * 3), covariates=jnp.array([2.0])))
    numerical = accept(signal.issue(numerical, history, 0))
    changed = replace(history, issue_covariates=history.issue_covariates.at[0].set(999.0))
    numerical = accept(signal.observe(numerical, changed, 1))
    np.testing.assert_array_equal(numerical.features[0], [1, 0.1, 0.1, 0.1, 2])


def test_signal_operation_rejections_preserve_state():
    cfg = config()
    runner = init_runner_state(cfg)
    signal = build_signal(cfg.signal, cfg.beta)
    numerical = runner.signal
    bad = signal.observe(numerical, runner.history, 2)
    assert int(bad.error) == UpdateError.OUT_OF_ORDER
    assert_tree_equal(bad.state, numerical)
    bad = signal.issue(numerical, runner.history, 0)
    assert int(bad.error) == UpdateError.UNKNOWN_ORIGIN
    numerical = accept(signal.observe(numerical, runner.history, 0))
    history = accept(add_origin(runner.history, 0, ActionMode.ORDINARY, jnp.array([0.1] * 3), covariates=jnp.zeros(1)))
    numerical = accept(signal.issue(numerical, history, 0))
    bad = signal.issue(numerical, history, 0)
    assert int(bad.error) == UpdateError.DUPLICATE
    assert_tree_equal(bad.state, numerical)
    evicted = replace(history, origin_id=history.origin_id.at[0].set(8))
    bad = signal.observe(numerical, evicted, 1)
    assert int(bad.error) == UpdateError.UNSAFE_OVERWRITE
    assert_tree_equal(bad.state, numerical)


def test_logistic_prior_and_factory_validation():
    for prior in [0, 1, np.nan]:
        with pytest.raises(ValueError, match="prior_probability"):
            LearnedProbabilitySignal(0.2, prior_probability=prior)
    with pytest.raises(ValueError, match="unknown"):
        build_signal(SignalConfig(kind="unknown"), 0.2)
    with pytest.raises(ValueError, match="window"):
        CompletedWindowProbabilitySignal(0.2, window=0)


def test_issue_feedback_queue_is_consumed_once_and_duplicate_calls_preserve_it():
    cfg = config("pending_valid")
    state = init_runner_state(cfg)
    kernels = build_feedback_kernels(cfg)
    state = accept(kernels.observe(state, 0))
    state = accept(record_issuance(state, 0, ActionMode.ORDINARY, inputs(state.history, 0), jnp.array([0.1] * 3)))
    state = accept(kernels.issue(state, 0))
    for call, time in [(kernels.issue, 0), (kernels.observe, 0), (kernels.observe, 2)]:
        result = call(state, time)
        assert not bool(result.accepted)
        assert_tree_equal(result.state, state)
        np.testing.assert_allclose(result.state.queued_issue_increment, 0.2)
    result = kernels.observe(state, 1)
    np.testing.assert_allclose(result.pre_issuance_increment, 0.2)
    assert float(result.state.queued_issue_increment) == 0


def test_model_overflow_rejected_without_booking_nonfinite_feedback():
    cfg = config()
    signal = build_signal(cfg.signal, cfg.beta)
    runner = init_runner_state(cfg)
    state = accept(signal.observe(runner.signal, runner.history, 0))
    history = accept(add_origin(runner.history, 0, ActionMode.ORDINARY, jnp.array([0.1] * 3), covariates=jnp.zeros(1)))
    poisoned = replace(state, theta=state.theta.at[0].set(jnp.nan))
    result = jax.jit(signal.issue)(poisoned, history, 0)
    assert int(result.error) == UpdateError.INVALID_INPUT
    assert_tree_equal(result.state, poisoned)
    assert not np.asarray(result.packet.event_mask).any()
    state = accept(signal.issue(state, history, 0))
    history = accept(resolve_horizon(history, 0, 1, 1, True, 0.05, True))
    history = accept(resolve_trajectory(history, 0, 1))
    poisoned = replace(state, theta=state.theta.at[0].set(jnp.inf))
    result = jax.jit(signal.observe)(poisoned, history, 1)
    assert int(result.error) == UpdateError.INVALID_INPUT
    assert_tree_equal(result.state, poisoned)
    assert float(result.packet.total) == 0


def test_feedback_state_composes_in_compiled_scan_and_independent_batches():
    cfg = config()
    initial = init_runner_state(cfg)
    kernels = build_feedback_kernels(cfg)
    failures = jnp.array([0, 1, 2, 3, 0, 1, 3, 2, 0, 1, 0, 3], jnp.int32)

    def step(state, t):
        def horizon_step(index, carry):
            history, ok = carry
            h = index + 1
            origin = t - h

            def observe(carry):
                history, ok = carry
                actual = failures[jnp.clip(origin, 0, 11)] == h
                update = resolve_horizon(history, origin, h, t, actual,
                                          jnp.where(actual, 0.01, 0.3), True)
                terminal = resolve_trajectory(update.state, origin, t)
                expected = ((terminal.error == UpdateError.OK) |
                            (terminal.error == UpdateError.DUPLICATE) |
                            (terminal.error == UpdateError.NOT_YET_RESOLVED))
                return terminal.state, ok & update.accepted & expected

            return jax.lax.cond((origin >= 0) & (origin < 12), observe, lambda carry: carry, carry)

        history, ok = jax.lax.fori_loop(0, 3, horizon_step, (state.history, jnp.asarray(True)))
        feedback = kernels.observe(replace(state, history=history), t)

        def issue(state):
            recorded = record_issuance(state, t, ActionMode.ORDINARY, inputs(state.history, t), jnp.array([0.1] * 3))
            issued = kernels.issue(recorded.state, t)
            return issued.state, recorded.accepted & issued.accepted

        updated, issued_ok = jax.lax.cond(t < 12, issue, lambda state: (state, jnp.asarray(True)), feedback.state)
        return updated, (ok & feedback.accepted & issued_ok, updated.accounting.q, feedback.pre_issuance_increment)

    run = lambda state: jax.lax.scan(step, state, jnp.arange(15, dtype=jnp.int32))
    eager = run(initial)
    compiled = jax.jit(run)(initial)
    assert_tree_equal(eager, compiled)
    assert np.asarray(compiled[1][0]).all()
    labels = np.asarray(failures) > 0
    reveals = np.arange(12) + np.where(labels, np.asarray(failures), 3)
    expected_q = [cfg.safety.eta_q * np.sum(cfg.beta - labels[reveals <= t]) for t in range(15)]
    np.testing.assert_allclose(compiled[1][1], expected_q, atol=1e-14)
    np.testing.assert_allclose(np.sum(compiled[1][2]), np.sum(cfg.beta - labels), atol=1e-13)
    stacked = jax.tree.map(lambda value: jnp.stack([value, value]), initial)
    batched = jax.jit(jax.vmap(kernels.observe))(stacked, jnp.array([0, 0]))
    np.testing.assert_array_equal(batched.error, [UpdateError.OK, UpdateError.OK])


def test_float32_signal_kernels_preserve_numerical_precision():
    from trajectory_risk.history import AlphaGrid, init_history

    history = init_history(AlphaGrid.create(jnp.array([0.05, 0.1, 0.2], jnp.float32)),
                           2, 5, 0.05, 0.2)
    signal = LearnedProbabilitySignal(0.2)
    state = accept(jax.jit(signal.observe)(signal.init_state(history), history, 0))
    history = accept(add_origin(history, 0, ActionMode.ORDINARY, jnp.array([0.1, 0.1])))
    state = accept(jax.jit(signal.issue)(state, history, 0))
    history = accept(resolve_horizon(history, 0, 1, 1, True, 0.05, True))
    history = accept(resolve_trajectory(history, 0, 1))
    updated = jax.jit(signal.observe)(state, history, 1)
    require_accepted(updated)
    assert updated.state.theta.dtype == jnp.float32
    assert updated.state.cumulative_increment.dtype == jnp.float32
    assert updated.packet.increments.dtype == jnp.float32


def test_missing_actual_terminal_label_cannot_expire_a_signal_unnoticed():
    cfg = config()
    runner = init_runner_state(cfg)
    signal = build_signal(cfg.signal, cfg.beta)
    state = accept(signal.observe(runner.signal, runner.history, 0))
    history = accept(add_origin(runner.history, 0, ActionMode.ORDINARY, jnp.array([0.1] * 3), covariates=jnp.zeros(1)))
    state = accept(signal.issue(state, history, 0))
    for t in [1, 2]:
        state = accept(signal.observe(state, history, t))
    bad = jax.jit(signal.observe)(state, history, 3)
    assert int(bad.error) == UpdateError.INVALID_INPUT
    assert_tree_equal(bad.state, state)
