from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajectory_risk.configs import SafetyConfig
from trajectory_risk.controllers import ExactAccountingGuard
from trajectory_risk.configs import ControllerConfig
from trajectory_risk.controllers import (
    ACIController, PIDBudgetController, DtACIController, FixedShareGainController, build_controller,
)
from trajectory_risk.history import ActionMode, UpdateError, add_origin, resolve_horizon, resolve_trajectory
from trajectory_risk.runner import init_runner_state

from test_signals import config, accept, assert_tree_equal


def test_exact_accounting_multiple_arrivals_and_idempotence():
    history = init_runner_state(config()).history
    guard = ExactAccountingGuard(0.2, SafetyConfig(eta_q=0.1))
    state = guard.init_state(history)
    for t in range(4):
        for h in [1, 2, 3]:
            origin = t - h
            if origin < 0:
                continue
            fail = (origin == 1 and h == 2) or (origin == 2 and h == 1)
            history = accept(resolve_horizon(history, origin, h, t, fail,
                                             critical_available=False))
            terminal = resolve_trajectory(history, origin, t)
            if terminal.accepted:
                history = terminal.state
        result = jax.jit(guard.update)(state, history, t)
        state = accept(result)
        if t < 3:
            assert float(state.q) == 0
            history = accept(add_origin(history, t, ActionMode.ORDINARY, jnp.array([0.1] * 3),
                                         covariates=jnp.zeros(1)))
        else:
            np.testing.assert_array_equal(result.resolved.origin_id, [0, 1, 2])
            np.testing.assert_array_equal(result.resolved.labels, [0, 1, 1])
            assert np.asarray(result.resolved.mask).all()
    np.testing.assert_allclose(state.q, 0.1 * (3 * 0.2 - 2))
    assert int(state.resolved_count) == 3
    duplicate = guard.update(state, history, 3)
    assert int(duplicate.error) == UpdateError.DUPLICATE
    assert_tree_equal(duplicate.state, state)


@pytest.mark.parametrize("q,mode,budget", [
    (-1.0, ActionMode.CONSERVATIVE_EXTREME, 0),
    (-2.0, ActionMode.CONSERVATIVE_EXTREME, 0),
    (0.0, ActionMode.ORDINARY, 0.4),
    (1.0, ActionMode.AGGRESSIVE_EXTREME, 1),
    (2.0, ActionMode.AGGRESSIVE_EXTREME, 1),
])
def test_guard_boundaries_override_proposal_without_projecting_q(q, mode, budget):
    history = init_runner_state(config()).history
    guard = ExactAccountingGuard(0.2, SafetyConfig())
    state = replace(guard.init_state(history), q=jnp.asarray(q))
    result = jax.jit(guard.protect)(state, 0.4)
    assert int(result.error) == UpdateError.OK
    assert int(result.action_mode) == mode
    np.testing.assert_allclose(result.budget, budget)
    assert bool(result.overridden) == (mode != ActionMode.ORDINARY)
    assert float(state.q) == q


def test_invalid_ordinary_proposal_fails_closed_and_boundary_override_remains_available():
    history = init_runner_state(config()).history
    guard = ExactAccountingGuard(0.2, SafetyConfig())
    state = guard.init_state(history)
    for proposal in [np.nan, np.inf, -0.1, 1.1]:
        result = guard.protect(state, proposal)
        assert int(result.error) == UpdateError.INVALID_INPUT
        assert int(result.action_mode) == ActionMode.CONSERVATIVE_EXTREME
        assert float(result.budget) == 0
    result = guard.protect(replace(state, q=jnp.asarray(2.0)), np.nan)
    assert int(result.error) == UpdateError.OK
    assert int(result.action_mode) == ActionMode.AGGRESSIVE_EXTREME


def test_no_lookahead_and_out_of_order_accounting_rejection():
    history = init_runner_state(config()).history
    guard = ExactAccountingGuard(0.2, SafetyConfig())
    state = accept(guard.update(guard.init_state(history), history, 0))
    history = accept(add_origin(history, 0, ActionMode.ORDINARY, jnp.array([0.1] * 3), covariates=jnp.zeros(1)))
    history = accept(resolve_horizon(history, 0, 1, 1, False, 0.2, True))
    history = accept(resolve_horizon(history, 0, 2, 2, True, 0.05, True))
    history = accept(resolve_trajectory(history, 0, 2))
    result = guard.update(state, history, 1)
    assert float(result.state.q) == 0
    assert int(result.state.resolved_count) == 0
    bad = guard.update(state, history, 2)
    assert int(bad.error) == UpdateError.OUT_OF_ORDER
    assert_tree_equal(bad.state, state)
    result = guard.update(result.state, history, 2)
    np.testing.assert_allclose(result.state.q, -0.008)


def test_accounting_state_is_not_clipped_when_feedback_crosses_safety_boundary():
    history = init_runner_state(config()).history
    guard = ExactAccountingGuard(0.2, SafetyConfig(eta_q=10))
    state = accept(guard.update(guard.init_state(history), history, 0))
    history = accept(add_origin(history, 0, ActionMode.ORDINARY, jnp.array([0.1] * 3), covariates=jnp.zeros(1)))
    history = accept(resolve_horizon(history, 0, 1, 1, True, 0.05, True))
    history = accept(resolve_trajectory(history, 0, 1))
    state = accept(guard.update(state, history, 1))
    np.testing.assert_allclose(state.q, -8)
    assert int(guard.protect(state, 0.7).action_mode) == ActionMode.CONSERVATIVE_EXTREME


def test_safety_actions_bound_q_on_an_adversarial_stream_with_censored_feedback():
    history = init_runner_state(config()).history
    safety = SafetyConfig(eta_q=0.1, q_lower=-0.05, q_upper=0.05)
    guard = ExactAccountingGuard(0.2, safety)

    def step(carry, t):
        history, accounting = carry

        def horizon_step(index, carry):
            history, ok = carry
            h = index + 1
            origin = t - h

            def observed(carry):
                history, ok = carry
                mode = history.action_mode[jnp.mod(origin, 8)]
                failed = (mode == ActionMode.AGGRESSIVE_EXTREME) | (
                    (mode == ActionMode.ORDINARY) & (origin % 7 != 0))
                update = resolve_horizon(history, origin, h, t, failed & (h == 1),
                                          critical_available=False)
                terminal = resolve_trajectory(update.state, origin, t)
                expected = ((terminal.error == UpdateError.OK) |
                            (terminal.error == UpdateError.DUPLICATE) |
                            (terminal.error == UpdateError.NOT_YET_RESOLVED))
                return terminal.state, ok & update.accepted & expected

            return jax.lax.cond(origin >= 0, observed, lambda c: c, carry)

        history, ok = jax.lax.fori_loop(0, 3, horizon_step, (history, jnp.asarray(True)))
        update = guard.update(accounting, history, t)
        decision = guard.protect(update.state, 0.1)
        alpha = jnp.where(decision.action_mode == ActionMode.ORDINARY, jnp.full(3, 0.1), jnp.full(3, jnp.nan))
        issued = add_origin(history, t, decision.action_mode, alpha, covariates=jnp.zeros(1))
        latent_failure = (decision.action_mode == ActionMode.AGGRESSIVE_EXTREME) | (
            (decision.action_mode == ActionMode.ORDINARY) & (t % 7 != 0))
        return (issued.state, update.state), (
            ok & update.accepted & (decision.error == 0) & issued.accepted,
            update.state.q, decision.action_mode, latent_failure)

    (_, final), (ok, q, modes, labels) = jax.jit(lambda: jax.lax.scan(
        step, (history, guard.init_state(history)), jnp.arange(80, dtype=jnp.int32)))()
    assert np.asarray(ok).all()
    assert np.any(np.asarray(modes) == ActionMode.CONSERVATIVE_EXTREME)
    assert np.min(q) >= safety.q_lower - safety.eta_q * (1 - 0.2) * 3 - 1e-13
    assert np.max(q) <= safety.q_upper + safety.eta_q * 0.2 * 3 + 1e-13
    labels = np.asarray(labels)
    reveals = np.arange(80) + np.where(labels, 1, 3)
    resolved = reveals <= 79
    np.testing.assert_allclose(final.q, safety.eta_q * np.sum(0.2 - labels[resolved]), atol=1e-14)
    assert int(final.resolved_count) == np.count_nonzero(resolved)
    assert np.count_nonzero(~resolved) <= 3
    assert abs(np.mean(labels) - 0.2) <= abs(float(final.q)) / (safety.eta_q * 80) + 3 / 80 + 1e-13


def test_missing_actual_terminal_label_cannot_silently_exceed_horizon_delay():
    history = init_runner_state(config()).history
    guard = ExactAccountingGuard(0.2, SafetyConfig())
    state = accept(guard.update(guard.init_state(history), history, 0))
    history = accept(add_origin(history, 0, ActionMode.ORDINARY, jnp.array([0.1] * 3), covariates=jnp.zeros(1)))
    for t in [1, 2]:
        state = accept(guard.update(state, history, t))
    bad = jax.jit(guard.update)(state, history, 3)
    assert int(bad.error) == UpdateError.INVALID_INPUT
    assert_tree_equal(bad.state, state)


def test_aci_signs_and_aggregate_projection_is_order_independent():
    controller = ACIController(ControllerConfig(budget_init=0.5, gamma=0.1))
    initial = controller.init_state()
    failed = accept(jax.jit(controller.update)(initial, 0.2-1, 0))
    succeeded = accept(controller.update(initial, 0.2, 0))
    assert float(failed.budget) < float(initial.budget) < float(succeeded.budget)
    controller = ACIController(ControllerConfig(budget_init=0.99, gamma=0.1))
    for events in [[0.2, -0.8], [-0.8, 0.2]]:
        result = controller.update(controller.init_state(), np.sum(events), 0)
        np.testing.assert_allclose(controller.propose_budget(result.state), 0.93)
        assert int(result.state.updates) == 1
    with pytest.raises(ValueError, match="aggregate scalar"):
        controller.update(controller.init_state(), jnp.array([0.2, -0.8]), 0)


@pytest.mark.parametrize("limit", [None, 0.5])
def test_pid_matches_physical_time_numpy_reference_including_empty_rounds(limit):
    cfg = ControllerConfig(kind="pid", budget_init=0.5, kp=0.1, ki=0.2, kd=0.3, integral_limit=limit)
    controller = build_controller(cfg)
    state = controller.init_state()
    integral, previous = 0.0, 0.0
    errors = [-0.8, 0.4, 0.0, 0.0, -1.0, 0.2]
    for t, error in enumerate(errors):
        old = state
        eager = controller.update(old, error, t)
        compiled = jax.jit(controller.update)(old, error, t)
        assert_tree_equal(eager, compiled)
        state = accept(compiled)
        integral += error
        if limit is not None:
            integral = np.clip(integral, -limit, limit)
        expected = np.clip(0.5 + 0.1*error + 0.2*integral + 0.3*(error-previous), 0, 1)
        np.testing.assert_allclose(state.budget, expected)
        np.testing.assert_allclose(state.integral, integral)
        previous = error
    assert int(state.updates) == len(errors)


@pytest.mark.parametrize("kind", ["aci", "pid"])
def test_controller_rejections_saturation_scan_and_batching(kind):
    controller = build_controller(ControllerConfig(kind=kind, gamma=0.5, kp=0.5, ki=0.1))
    initial = controller.init_state()
    for value, time, error in [(np.nan, 0, UpdateError.INVALID_INPUT), (0.1, 1, UpdateError.OUT_OF_ORDER)]:
        result = controller.update(initial, value, time)
        assert int(result.error) == error
        assert_tree_equal(initial, result.state)
    state = accept(controller.update(initial, -1.0, 0))
    duplicate = controller.update(state, 0.5, 0)
    assert int(duplicate.error) == UpdateError.DUPLICATE
    assert_tree_equal(state, duplicate.state)
    increments = jnp.array([-3.0, 0.0, 2.0, 0.0, -1.0])

    def step(state, item):
        time, increment = item
        result = controller.update(state, increment, time)
        return result.state, (result.error, result.state.budget)

    final, (errors, budgets) = jax.jit(lambda: jax.lax.scan(step, initial,
        (jnp.arange(5, dtype=jnp.int32), increments)))()
    np.testing.assert_array_equal(errors, 0)
    assert np.asarray((budgets >= 0) & (budgets <= 1)).all()
    assert int(final.updates) == 5
    batched = jax.jit(jax.vmap(controller.update, in_axes=(None, 0, None)))(initial, jnp.array([-0.8, 0.2]), 0)
    np.testing.assert_array_equal(batched.error, 0)
    assert float(batched.state.budget[0]) < float(batched.state.budget[1])


@pytest.mark.parametrize("constructor,message", [(FixedShareGainController, "counterfactual")])
def test_deferred_controller_interfaces_raise_informative_errors(constructor, message):
    with pytest.raises(NotImplementedError, match=message):
        constructor(ControllerConfig())


def test_float32_controller_state_stays_float32():
    for constructor in [ACIController, PIDBudgetController]:
        controller = constructor(ControllerConfig())
        result = jax.jit(controller.update)(controller.init_state(jnp.float32), -0.8, 0)
        assert result.state.budget.dtype == jnp.float32
        assert result.state.integral.dtype == jnp.float32
