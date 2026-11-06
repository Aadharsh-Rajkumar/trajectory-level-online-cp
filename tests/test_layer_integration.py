from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajectory_risk.history import ActionMode, resolve_horizon, resolve_trajectory
from trajectory_risk.runner import (
    ForecastInputs, build_feedback_kernels, build_preallocation_kernel,
    build_allocation_kernel, init_runner_state, record_issuance,
)
from trajectory_risk.dependence import build_overlap_credit_selector
from trajectory_risk.configs import RiskMode
from test_signals import config, accept, assert_tree_equal
from trajectory_risk.solver import SolverStatus


@pytest.mark.parametrize("controller", ["aci", "pid"])
@pytest.mark.parametrize("weight_kind", ["static", "marginal_efficiency", "failure_aware"])
def test_preallocation_layer_order_queue_accounting_and_extreme_reference(controller, weight_kind):
    cfg = config("learned")
    cfg.t_max = 10
    cfg.dependence.kind = "adaptive"
    cfg.dependence.windows = (1, 2)
    cfg.dependence.window = 2
    cfg.dependence.assumption = "unassumed"
    cfg.weights.kind = weight_kind
    cfg.weights.failure_window = 2
    cfg.controller.kind = controller
    cfg.controller.gamma = 0.1
    cfg.controller.kp, cfg.controller.ki, cfg.controller.kd = 0.03, 0.02, 0.005
    cfg.controller.integral_limit = 0.5
    cfg.safety.q_lower, cfg.safety.q_upper = -0.015, 0.005
    state = init_runner_state(cfg)
    before = jax.jit(build_preallocation_kernel(cfg))
    issue = jax.jit(build_feedback_kernels(cfg).issue)
    selector = jax.jit(build_overlap_credit_selector(RiskMode.CERTIFIED))
    modes, labels, reveals = [], [], []
    budget, integral, previous = cfg.controller.budget_init, 0.0, 0.0
    reference = np.asarray(state.alpha_reference)
    for t in range(13):
        for h in [1, 2, 3]:
            origin = t-h
            if not 0 <= origin < 10:
                continue
            fail = bool(labels[origin]) and h == 1
            level = (0.3 if modes[origin] == ActionMode.AGGRESSIVE_EXTREME else
                     0.01 if modes[origin] == ActionMode.CONSERVATIVE_EXTREME else
                     0.01 if fail else 0.3)
            history = accept(resolve_horizon(state.history, origin, h, t, fail, level, True))
            terminal = resolve_trajectory(history, origin, t)
            state = replace(state, history=terminal.state)
        current = ForecastInputs(10-jnp.array([1, 4, 2])[:, None]*state.history.grid.values,
                                 jnp.ones(3, bool), jnp.array([t/10]))
        old = state
        result = before(state, t, current)
        state = accept(result)
        signal = float(result.pre_issuance_increment)
        np.testing.assert_allclose(signal, result.packet.total + old.queued_issue_increment)
        if controller == "aci":
            budget = np.clip(budget + 0.1*signal, 0, 1)
        else:
            integral = np.clip(integral+signal, -0.5, 0.5)
            budget = np.clip(0.1 + 0.03*signal + 0.02*integral + 0.005*(signal-previous), 0, 1)
            previous = signal
        np.testing.assert_allclose(result.proposal, budget)
        assert int(state.controller.updates) == t+1
        assert int(state.dependence.last_time) == t
        assert int(state.utility.last_time) == t
        assert float(state.queued_issue_increment) == 0
        q = cfg.safety.eta_q * sum(cfg.beta-labels[k] for k in range(len(labels)) if reveals[k] <= t)
        np.testing.assert_allclose(state.accounting.q, q, atol=1e-14)
        np.testing.assert_allclose(selector(result.dependence).surface, 0)
        assert np.linalg.norm(state.utility.values-old.utility.values) <= cfg.weights.max_step_norm + 1e-13
        duplicate = before(state, t, current)
        assert not bool(duplicate.accepted)
        assert_tree_equal(duplicate.state, state)
        if t < 10:
            mode = int(result.safety.action_mode)
            modes.append(mode)
            fail = mode == ActionMode.AGGRESSIVE_EXTREME or (mode == ActionMode.ORDINARY and t % 4 != 0)
            labels.append(int(fail))
            reveals.append(t+(1 if fail else 3))
            alpha = jnp.array([0.1, 0.12, 0.14]) if mode == ActionMode.ORDINARY else None
            state = accept(record_issuance(state, t, mode, current, alpha))
            if mode == ActionMode.ORDINARY:
                reference = np.asarray(alpha)
            np.testing.assert_allclose(state.alpha_reference, reference)
            state = accept(issue(state, t))
    assert ActionMode.CONSERVATIVE_EXTREME in modes
    np.testing.assert_allclose(state.accounting.q, cfg.safety.eta_q*np.sum(cfg.beta-np.asarray(labels)))
    np.testing.assert_allclose(state.signal.cumulative_increment, np.sum(cfg.beta-np.asarray(labels)))


def test_preallocation_rejection_is_transactional_and_preserves_issue_queue():
    cfg = config("pending_valid")
    cfg.weights.kind = "marginal_efficiency"
    state = init_runner_state(cfg)
    before = build_preallocation_kernel(cfg)
    current = ForecastInputs(jnp.full((3, 4), jnp.nan), jnp.zeros(3, bool), jnp.zeros(1))
    state = accept(before(state, 0, current))
    state = accept(record_issuance(state, 0, ActionMode.ORDINARY, current, jnp.full(3, 0.1)))
    state = accept(build_feedback_kernels(cfg).issue(state, 0))
    corrupted = replace(state, alpha_reference=jnp.full(3, jnp.nan))
    result = jax.jit(before)(corrupted, 1, current)
    assert not bool(result.accepted)
    assert_tree_equal(result.state, corrupted)
    assert float(result.state.queued_issue_increment) == cfg.beta
    assert int(result.state.accounting.last_observe_time) == 0
    assert int(result.state.controller.last_time) == 0


@pytest.mark.parametrize("mode", [RiskMode.CERTIFIED, RiskMode.EXPERIMENTAL_PLUGIN])
def test_allocation_kernel_returns_checked_ordinary_allocation_without_issuing_it(mode):
    cfg = config("pending_valid")
    cfg.risk_mode = mode
    cfg.controller.budget_init = 0.3
    cfg.solver.mode, cfg.solver.max_sqp_iterations = "full_sqp", 30
    state = init_runner_state(cfg)
    current = ForecastInputs(jnp.full((3, 4), jnp.nan), jnp.zeros(3, bool), jnp.zeros(1))
    pre = build_preallocation_kernel(cfg)(state, 0, current)
    allocated = jax.jit(build_allocation_kernel(cfg, state.history))(pre)
    assert bool(allocated.accepted) and bool(allocated.result.feasible)
    assert bool(allocated.result.certified) == (mode == RiskMode.CERTIFIED)
    assert int(allocated.action_mode) == ActionMode.ORDINARY
    assert int(allocated.state.solver.solves) == 1
    assert int(allocated.state.history.next_origin_id) == 0
    assert float(allocated.state.queued_issue_increment) == 0
    np.testing.assert_array_equal(allocated.state.alpha_reference, state.alpha_reference)
    assert float(allocated.result.acceptance_risk) <= float(pre.safety.budget)+cfg.solver.feasibility_tol


@pytest.mark.parametrize("q_init,expected", [(-0.01, ActionMode.CONSERVATIVE_EXTREME),
                                           (0.01, ActionMode.AGGRESSIVE_EXTREME)])
def test_formal_safety_extremes_bypass_solver_and_preserve_warm_start(q_init, expected):
    cfg = config("pending_valid")
    cfg.safety.q_lower, cfg.safety.q_upper, cfg.safety.q_init = -0.01, 0.01, q_init
    state = init_runner_state(cfg)
    current = ForecastInputs(jnp.full((3, 4), jnp.nan), jnp.zeros(3, bool), jnp.zeros(1))
    pre = build_preallocation_kernel(cfg)(state, 0, current)
    allocated = jax.jit(build_allocation_kernel(cfg, state.history))(pre)
    assert bool(allocated.accepted)
    assert int(allocated.action_mode) == expected
    assert int(allocated.result.required_action) == expected
    assert int(allocated.result.status) == SolverStatus.SKIPPED_EXTREME
    assert np.isnan(allocated.result.alpha).all()
    assert_tree_equal(allocated.state.solver, state.solver)
    issued = accept(record_issuance(allocated.state, 0, expected, current))
    assert np.isnan(issued.history.issued_alpha[0]).all()
    np.testing.assert_array_equal(issued.alpha_reference, state.alpha_reference)


def test_ordinary_solver_failure_requests_safe_extreme_instead_of_unchecked_alpha():
    cfg = config("pending_valid")
    cfg.controller.budget_init = 0
    state = init_runner_state(cfg)
    current = ForecastInputs(jnp.full((3, 4), jnp.nan), jnp.zeros(3, bool), jnp.zeros(1))
    pre = build_preallocation_kernel(cfg)(state, 0, current)
    assert int(pre.safety.action_mode) == ActionMode.ORDINARY
    allocated = jax.jit(build_allocation_kernel(cfg, state.history))(pre)
    assert bool(allocated.accepted)
    assert not bool(allocated.result.feasible)
    assert int(allocated.action_mode) == ActionMode.CONSERVATIVE_EXTREME
    assert np.isnan(allocated.result.alpha).all()
    assert_tree_equal(allocated.state.solver, state.solver)
