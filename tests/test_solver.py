from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajectory_risk.configs import DependenceConfig, RiskMode, SolverConfig
from trajectory_risk.dependence import build_dependence_estimator
from trajectory_risk.history import AlphaGrid, UpdateError, init_history
from trajectory_risk.solver import (
    ADMMConfig, QPStatus, bilinear_credit, build_risk_model, dense_admm_qp, init_qp_state,
    lower_grid_credit, maximum_spanning_forest, monotone_lower_grid,
    SQPConfig, SolverStatus, acceptance_feasible, allocation_cost, build_solver, restore_feasibility,
)


def frozen(horizon=3, mode=RiskMode.CERTIFIED, point=None, lower=None, config=None,
           grid_values=(0.01, 0.1, 0.2, 0.4), reference=None):
    history = init_history(AlphaGrid.create(grid_values), horizon, horizon+3, grid_values[0], grid_values[-1])
    estimator = build_dependence_estimator(DependenceConfig(kind="full_history"), 100)
    numerical = estimator.update(estimator.init_state(history), history, 0).state
    estimate = estimator.estimate(numerical, history, 0)
    shape = estimate.point_surface.shape
    point = jnp.zeros(shape) if point is None else jnp.broadcast_to(jnp.asarray(point), shape)
    lower = point if lower is None else jnp.broadcast_to(jnp.asarray(lower), shape)
    estimate = replace(estimate, point_surface=point, lower_surface=lower,
                       certified=jnp.ones(shape[0], bool), ready=jnp.ones(shape[0], bool))
    builder = build_risk_model(SolverConfig() if config is None else config, mode, history)
    reference = jnp.full(horizon, 0.1) if reference is None else jnp.asarray(reference)
    return builder(numerical, estimate, reference), (history, numerical, estimate, builder, reference)


def test_maximum_spanning_forest_and_lexicographic_ties_independent_of_input_order():
    edges = jnp.array([[2, 3], [1, 3], [1, 2]], jnp.int32)
    selected = jax.jit(maximum_spanning_forest, static_argnames=("horizon",))(edges, jnp.zeros(3), 3)
    np.testing.assert_array_equal(selected, [False, True, True])
    np.testing.assert_array_equal(maximum_spanning_forest(edges, jnp.array([3.0, 2.0, 1.0]), 3), [True, True, False])
    sparse = jnp.array([[1, 2], [3, 4]], jnp.int32)
    np.testing.assert_array_equal(maximum_spanning_forest(sparse, jnp.zeros(2), 4), [True, True])


def test_exact_southwest_extension_including_boundaries_and_below_grid():
    grid = AlphaGrid.create([0.05, 0.1, 0.2])
    surface = jnp.array([[[0.01, 0.02, 0.03], [0.03, 0.02, 0.08], [0.02, 0.1, 0.09]]])
    monotone = jax.jit(monotone_lower_grid)(surface)
    edge = jnp.array([[1, 2]], jnp.int32)
    for a, b in [(0.04, 0.2), (0.05, 0.1), (0.15, 0.15), (0.2, 0.2), (0.3, 0.3)]:
        corners = [(i, j) for i, x in enumerate(grid.values) for j, y in enumerate(grid.values) if x <= a and y <= b]
        expected = max([float(surface[0, i, j]) for i, j in corners], default=0)
        actual = jax.jit(lower_grid_credit)(grid, monotone, edge, jnp.array([a, b]))
        np.testing.assert_allclose(actual, [expected])


def test_bilinear_model_can_be_optimistic_and_never_replaces_certified_acceptance():
    grid = np.array([0.01, 0.1, 0.2, 0.4])
    values = grid[:, None]*grid[None, :]
    model, _ = frozen(horizon=2, point=values)
    alpha = jnp.array([0.15, 0.15])
    np.testing.assert_allclose(model.risk_model_for_sqp(alpha), 0.3-0.15**2)
    np.testing.assert_allclose(model.certified_risk(alpha), 0.3-0.1**2)
    assert model.risk_model_for_sqp(alpha) <= 0.28 < model.acceptance_risk(alpha)
    plugin, _ = frozen(horizon=2, point=values, mode=RiskMode.EXPERIMENTAL_PLUGIN)
    assert not bool(plugin.certified)
    np.testing.assert_allclose(plugin.acceptance_risk(alpha), plugin.risk_model_for_sqp(alpha))


def test_bilinear_gradients_match_product_reference_including_one_sided_endpoints():
    grid = AlphaGrid.create([0.01, 0.1, 0.2, 0.4])
    surface = grid.values[:, None]*grid.values[None, :]
    edge = jnp.array([[1, 2]], jnp.int32)
    function = lambda alpha: bilinear_credit(grid, surface[None], edge, alpha).sum()
    for alpha in [jnp.array([0.13, 0.27]), jnp.array([0.01, 0.4])]:
        np.testing.assert_allclose(jax.grad(function)(alpha), alpha[::-1], atol=1e-14)
    alpha = np.array([0.13, 0.27])
    finite_difference = [(float(function(jnp.asarray(alpha + 1e-6*np.eye(2)[i]))) -
                          float(function(jnp.asarray(alpha - 1e-6*np.eye(2)[i]))))/(2e-6) for i in range(2)]
    np.testing.assert_allclose(jax.grad(function)(jnp.asarray(alpha)), finite_difference, atol=1e-10)


def test_forest_is_selected_at_reference_and_stays_frozen_across_candidates():
    point = np.zeros((3, 4, 4))
    point[0, 1:, 1:] = 0.08
    point[1, 1:, 1:] = 0.06
    point[2, 2:, 2:] = 0.15
    model, data = frozen(point=point)
    np.testing.assert_array_equal(model.forest_mask, [True, True, False])
    old_mask = np.asarray(model.forest_mask).copy()
    model.acceptance_risk(jnp.full(3, 0.25))
    np.testing.assert_array_equal(model.forest_mask, old_mask)
    _, numerical, estimate, builder, _ = data
    new = jax.jit(builder)(numerical, estimate, jnp.full(3, 0.25))
    np.testing.assert_array_equal(new.forest_mask, [True, False, True])


def test_uncertified_or_unready_edges_use_zero_credit_in_certified_mode():
    _, (_, numerical, estimate, builder, reference) = frozen(point=jnp.full((3, 4, 4), 0.1))
    bad = replace(estimate, certified=jnp.zeros(3, bool))
    model = jax.jit(builder)(numerical, bad, reference)
    assert np.asarray(model.fallback_mask).all()
    np.testing.assert_array_equal(model.certified_surface, 0)
    np.testing.assert_array_equal(model.model_surface, 0)
    np.testing.assert_allclose(model.acceptance_risk(jnp.full(3, 0.1)), 0.3)
    assert bool(model.certified)


def test_configured_fixed_forest_empty_forest_and_invalid_candidate_subset():
    model, _ = frozen(config=SolverConfig(forest_kind="fixed", fixed_forest=((2, 3),)))
    np.testing.assert_array_equal(model.forest_mask, [False, False, True])
    model, _ = frozen(config=SolverConfig(forest_kind="fixed"))
    np.testing.assert_array_equal(model.forest_mask, False)
    _, (_, numerical, estimate, builder, reference) = frozen(config=SolverConfig(forest_kind="fixed", fixed_forest=((2, 3),)))
    subset = replace(numerical, edges=numerical.edges[:1])
    partial = replace(estimate, point_surface=estimate.point_surface[:1], lower_surface=estimate.lower_surface[:1],
                      ready=estimate.ready[:1], certified=estimate.certified[:1])
    assert int(builder(subset, partial, reference).error) == UpdateError.INVALID_INPUT
    with pytest.raises(ValueError, match="cycle"):
        frozen(config=SolverConfig(forest_kind="fixed", fixed_forest=((1, 2), (2, 3), (1, 3))))


def test_single_horizon_model_is_boole_and_freeze_matches_jit():
    model, (_, numerical, estimate, builder, reference) = frozen(horizon=1)
    compiled = jax.jit(builder)(numerical, estimate, reference)
    assert compiled.forest_mask.shape == (0,)
    np.testing.assert_allclose(compiled.acceptance_risk(jnp.array([0.13])), 0.13)
    for a, b in zip(jax.tree.leaves(model), jax.tree.leaves(compiled)):
        np.testing.assert_array_equal(a, b)


def test_invalid_frozen_inputs_are_not_accepted_as_risk_certificates():
    _, (_, numerical, estimate, builder, reference) = frozen()
    for bad in [replace(estimate, lower_surface=jnp.full_like(estimate.lower_surface, jnp.nan)),
                replace(estimate, error=jnp.asarray(UpdateError.OUT_OF_ORDER, jnp.int32))]:
        model = builder(numerical, bad, reference)
        assert int(model.error) != 0
        assert not np.isfinite(model.acceptance_risk(reference))


def test_dense_qp_against_high_accuracy_reference_and_original_unit_duals():
    from scipy.optimize import Bounds, LinearConstraint, minimize

    q = np.array([[2.0, 0.3, 0.1], [0.3, 3.0, -0.2], [0.1, -0.2, 4.0]])
    g = np.array([-1.0, -2.0, -0.5])
    a = np.vstack([np.ones((1, 3)), np.eye(3)])
    lower = np.array([-np.inf, 0, 0, 0])
    upper = np.array([0.65, 1, 1, 1])
    reference = minimize(lambda x: .5*x @ q @ x + g @ x, np.full(3, 0.1),
        jac=lambda x: q @ x + g, bounds=Bounds(np.zeros(3), np.ones(3)),
        constraints=[LinearConstraint(np.ones((1, 3)), -np.inf, 0.65)], method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 200})
    assert reference.success
    initial = init_qp_state(3, 4, jnp.float64)
    args = tuple(jnp.asarray(value) for value in (q, g, a, lower, upper))
    cfg = ADMMConfig(abs_tol=1e-11, rel_tol=1e-10)
    state, result = jax.jit(dense_admm_qp)(initial, *args, cfg)
    assert bool(result.converged) and bool(result.feasible)
    np.testing.assert_allclose(result.primal, reference.x, atol=1e-7)
    np.testing.assert_allclose(q @ result.primal + g + a.T @ result.dual, 0, atol=1e-9)
    assert float(result.dual[0]) > 0
    np.testing.assert_allclose(result.dual[1:], 0, atol=1e-9)
    warm, warmed = jax.jit(dense_admm_qp)(state, *args, cfg)
    assert int(warmed.iterations) <= int(result.iterations)
    np.testing.assert_allclose(warm.primal, state.primal, atol=1e-9)
    eager = dense_admm_qp(initial, *args, cfg)
    for left, right in zip(jax.tree.leaves(eager), jax.tree.leaves((state, result))):
        np.testing.assert_allclose(left, right, atol=1e-10)


def test_qp_scaling_large_curvature_and_active_box_constraints():
    q = jnp.diag(jnp.array([1e6, 1e3]))
    g = jnp.array([-1e3, -20.0])
    a = jnp.vstack([jnp.ones((1, 2)), jnp.eye(2)])
    lower = jnp.array([-jnp.inf, 0.0, 0.0])
    upper = jnp.array([0.01, 0.001, 0.007])
    _, result = jax.jit(dense_admm_qp)(init_qp_state(2, 3, jnp.float64), q, g, a, lower, upper,
                                     ADMMConfig(abs_tol=1e-11, rel_tol=1e-10))
    assert bool(result.converged)
    np.testing.assert_allclose(result.primal, [0.001, 0.007], atol=1e-9)
    assert float(result.dual[2]) > 0
    np.testing.assert_allclose(q @ result.primal + g + a.T @ result.dual, 0, atol=1e-7)


def test_qp_invalid_and_structurally_infeasible_inputs_fail_explicitly():
    initial = init_qp_state(2, 1, jnp.float64)
    for q, a, lo, hi, expected in [
        (-jnp.eye(2), jnp.ones((1, 2)), jnp.array([-jnp.inf]), jnp.array([1.0]), QPStatus.INVALID_INPUT),
        (jnp.eye(2), jnp.zeros((1, 2)), jnp.array([-jnp.inf]), jnp.array([-0.1]), QPStatus.INFEASIBLE),
        (jnp.eye(2), jnp.ones((1, 2)), jnp.array([1.0]), jnp.array([0.0]), QPStatus.INFEASIBLE),
    ]:
        state, result = jax.jit(dense_admm_qp)(initial, q, jnp.zeros(2), a, lo, hi)
        assert int(result.status) == expected
        assert not bool(result.feasible)
        assert np.isnan(result.primal).all()
        np.testing.assert_array_equal(state.primal, initial.primal)


def test_qp_unconstrained_case_and_float32():
    _, result = jax.jit(dense_admm_qp)(init_qp_state(2, 0, jnp.float32),
        jnp.diag(jnp.array([2, 4], jnp.float32)), jnp.array([-1, -2], jnp.float32),
        jnp.empty((0, 2), jnp.float32), jnp.empty(0, jnp.float32), jnp.empty(0, jnp.float32),
        ADMMConfig(abs_tol=1e-6, rel_tol=1e-6))
    assert bool(result.converged)
    np.testing.assert_allclose(result.primal, [0.5, 0.5], atol=1e-6)
    assert result.primal.dtype == jnp.float32


def test_symmetric_active_qp_requires_consensus_not_just_inequality_feasibility():
    # A stationary but negative-slack intermediate ADMM iterate has positive
    # risk dual and is not optimal. The exact symmetric solution is d == 0.
    h = 10
    q = jnp.eye(h)*1000
    g = jnp.full(h, -10.0)
    a = jnp.concatenate((jnp.ones((1, h)), jnp.eye(h)))
    lower = jnp.concatenate((jnp.array([-jnp.inf]), jnp.full(h, -0.009)))
    upper = jnp.concatenate((jnp.array([0.0]), jnp.full(h, 0.24)))
    _, result = jax.jit(dense_admm_qp)(init_qp_state(h, h+1, jnp.float64), q, g, a, lower, upper)
    assert bool(result.converged)
    np.testing.assert_allclose(result.primal, 0, atol=1e-8)
    np.testing.assert_allclose(result.dual[0], 10, atol=1e-6)
    assert abs(float(result.dual[0]*(a @ result.primal)[0])) < 1e-6


@pytest.mark.parametrize("budget,weights", [(0.35, [0.2, 0.3, 0.5]), (0.6, [0.05, 0.15, 0.8]),
                                             (0.06, [0.01, 0.09, 0.9])])
def test_full_sqp_boole_matches_high_accuracy_offline_reference(budget, weights):
    from scipy.optimize import Bounds, LinearConstraint, minimize

    model, (history, *_) = frozen()
    weights = np.array(weights)
    reference = minimize(lambda a: -np.sum(weights*np.log(a)), np.full(3, budget/3),
        jac=lambda a: -weights/a, bounds=Bounds(np.full(3, 0.01), np.full(3, 0.4)),
        constraints=[LinearConstraint(np.ones((1, 3)), -np.inf, budget)], method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 300})
    assert reference.success
    solver = build_solver(SolverConfig(mode="full_sqp", max_sqp_iterations=50, step_tol=1e-7,
                                      feasibility_tol=1e-8, qp_abs_tol=1e-11, qp_rel_tol=1e-10))
    state, result = jax.jit(solver.solve)(solver.init_state(history), budget, jnp.asarray(weights), model)
    assert bool(result.feasible) and bool(result.certified)
    assert bool(result.converged)
    assert not bool(result.requires_extreme)
    np.testing.assert_allclose(result.alpha, reference.x, atol=2e-7)
    assert float(result.acceptance_risk) <= budget+1e-8
    np.testing.assert_allclose(result.objective, reference.fun, atol=1e-7)
    np.testing.assert_array_equal(state.alpha, result.alpha)
    assert int(result.iterations) <= 50


def test_nonlinear_experimental_sqp_matches_offline_reference_without_certification():
    from scipy.optimize import Bounds, minimize

    grid = np.array([0.01, 0.1, 0.2, 0.4])
    model, (history, *_) = frozen(point=grid[:, None]*grid[None, :], mode=RiskMode.EXPERIMENTAL_PLUGIN)
    weights = np.array([0.2, 0.3, 0.5])
    risk = lambda a: a.sum()-a[0]*a[1]-a[0]*a[2]
    reference = minimize(lambda a: -np.sum(weights*np.log(a)), np.full(3, 0.1), jac=lambda a: -weights/a,
        bounds=Bounds(np.full(3, 0.01), np.full(3, 0.4)),
        constraints=[{"type": "ineq", "fun": lambda a: 0.35-risk(a),
                      "jac": lambda a: -np.array([1-a[1]-a[2], 1-a[0], 1-a[0]])}],
        method="SLSQP", options={"ftol": 1e-13, "maxiter": 300})
    assert reference.success
    solver = build_solver(SolverConfig(mode="full_sqp", max_sqp_iterations=80, step_tol=1e-7,
                                      feasibility_tol=1e-10, qp_abs_tol=1e-11, qp_rel_tol=1e-10))
    _, result = jax.jit(solver.solve)(solver.init_state(history, jnp.full(3, 0.1)), 0.35, jnp.asarray(weights), model)
    assert bool(result.feasible)
    assert not bool(result.certified)
    np.testing.assert_allclose(result.alpha, reference.x, atol=5e-6)
    np.testing.assert_allclose(result.objective, reference.fun, atol=1e-7)
    assert float(result.acceptance_risk) <= 0.35+1e-8


def test_objective_autodifferentiation_matches_reference_gradients_and_hessian():
    alpha, weights = jnp.array([0.07, 0.11, 0.16]), jnp.array([0.2, 0.3, 0.5])
    np.testing.assert_allclose(jax.grad(allocation_cost)(alpha, weights), -weights/alpha)
    np.testing.assert_allclose(jax.hessian(allocation_cost)(alpha, weights), np.diag(np.asarray(weights/alpha**2)))
    epsilon = 1e-6
    reference = [(float(allocation_cost(alpha+epsilon*jnp.eye(3)[h], weights))-
                  float(allocation_cost(alpha-epsilon*jnp.eye(3)[h], weights)))/(2*epsilon) for h in range(3)]
    np.testing.assert_allclose(jax.grad(allocation_cost)(alpha, weights), reference, rtol=1e-9)


def test_certified_restoration_rejects_optimistic_interpolation_and_retains_best_safe_point():
    grid = np.array([0.01, 0.1, 0.2, 0.4])
    model, _ = frozen(horizon=2, point=grid[:, None]*grid[None, :])
    proposal = jnp.full(2, 0.15)
    assert float(model.risk_model_for_sqp(proposal)) < 0.28
    assert not bool(acceptance_feasible(proposal, 0.28, model, 1e-8))
    result = jax.jit(restore_feasibility)(proposal, jnp.full(2, 0.1), jnp.full(2, 0.5), 0.28, model,
                                        SQPConfig(feasibility_tol=1e-8))
    assert bool(result.feasible) and bool(result.restored)
    np.testing.assert_allclose(result.alpha, [0.145, 0.145], atol=2e-8)
    assert float(model.acceptance_risk(result.alpha)) <= 0.28+1e-8


def test_disconnected_feasible_segment_is_sampled_without_monotonicity_assumption():
    # Both segment endpoints are infeasible. A certified-credit jump creates
    # an interior feasible piece, and every tested point uses the step evaluator.
    values = np.array([0.1, 0.10001, 0.2, 0.4])
    surface = np.zeros((4, 4))
    surface[1:, 1:] = 0.1
    model, _ = frozen(horizon=2, point=surface, grid_values=values, reference=[0.1, 0.1])
    assert not bool(acceptance_feasible(model.lower, 0.15, model, 1e-8))
    assert not bool(acceptance_feasible(jnp.full(2, 0.2), 0.15, model, 1e-8))
    result = jax.jit(restore_feasibility)(jnp.full(2, 0.2), model.lower, jnp.full(2, 0.5), 0.15, model,
                                        SQPConfig(feasibility_tol=1e-8))
    assert bool(result.feasible)
    assert float(model.acceptance_risk(result.alpha)) <= 0.15+1e-8
    assert np.asarray(result.alpha).min() >= 0.10001


def test_solver_no_safe_ordinary_allocation_requires_conservative_extreme_and_preserves_warm_state():
    model, (history, *_) = frozen()
    solver = build_solver(SolverConfig(max_sqp_iterations=1))
    initial = solver.init_state(history, jnp.full(3, 0.1))
    state, result = jax.jit(solver.solve)(initial, 0.015, jnp.full(3, 1/3), model)
    assert not bool(result.feasible) and not bool(result.certified)
    assert bool(result.requires_extreme)
    assert int(result.required_action) == 1
    assert int(result.status) in (SolverStatus.NO_ACCEPTED_ALLOCATION, SolverStatus.NUMERICAL_FAILURE)
    assert np.isnan(result.alpha).all()
    for left, right in zip(jax.tree.leaves(state), jax.tree.leaves(initial)):
        np.testing.assert_array_equal(left, right)


def test_solver_eager_jit_equivalence_warm_cold_and_rti_tracking_under_changing_budget_weights():
    model, (history, *_) = frozen()
    full = build_solver(SolverConfig(mode="full_sqp", max_sqp_iterations=40))
    initial = full.init_state(history)
    args = (initial, 0.3, jnp.array([0.2, 0.3, 0.5]), model)
    eager, compiled = full.solve(*args), jax.jit(full.solve)(*args)
    for left, right in zip(jax.tree.leaves(eager), jax.tree.leaves(compiled)):
        np.testing.assert_allclose(left, right, atol=2e-8)
    state = compiled[0]
    old_qp_iterations = int(compiled[1].qp_iterations)
    warmed, result = jax.jit(full.solve)(state, 0.3, jnp.array([0.2, 0.3, 0.5]), model)
    assert bool(result.feasible)
    assert int(result.qp_iterations) <= old_qp_iterations
    cold = build_solver(SolverConfig(mode="full_sqp", max_sqp_iterations=40, warm_start=False))
    _, restarted = jax.jit(cold.solve)(warmed, 0.3, jnp.array([0.2, 0.3, 0.5]), model)
    np.testing.assert_allclose(restarted.alpha, result.alpha, atol=2e-6)
    rti = build_solver(SolverConfig(mode="rti", max_sqp_iterations=1, feasibility_tol=1e-8))
    for index in range(10):
        budget = 0.3+index*0.001
        weights = jnp.array([0.2+0.0005*index, 0.3-0.0005*index, 0.5])
        state, result = jax.jit(rti.solve)(state, budget, weights, model)
        assert bool(result.feasible) and bool(result.certified)
        assert int(result.iterations) == 1
        np.testing.assert_allclose(result.alpha, budget*weights, atol=2e-5)
        assert float(result.acceptance_risk) <= budget+1e-8


def test_invalid_solver_inputs_fail_explicitly_and_experimental_results_always_noncertified():
    model, (history, *_) = frozen(mode=RiskMode.EXPERIMENTAL_PLUGIN)
    solver = build_solver(SolverConfig())
    initial = solver.init_state(history)
    for budget, weights in [(np.nan, jnp.full(3, 1/3)), (-0.1, jnp.full(3, 1/3)), (0.1, jnp.array([0, 0.5, 0.5]))]:
        _, result = jax.jit(solver.solve)(initial, budget, weights, model)
        assert int(result.status) == SolverStatus.INVALID_INPUT
        assert not bool(result.feasible) and not bool(result.certified)
        assert bool(result.requires_extreme)
    _, result = solver.solve(initial, 0.1, jnp.full(3, 1/3), model)
    assert bool(result.feasible) and not bool(result.certified)


def test_truncated_qp_does_not_bypass_acceptance_or_claim_convergence():
    model, (history, *_) = frozen()
    solver = build_solver(SolverConfig(qp_max_iterations=1))
    _, result = jax.jit(solver.solve)(solver.init_state(history, jnp.full(3, 0.2/3)),
                                     0.2, jnp.array([0.1, 0.2, 0.7]), model)
    assert bool(result.feasible) and bool(result.certified)
    assert not bool(result.qp_converged) and not bool(result.converged)
    assert bool(acceptance_feasible(result.alpha, 0.2, model, 1e-6))


def test_solver_composes_in_scan_independent_batches_and_float32():
    grid = jnp.array([0.01, 0.1, 0.2, 0.4], jnp.float32)
    model, (history, *_) = frozen(grid_values=grid)
    solver = build_solver(SolverConfig(mode="rti", max_sqp_iterations=2,
                                      qp_abs_tol=1e-6, qp_rel_tol=1e-6))
    initial = solver.init_state(history, jnp.full(3, 0.1, jnp.float32))

    def step(state, index):
        budget = jnp.asarray(0.3, jnp.float32)+jnp.asarray(index, jnp.float32)*0.001
        state, result = solver.solve(state, budget, jnp.array([0.2, 0.3, 0.5], jnp.float32), model)
        return state, (result.feasible, result.certified, result.acceptance_risk-budget, result.iterations)

    state, (feasible, certified, violation, iterations) = jax.jit(lambda: jax.lax.scan(
        step, initial, jnp.arange(8, dtype=jnp.int32)))()
    assert np.asarray(feasible & certified).all()
    assert float(jnp.max(violation)) <= 1e-6
    np.testing.assert_array_equal(iterations, 2)
    assert state.alpha.dtype == jnp.float32
    batch = jax.jit(jax.vmap(solver.solve, in_axes=(None, 0, None, None)))(
        initial, jnp.array([0.3, 0.31], jnp.float32), jnp.array([0.2, 0.3, 0.5], jnp.float32), model)
    assert np.asarray(batch[1].feasible).all()


def test_solver_startup_rejects_invalid_numerical_controls():
    for config in [SolverConfig(qp_max_iterations=0), SolverConfig(qp_abs_tol=-1),
                   SolverConfig(backtrack_factor=1), SolverConfig(qp_initial_penalty=1e10)]:
        with pytest.raises(ValueError):
            build_solver(config)
