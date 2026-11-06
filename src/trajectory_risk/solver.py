"""Frozen forest risk representations and (Phase 6) dense SQP/RTI.

Interpolation is a derivative model, never a statistical certificate. Ordinary
allocations must pass the distinct acceptance evaluator. Statistical validity
also depends on the adapter's marginal conformal contract; this module cannot
certify arbitrary forecasting geometry.
"""

from dataclasses import dataclass, replace
from enum import IntEnum
from functools import partial

import jax
import jax.numpy as jnp

from .configs import RiskMode, SolverConfig, _validate_forest, validate_solver_numerics
from .dependence import DependenceEstimate, DependenceState
from .history import ActionMode, AlphaGrid, HistoryState, UpdateError


def monotone_lower_grid(surface):
    """Prefix maxima implement max over all certified southwest grid corners."""
    return jax.lax.associative_scan(jnp.maximum,
        jax.lax.associative_scan(jnp.maximum, surface, axis=-1), axis=-2)


def lower_grid_credit(grid: AlphaGrid, surface, edges, alpha):
    """Exact lower-grid extension; zero below either first grid coordinate."""
    pairs = alpha[edges-1]
    indices = jnp.searchsorted(grid.values, pairs, side="right") - 1
    inside = jnp.all(indices >= 0, axis=-1)
    indices = jnp.clip(indices, 0, grid.values.size-1)
    value = surface[jnp.arange(edges.shape[0]), indices[:, 0], indices[:, 1]]
    return jnp.where(inside, value, 0)


def bilinear_credit(grid: AlphaGrid, surface, edges, alpha):
    """Bilinear interpolation with one-sided derivatives at grid endpoints."""
    pairs = alpha[edges-1]
    lo = jnp.clip(jnp.searchsorted(grid.values, pairs, side="right")-1, 0, grid.values.size-2)
    fraction = (pairs-grid.values[lo])/(grid.values[lo+1]-grid.values[lo])
    # At the endpoints retain the in-domain derivative rather than clip's half derivative.
    fraction = jnp.where(pairs < grid.values[0], 0,
                        jnp.where(pairs > grid.values[-1], 1, fraction))
    e = jnp.arange(edges.shape[0])
    x, y = fraction[:, 0], fraction[:, 1]
    i, j = lo[:, 0], lo[:, 1]
    return ((1-x)*(1-y)*surface[e, i, j] + x*(1-y)*surface[e, i+1, j]
            + (1-x)*y*surface[e, i, j+1] + x*y*surface[e, i+1, j+1])


def maximum_spanning_forest(edges, weights, horizon: int):
    """Kruskal with descending credit and lexicographic endpoint tie-breaking.

    horizon is static under JIT. The result is an E-entry mask, also for sparse
    candidate graphs. With all unordered pairs this selects an H-1 edge tree.
    """
    if edges.shape[0] == 0:
        return jnp.zeros(0, bool)
    order = jnp.lexsort((edges[:, 1], edges[:, 0], -weights))

    def root(parents, vertex):
        return jax.lax.fori_loop(0, horizon, lambda _, v: parents[v], vertex)

    def edge_step(index, carry):
        parents, selected = carry
        e = order[index]
        s = jnp.clip(edges[e, 0]-1, 0, horizon-1)
        r = jnp.clip(edges[e, 1]-1, 0, horizon-1)
        a, b = root(parents, s), root(parents, r)
        take = a != b
        parents = parents.at[b].set(jnp.where(take, a, parents[b]))
        return parents, selected.at[e].set(take)

    return jax.lax.fori_loop(0, edges.shape[0], edge_step,
        (jnp.arange(horizon, dtype=jnp.int32), jnp.zeros(edges.shape[0], bool)))[1]


def _frechet_cap(credits, edges, alpha):
    """An overlap cannot exceed either marginal: p_st <= min(P_s, P_t) <= min(alpha_s, alpha_t)
    under the adapter's marginal contract. Noisy point surfaces can violate this."""
    return jnp.minimum(credits, jnp.min(alpha[edges-1], axis=-1))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FrozenRiskModel:
    grid: AlphaGrid
    edges: jax.Array
    forest_mask: jax.Array
    model_surface: jax.Array
    certified_surface: jax.Array  # Prefix maximum of usable lower certificates.
    lower: jax.Array
    upper: jax.Array
    certified: jax.Array  # Allocation-risk mode, conditional on the adapter contract.
    fallback_mask: jax.Array
    error: jax.Array

    def risk_model_for_sqp(self, alpha):
        credits = _frechet_cap(bilinear_credit(self.grid, self.model_surface, self.edges, alpha),
                               self.edges, alpha)
        value = jnp.sum(alpha) - jnp.sum(jnp.where(self.forest_mask, credits, 0))
        return jnp.where(self.error == 0, value, jnp.inf)

    def certified_risk(self, alpha):
        credits = _frechet_cap(lower_grid_credit(self.grid, self.certified_surface, self.edges, alpha),
                               self.edges, alpha)
        value = jnp.sum(alpha) - jnp.sum(jnp.where(self.forest_mask, credits, 0))
        return jnp.where(self.error == 0, value, jnp.inf)

    def acceptance_risk(self, alpha):
        return jax.lax.cond(self.certified, lambda a: self.certified_risk(a),
                            lambda a: self.risk_model_for_sqp(a), alpha)


def build_risk_model(config: SolverConfig, mode: RiskMode, history: HistoryState):
    """Select forest policy and risk mode BEFORE compilation; return a freeze kernel.

    Certified mode interpolates the credited LOWER grid for search. Experimental
    mode interpolates the POINT grid for both search and acceptance.
    Forest selection uses the credited surface of the selected mode at the last ordinary
    alpha reference (point in experimental mode, certified lower otherwise).
    No forest selection occurs while evaluating a candidate alpha.
    """
    if not isinstance(mode, RiskMode):
        raise ValueError("risk mode must be a RiskMode")
    if config.forest_kind not in ("maximum_spanning_tree", "fixed"):
        raise ValueError("unknown forest policy")
    h = history.issued_alpha.shape[1]
    _validate_forest(config.fixed_forest, h)
    if config.forest_kind != "fixed" and config.fixed_forest:
        raise ValueError("fixed_forest requires forest_kind=fixed")
    fixed = tuple(config.fixed_forest)
    use_fixed = config.forest_kind == "fixed"
    experimental = mode is RiskMode.EXPERIMENTAL_PLUGIN
    lower = jnp.full(h, history.alpha_min, history.grid.values.dtype)
    upper = jnp.full(h, history.alpha_max, history.grid.values.dtype)

    def freeze(dependence: DependenceState, estimate: DependenceEstimate, alpha_reference):
        grid, edges = dependence.grid, dependence.edges
        reference = jnp.asarray(alpha_reference, grid.values.dtype)
        if reference.shape != (h,) or estimate.point_surface.shape != (edges.shape[0], grid.values.size, grid.values.size):
            raise ValueError("risk inputs must match the static horizon, candidate edges, and grid")
        usable = estimate.certified & estimate.ready & (estimate.error == 0)
        point = jnp.asarray(estimate.point_surface, grid.values.dtype)
        credited = jnp.where(usable[:, None, None], jnp.asarray(estimate.lower_surface, grid.values.dtype), 0)
        relevant = point if experimental else credited
        valid = (jnp.all(jnp.isfinite(relevant) & (relevant >= 0) & (relevant <= 1)) &
                 jnp.all(jnp.isfinite(credited) & (credited >= 0) & (credited <= 1)) &
                 jnp.all(jnp.isfinite(reference) & (reference >= lower) & (reference <= upper)) &
                 jnp.all((edges[:, 0] >= 1) & (edges[:, 0] < edges[:, 1]) & (edges[:, 1] <= h)) &
                 (grid.values[0] <= lower[0]) & (grid.values[-1] >= upper[0]))
        monotone = monotone_lower_grid(credited)
        # Select the tree with the surface this mode actually credits. Using the
        # certified surface in experimental mode makes every weight zero whenever
        # the certificate radius exceeds the overlap, so ties collapse the tree to
        # an arbitrary star around horizon 1.
        edge_weights = (bilinear_credit(grid, point, edges, reference) if experimental
                        else lower_grid_credit(grid, monotone, edges, reference))
        if use_fixed:
            fixed_edges = jnp.asarray(fixed, jnp.int32).reshape((-1, 2))
            mask = jnp.any(jnp.all(edges[:, None, :] == fixed_edges[None, :, :], axis=-1), axis=-1)
            valid &= jnp.sum(mask) == len(fixed)
        else:
            mask = maximum_spanning_forest(edges, edge_weights, h)
        model_surface = point if experimental else monotone
        error = jnp.where(estimate.error != 0, estimate.error,
                          jnp.where(valid, UpdateError.OK, UpdateError.INVALID_INPUT)).astype(jnp.int32)
        return FrozenRiskModel(grid, edges, mask, model_surface, monotone, lower, upper,
                               jnp.asarray(not experimental), ~usable, error)

    return freeze


@partial(jax.tree_util.register_dataclass,
         data_fields=["abs_tol", "rel_tol", "initial_penalty", "min_penalty", "max_penalty", "proximal"],
         meta_fields=["max_iterations", "update_interval"])
@dataclass(frozen=True)
class ADMMConfig:
    max_iterations: int = 400
    update_interval: int = 25
    abs_tol: float = 1e-8
    rel_tol: float = 1e-7
    initial_penalty: float = 1.0
    min_penalty: float = 1e-4
    max_penalty: float = 1e6
    proximal: float = 1e-9


class QPStatus(IntEnum):
    CONVERGED = 0
    MAX_ITERATIONS = 1
    INVALID_INPUT = 2
    INFEASIBLE = 3


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class QPState:
    primal: jax.Array
    auxiliary: jax.Array
    dual: jax.Array  # Unscaled multipliers in the ORIGINAL constraint units.
    penalty: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class QPResult:
    primal: jax.Array
    dual: jax.Array
    objective: jax.Array
    primal_residual: jax.Array
    dual_residual: jax.Array
    feasible: jax.Array
    converged: jax.Array
    iterations: jax.Array
    status: jax.Array


def init_qp_state(dimension: int, constraints: int, dtype, penalty=1.0):
    return QPState(jnp.zeros(dimension, dtype), jnp.zeros(constraints, dtype),
                   jnp.zeros(constraints, dtype), jnp.asarray(penalty, dtype))


def _norm_inf(value):
    return jnp.max(jnp.abs(value), initial=0)


def _qp_residuals(x, dual, hessian, gradient, matrix, lower, upper):
    image = matrix @ x
    primal = jnp.maximum(jnp.max(jnp.maximum(lower-image, 0), initial=0),
                         jnp.max(jnp.maximum(image-upper, 0), initial=0))
    dual_residual = _norm_inf(hessian @ x + gradient + matrix.T @ dual)
    return primal, dual_residual


def dense_admm_qp(state: QPState, hessian, gradient, matrix, lower, upper,
                  config: ADMMConfig = ADMMConfig()):
    """Dense convex QP: min .5*x'Q*x+g'x, lower <= A*x <= upper.

    Diagonal variable and row scaling whiten the small QP. Warm starts and
    returned multipliers stay in original units. The factorization is cached
    until rho changes. Independent implementation; no SLS dynamics are used.
    """
    n, k = gradient.size, matrix.shape[0]
    if n < 1 or gradient.ndim != 1 or hessian.shape != (n, n) or matrix.shape != (k, n) or lower.shape != (k,) or upper.shape != (k,):
        raise ValueError("incompatible dense QP dimensions")
    if state.primal.shape != (n,) or state.auxiliary.shape != (k,) or state.dual.shape != (k,):
        raise ValueError("QP warm-start shapes do not match this problem")
    q = (hessian+hessian.T)/2
    dtype = q.dtype
    finite = (jnp.all(jnp.isfinite(q)) & jnp.all(jnp.isfinite(gradient)) &
              jnp.all(jnp.isfinite(matrix)) & jnp.all(~jnp.isnan(lower)) & jnp.all(~jnp.isnan(upper)))
    positive = jnp.min(jnp.linalg.eigvalsh(q)) > 0
    zero_row = jnp.all(matrix == 0, axis=-1)
    impossible = jnp.any(lower > upper) | jnp.any(zero_row & ((lower > 0) | (upper < 0)))
    valid = finite & positive & ~impossible

    def solve(_):
        scale = 1/jnp.sqrt(jnp.diag(q))
        transformed = matrix*scale[None, :]
        norms = jnp.linalg.norm(transformed, axis=-1)
        row_scale = jnp.where(norms > 0, 1/jnp.maximum(norms, jnp.finfo(dtype).tiny), 1)
        a = row_scale[:, None]*transformed
        lo, hi = row_scale*lower, row_scale*upper
        objective = scale[:, None]*q*scale[None, :]
        linear = scale*gradient
        rho = jnp.where(jnp.isfinite(state.penalty) & (state.penalty > 0),
                        jnp.clip(state.penalty, config.min_penalty, config.max_penalty), config.initial_penalty)
        warm_valid = (jnp.all(jnp.isfinite(state.primal)) & jnp.all(jnp.isfinite(state.dual)) &
                      jnp.all(jnp.isfinite(state.auxiliary)))
        x = jnp.where(warm_valid, state.primal/scale, 0)
        z = jnp.clip(jnp.where(warm_valid, row_scale*state.auxiliary, a @ x), lo, hi)
        y = jnp.where(warm_valid, state.dual/(row_scale*rho), 0)
        identity = jnp.eye(n, dtype=dtype)

        def factorize(rho):
            return jnp.linalg.cholesky(objective + rho*(a.T @ a) + config.proximal*identity)

        factor = factorize(rho)
        carry = (x, z, y, rho, factor, jnp.asarray(False), jnp.asarray(0, jnp.int32))

        def iteration(_, carry):
            def active(carry):
                x, z, y, rho, factor, _, count = carry
                rhs = -linear + rho*a.T @ (z-y) + config.proximal*x
                intermediate = jax.scipy.linalg.solve_triangular(factor, rhs, lower=True)
                new_x = jax.scipy.linalg.solve_triangular(factor.T, intermediate, lower=False)
                image = a @ new_x
                new_z = jnp.clip(image+y, lo, hi)
                new_y = y + image-new_z
                primal_scaled = _norm_inf(image-new_z)
                dual_scaled = _norm_inf(objective @ new_x + linear + a.T @ (rho*new_y))
                original_x = scale*new_x
                original_dual = row_scale*rho*new_y
                primal, dual_residual = _qp_residuals(original_x, original_dual, q, gradient, matrix, lower, upper)
                # Inequality feasibility alone does not imply ADMM consensus.
                # A negative slack with a positive multiplier can be stationary
                # yet violate complementarity if A*x differs from projected z.
                consensus = _norm_inf((image-new_z)/row_scale)
                primal = jnp.maximum(primal, consensus)
                p_tol = config.abs_tol + config.rel_tol*_norm_inf(matrix @ original_x)
                d_tol = config.abs_tol + config.rel_tol*jnp.maximum(_norm_inf(gradient),
                    jnp.maximum(_norm_inf(q @ original_x), _norm_inf(matrix.T @ original_dual)))
                converged = (primal <= p_tol) & (dual_residual <= d_tol) & jnp.all(jnp.isfinite(new_x))
                ratio = jnp.sqrt(jnp.maximum(primal_scaled, 1e-12)/jnp.maximum(dual_scaled, 1e-12))
                proposal = jnp.clip(rho*jnp.clip(ratio, 0.2, 5), config.min_penalty, config.max_penalty)
                change = ((count+1) % config.update_interval == 0) & ~converged & (
                    (primal_scaled > 10*dual_scaled) | (dual_scaled > 10*primal_scaled))
                new_rho = jnp.where(change, proposal, rho)
                new_y *= rho/new_rho  # Preserve unscaled multipliers when rho changes.
                new_factor = jax.lax.cond(change, factorize, lambda _: factor, new_rho)
                return new_x, new_z, new_y, new_rho, new_factor, converged, count+1

            return jax.lax.cond(carry[5], lambda c: c, active, carry)

        x, z, y, rho, _, converged, iterations = jax.lax.fori_loop(0, config.max_iterations, iteration, carry)
        primal, dual = scale*x, row_scale*rho*y
        p_res, d_res = _qp_residuals(primal, dual, q, gradient, matrix, lower, upper)
        feasible = jnp.all(jnp.isfinite(primal)) & (p_res <= config.abs_tol + config.rel_tol*_norm_inf(matrix @ primal))
        p_res = jnp.maximum(p_res, _norm_inf(matrix @ primal-z/row_scale))
        result = QPResult(primal, dual, .5*primal @ q @ primal + gradient @ primal,
                          p_res, d_res, feasible, converged, iterations,
                          jnp.where(converged, QPStatus.CONVERGED, QPStatus.MAX_ITERATIONS).astype(jnp.int32))
        return QPState(primal, z/row_scale, dual, rho), result

    def reject(_):
        status = jnp.where(impossible, QPStatus.INFEASIBLE, QPStatus.INVALID_INPUT).astype(jnp.int32)
        return state, QPResult(jnp.full(n, jnp.nan, dtype), jnp.full(k, jnp.nan, dtype),
            jnp.asarray(jnp.inf, dtype), jnp.asarray(jnp.inf, dtype), jnp.asarray(jnp.inf, dtype),
            jnp.asarray(False), jnp.asarray(False), jnp.asarray(0, jnp.int32), status)

    return jax.lax.cond(valid, solve, reject, operand=None)


@partial(jax.tree_util.register_dataclass,
         data_fields=["feasibility_tol", "step_tol", "hessian_regularization", "armijo", "backtrack_factor", "merit_penalty"],
         meta_fields=["max_iterations", "line_search_steps", "restoration_steps", "line_search", "warm_start", "full_mode"])
@dataclass(frozen=True)
class SQPConfig:
    max_iterations: int = 1
    line_search_steps: int = 20
    restoration_steps: int = 24
    line_search: bool = True
    warm_start: bool = True
    full_mode: bool = False
    feasibility_tol: float = 1e-6
    step_tol: float = 1e-5
    hessian_regularization: float = 1e-6
    armijo: float = 1e-4
    backtrack_factor: float = 0.5
    merit_penalty: float = 10.0


class SolverStatus(IntEnum):
    CONVERGED = 0
    FEASIBLE = 1
    FEASIBLE_RESTORED = 2
    NO_ACCEPTED_ALLOCATION = 3  # Does NOT prove global problem infeasibility.
    INVALID_INPUT = 4
    NUMERICAL_FAILURE = 5
    SKIPPED_EXTREME = 6


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SolverState:
    alpha: jax.Array
    initial_alpha: jax.Array
    risk_dual: jax.Array
    box_duals: jax.Array
    qp: QPState
    last_kkt_residual: jax.Array
    last_acceptance_risk: jax.Array
    last_feasible: jax.Array
    solves: jax.Array

    @property
    def qp_primal(self):
        return self.qp.primal

    @property
    def qp_dual(self):
        return self.qp.dual

    @property
    def qp_penalty(self):
        return self.qp.penalty


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SolverResult:
    alpha: jax.Array  # NaN if no ordinary allocation may be issued.
    risk_dual: jax.Array
    box_duals: jax.Array
    objective: jax.Array  # Minimization cost: -sum(w*log(alpha)).
    model_risk: jax.Array
    acceptance_risk: jax.Array
    kkt_residual: jax.Array  # Scaled MODEL KKT residual at the returned allocation.
    feasible: jax.Array
    model_feasible: jax.Array
    certified: jax.Array
    converged: jax.Array
    iterations: jax.Array
    qp_iterations: jax.Array
    qp_converged: jax.Array
    restored: jax.Array
    requires_extreme: jax.Array
    required_action: jax.Array
    status: jax.Array

    @property
    def risk_value(self):
        return self.acceptance_risk

    @property
    def utility(self):
        return -self.objective


def allocation_cost(alpha, weights):
    return -jnp.sum(weights*jnp.log(alpha))


def _model_risk(alpha, model):
    return model.risk_model_for_sqp(alpha)


def acceptance_feasible(alpha, budget, model: FrozenRiskModel, tolerance):
    value = model.acceptance_risk(alpha)
    return (jnp.all(jnp.isfinite(alpha) & (alpha > 0) & (alpha < 1)) & jnp.all(alpha >= model.lower) &
            jnp.all(alpha <= model.upper) & jnp.isfinite(value) & (value <= budget+tolerance))


def model_kkt_residual(alpha, risk_dual, box_duals, weights, budget, model):
    """Scaled stationarity, primal/dual feasibility, and complementarity.

    This measures the interpolated model, not stationarity of the discontinuous
    certified acceptance problem and not a global optimization gap.
    """
    gradient = jax.grad(allocation_cost)(alpha, weights)
    risk, jacobian = jax.value_and_grad(_model_risk)(alpha, model)
    stationarity = _norm_inf(gradient + risk_dual*jacobian + box_duals)/(1+_norm_inf(gradient))
    complement = jnp.abs(risk_dual*(risk-budget))/(1+jnp.abs(risk_dual))
    upper = jnp.maximum(box_duals, 0)
    lower = jnp.maximum(-box_duals, 0)
    box_complement = jnp.maximum(_norm_inf(upper*(model.upper-alpha)),
                                _norm_inf(lower*(alpha-model.lower)))/(1+_norm_inf(box_duals))
    primal = jnp.maximum(jnp.maximum(risk-budget, 0),
                        jnp.maximum(jnp.max(model.lower-alpha), jnp.max(alpha-model.upper)))
    return jnp.maximum(jnp.maximum(stationarity, complement),
                       jnp.maximum(box_complement, jnp.maximum(primal, jnp.maximum(-risk_dual, 0))))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RestorationResult:
    alpha: jax.Array
    objective: jax.Array
    feasible: jax.Array
    restored: jax.Array


def restore_feasibility(proposal, base, weights, budget, model, config: SQPConfig):
    """Sample a segment, then refine one feasible/infeasible local bracket.

    EVERY candidate is checked with acceptance_risk. No monotonicity of that
    evaluator is assumed or used to assert global feasibility/infeasibility.
    The best feasible utility among tested points is retained, even if a segment
    contains disconnected feasible pieces. The base need not itself be feasible.
    """
    fractions = jnp.concatenate((config.backtrack_factor ** jnp.arange(config.line_search_steps, dtype=proposal.dtype),
                                 jnp.zeros(1, proposal.dtype)))
    candidates = jnp.clip(base[None, :] + fractions[:, None]*(proposal-base)[None, :], model.lower, model.upper)
    valid = jax.vmap(lambda a: acceptance_feasible(a, budget, model, config.feasibility_tol))(candidates)
    costs = jax.vmap(allocation_cost, in_axes=(0, None))(candidates, weights)
    index = jnp.argmin(jnp.where(valid, costs, jnp.inf))
    exists = jnp.any(valid)
    best_alpha = jnp.where(exists, candidates[index], model.lower)
    best_cost = jnp.where(exists, costs[index], jnp.inf)
    lo = fractions[index]
    hi = jnp.min(jnp.where(fractions > lo, fractions, 1))
    # Refine only when the nearest tested larger fraction was infeasible.
    high_alpha = jnp.clip(base + hi*(proposal-base), model.lower, model.upper)
    refine = exists & (hi > lo) & ~acceptance_feasible(high_alpha, budget, model, config.feasibility_tol)

    def step(_, carry):
        def active(carry):
            lo, hi, best_alpha, best_cost = carry
            mid = lo+(hi-lo)/2
            candidate = jnp.clip(base + mid*(proposal-base), model.lower, model.upper)
            feasible = acceptance_feasible(candidate, budget, model, config.feasibility_tol)
            cost = allocation_cost(candidate, weights)
            improve = feasible & (cost < best_cost)
            return (jnp.where(feasible, mid, lo), jnp.where(feasible, hi, mid),
                    jnp.where(improve, candidate, best_alpha), jnp.where(improve, cost, best_cost))

        return jax.lax.cond(refine, active, lambda c: c, carry)

    _, _, best_alpha, best_cost = jax.lax.fori_loop(0, config.restoration_steps, step,
                                                  (lo, hi, best_alpha, best_cost))
    # Independent final check also protects boundary-rounding effects.
    exists &= acceptance_feasible(best_alpha, budget, model, config.feasibility_tol)
    restored = exists & (_norm_inf(best_alpha-proposal) > config.feasibility_tol)
    return RestorationResult(best_alpha, best_cost, exists, restored)


def sqp_search_direction(alpha, weights, budget, model, qp_state, sqp_config, qp_config):
    gradient = jax.grad(allocation_cost)(alpha, weights)
    hessian = jax.hessian(allocation_cost)(alpha, weights) + sqp_config.hessian_regularization*jnp.eye(alpha.size, dtype=alpha.dtype)
    risk, jacobian = jax.value_and_grad(_model_risk)(alpha, model)
    matrix = jnp.concatenate((jacobian[None, :], jnp.eye(alpha.size, dtype=alpha.dtype)), axis=0)
    lower = jnp.concatenate((jnp.asarray([-jnp.inf], alpha.dtype), model.lower-alpha))
    upper = jnp.concatenate(((budget-risk)[None], model.upper-alpha))
    qp, result = dense_admm_qp(qp_state, hessian, gradient, matrix, lower, upper, qp_config)
    return qp, result, gradient, risk, jacobian


def _model_line_search(alpha, direction, weights, budget, model, gradient, risk, jacobian, risk_dual, config):
    if not config.line_search:
        return jnp.clip(alpha+direction, model.lower, model.upper)
    penalty = jnp.maximum(config.merit_penalty, 1+jnp.abs(risk_dual))
    current = allocation_cost(alpha, weights) + penalty*jnp.maximum(risk-budget, 0)
    violation_slope = jnp.where(risk > budget, jacobian @ direction,
                                jnp.where(risk == budget, jnp.maximum(jacobian @ direction, 0), 0))
    slope = gradient @ direction + penalty*violation_slope
    fractions = config.backtrack_factor ** jnp.arange(config.line_search_steps, dtype=alpha.dtype)
    candidates = jnp.clip(alpha[None, :]+fractions[:, None]*direction, model.lower, model.upper)
    merit = jax.vmap(lambda a: allocation_cost(a, weights) + penalty*jnp.maximum(_model_risk(a, model)-budget, 0))(candidates)
    armijo = jnp.isfinite(merit) & (merit <= current + config.armijo*fractions*jnp.minimum(slope, 0))
    index = jnp.argmax(armijo)  # First (largest) accepted fraction.
    return jnp.where(jnp.any(armijo), candidates[index], alpha)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _SQPCarry:
    alpha: jax.Array
    risk_dual: jax.Array
    box_duals: jax.Array
    qp: QPState
    best_alpha: jax.Array
    best_cost: jax.Array
    has_best: jax.Array
    model_converged: jax.Array
    qp_converged: jax.Array
    failed: jax.Array
    iterations: jax.Array
    qp_iterations: jax.Array


def empty_solver_result(state, status, action=ActionMode.CONSERVATIVE_EXTREME):
    dtype = state.alpha.dtype
    return SolverResult(jnp.full_like(state.alpha, jnp.nan), state.risk_dual, state.box_duals,
        jnp.asarray(jnp.inf, dtype), jnp.asarray(jnp.inf, dtype), jnp.asarray(jnp.inf, dtype),
        jnp.asarray(jnp.inf, dtype), jnp.asarray(False), jnp.asarray(False), jnp.asarray(False),
        jnp.asarray(False), jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32), jnp.asarray(False),
        jnp.asarray(False), jnp.asarray(True), jnp.asarray(action, jnp.int32), jnp.asarray(status, jnp.int32))


def _solve_sqp(state, budget, weights, model, config, qp_config):
    dtype = state.alpha.dtype
    budget, weights = jnp.asarray(budget, dtype), jnp.asarray(weights, dtype)
    if budget.shape != () or weights.shape != state.alpha.shape or model.lower.shape != state.alpha.shape:
        raise ValueError("solver requires scalar budget and H-entry weights/bounds")
    seed = state.alpha if config.warm_start else state.initial_alpha
    seed = jnp.clip(seed, model.lower, model.upper)
    valid = (jnp.isfinite(budget) & (budget >= 0) & (budget <= 1) &
             jnp.all(jnp.isfinite(weights) & (weights > 0)) & jnp.all(jnp.isfinite(seed)) &
             jnp.all(jnp.isfinite(model.lower) & jnp.isfinite(model.upper) &
                     (model.lower > 0) & (model.lower <= model.upper) & (model.upper < 1)) & (model.error == 0))

    def run(_):
        qp = state.qp if config.warm_start else init_qp_state(seed.size, seed.size+1, dtype, qp_config.initial_penalty)
        dual = jnp.where(jnp.isfinite(state.risk_dual) & (state.risk_dual >= 0), state.risk_dual, 0) if config.warm_start else jnp.asarray(0, dtype)
        box = jnp.where(jnp.isfinite(state.box_duals), state.box_duals, 0) if config.warm_start else jnp.zeros_like(seed)
        initial = restore_feasibility(seed, model.lower, weights, budget, model, config)
        carry = _SQPCarry(seed, dual, box, qp, initial.alpha, initial.objective, initial.feasible,
                          jnp.asarray(False), jnp.asarray(False), jnp.asarray(False),
                          jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32))

        def iteration(_, carry):
            def active(carry):
                qp, direction, gradient, risk, jacobian = sqp_search_direction(
                    carry.alpha, weights, budget, model, carry.qp, config, qp_config)
                usable = (jnp.all(jnp.isfinite(direction.primal)) & jnp.all(jnp.isfinite(direction.dual)) &
                          jnp.isfinite(direction.objective)) & (
                    (direction.status == QPStatus.CONVERGED) | (direction.status == QPStatus.MAX_ITERATIONS))

                def take(_):
                    alpha = _model_line_search(carry.alpha, direction.primal, weights, budget, model,
                                               gradient, risk, jacobian, direction.dual[0], config)
                    base = jnp.where(carry.has_best, carry.best_alpha, model.lower)
                    filtered = restore_feasibility(alpha, base, weights, budget, model, config)
                    improve = filtered.feasible & (~carry.has_best | (filtered.objective < carry.best_cost))
                    best = jnp.where(improve, filtered.alpha, carry.best_alpha)
                    best_cost = jnp.where(improve, filtered.objective, carry.best_cost)
                    residual = model_kkt_residual(alpha, direction.dual[0], direction.dual[1:], weights, budget, model)
                    relative_step = _norm_inf(alpha-carry.alpha)/(1+_norm_inf(carry.alpha))
                    converged = direction.converged & (residual <= config.step_tol) & (relative_step <= config.step_tol)
                    return _SQPCarry(alpha, direction.dual[0], direction.dual[1:], qp,
                        best, best_cost, carry.has_best | filtered.feasible, converged, direction.converged,
                        jnp.asarray(False), carry.iterations+1, carry.qp_iterations+direction.iterations)

                return jax.lax.cond(usable, take,
                    lambda _: replace(carry, failed=jnp.asarray(True), qp_converged=jnp.asarray(False),
                                      iterations=carry.iterations+1, qp_iterations=carry.qp_iterations+direction.iterations), operand=None)

            stop = carry.failed | (carry.model_converged & config.full_mode)
            return jax.lax.cond(stop, lambda c: c, active, carry)

        final = jax.lax.fori_loop(0, config.max_iterations, iteration, carry)
        # A feasibility-tolerance-sized utility tie should not retain an older
        # nonstationary iterate instead of an accepted model-converged iterate.
        current_cost = allocation_cost(final.alpha, weights)
        utility_allowance = (1+jnp.abs(final.risk_dual))*config.feasibility_tol
        prefer_current = (final.model_converged &
                          acceptance_feasible(final.alpha, budget, model, config.feasibility_tol) &
                          (current_cost <= final.best_cost + utility_allowance))
        alpha = jnp.where(prefer_current, final.alpha, final.best_alpha)
        feasible = final.has_best & acceptance_feasible(alpha, budget, model, config.feasibility_tol)
        residual = model_kkt_residual(alpha, final.risk_dual, final.box_duals, weights, budget, model)
        converged = feasible & ~final.failed & final.qp_converged & (residual <= config.step_tol)
        restored = feasible & (_norm_inf(alpha-final.alpha) > config.feasibility_tol)
        status = jnp.where(feasible,
            jnp.where(converged, SolverStatus.CONVERGED,
                      jnp.where(restored | final.failed, SolverStatus.FEASIBLE_RESTORED, SolverStatus.FEASIBLE)),
            jnp.where(final.failed, SolverStatus.NUMERICAL_FAILURE, SolverStatus.NO_ACCEPTED_ALLOCATION)).astype(jnp.int32)
        risk = model.acceptance_risk(alpha)
        model_risk = model.risk_model_for_sqp(alpha)
        result = SolverResult(jnp.where(feasible, alpha, jnp.nan), final.risk_dual, final.box_duals,
            jnp.where(feasible, allocation_cost(alpha, weights), jnp.inf), model_risk, risk,
            jnp.where(feasible, residual, jnp.inf), feasible,
            feasible & (model_risk <= budget+config.feasibility_tol), feasible & model.certified,
            converged, final.iterations, final.qp_iterations, final.qp_converged, restored,
            ~feasible, jnp.where(feasible, ActionMode.ORDINARY, ActionMode.CONSERVATIVE_EXTREME).astype(jnp.int32), status)
        updated = replace(state, alpha=alpha, risk_dual=final.risk_dual, box_duals=final.box_duals,
                          qp=final.qp, last_kkt_residual=residual, last_acceptance_risk=risk,
                          last_feasible=feasible, solves=state.solves+1)
        # A failed solve never corrupts the last usable primal/dual warm start.
        return jax.lax.cond(feasible, lambda _: updated, lambda _: state, operand=None), result

    return jax.lax.cond(valid, run,
        lambda _: (state, empty_solver_result(state, SolverStatus.INVALID_INPUT)), operand=None)


class SQPSolver:
    """Construction-time wrapper around fixed-shape, independently written JAX kernels."""

    def __init__(self, config: SolverConfig):
        validate_solver_numerics(config)
        self.config = SQPConfig(config.max_sqp_iterations, config.line_search_steps, config.restoration_steps,
            config.line_search, config.warm_start, config.mode == "full_sqp", config.feasibility_tol,
            config.step_tol, config.hessian_regularization, config.armijo, config.backtrack_factor, config.merit_penalty)
        self.qp_config = ADMMConfig(config.qp_max_iterations, config.qp_penalty_update_interval,
            config.qp_abs_tol, config.qp_rel_tol, config.qp_initial_penalty, config.qp_min_penalty,
            config.qp_max_penalty, config.qp_proximal_regularization)

    def init_state(self, history: HistoryState, alpha_reference=None):
        h, dtype = history.issued_alpha.shape[1], history.grid.values.dtype
        alpha = jnp.full(h, history.alpha_min, dtype) if alpha_reference is None else jnp.asarray(alpha_reference, dtype)
        if alpha.shape != (h,) or not bool(jnp.all(jnp.isfinite(alpha) & (alpha >= history.alpha_min) & (alpha <= history.alpha_max))):
            raise ValueError("solver initialization must be an H-entry allocation inside the ordinary box")
        return SolverState(alpha, alpha, jnp.asarray(0, dtype), jnp.zeros(h, dtype),
            init_qp_state(h, h+1, dtype, self.qp_config.initial_penalty),
            jnp.asarray(jnp.inf, dtype), jnp.asarray(jnp.inf, dtype), jnp.asarray(False), jnp.asarray(0, jnp.int32))

    def solve(self, state, budget, weights, risk_model: FrozenRiskModel):
        return _solve_sqp(state, budget, weights, risk_model, self.config, self.qp_config)


def build_solver(config: SolverConfig) -> SQPSolver:
    return SQPSolver(config)
