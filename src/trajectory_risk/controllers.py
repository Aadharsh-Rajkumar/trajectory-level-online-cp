"""Ordinary scalar actuation and a separate exact accounting/safety guard.

Controllers consume aggregate physical-round increments, independently of the
signal construction. Only actual resolved labels can change q; q is never
projected or clipped. PID is a generic budget adaptation, not original score PID.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from math import atan, isfinite, log, pi

import jax
import jax.numpy as jnp

from .configs import ControllerConfig, SafetyConfig
from .history import ActionMode, HistoryState, ResolvedLabels, UpdateError, resolved_labels_at


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AccountingState:
    q: jax.Array
    accounted_origin: jax.Array
    resolved_count: jax.Array
    last_observe_time: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AccountingUpdate:
    state: AccountingState
    resolved: ResolvedLabels
    error: jax.Array

    @property
    def accepted(self):
        return self.error == UpdateError.OK


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SafetyDecision:
    budget: jax.Array
    action_mode: jax.Array
    overridden: jax.Array
    error: jax.Array


class ExactAccountingGuard:
    """Construction-time configuration around pure, JIT-compatible operations.

    Call update once at every physical round, starting at t=0, after history
    has resolved all available labels. Duplicate, skipped, or stale operations
    are rejected without changing state. protect must be used before issuance.
    """

    def __init__(self, beta: float, config: SafetyConfig):
        if not isfinite(beta) or not 0 < beta < 1:
            raise ValueError("beta must be in (0, 1)")
        if not (all(isfinite(x) for x in (config.eta_q, config.q_init, config.q_lower, config.q_upper))
                and config.eta_q > 0 and config.q_lower < config.q_upper
                and config.q_lower <= config.q_init <= config.q_upper):
            raise ValueError("invalid exact-accounting gain, initial state, or safety thresholds")
        self.beta = beta
        self.eta_q = config.eta_q
        self.q_init = config.q_init
        self.q_lower = config.q_lower
        self.q_upper = config.q_upper

    def init_state(self, history: HistoryState) -> AccountingState:
        return AccountingState(jnp.asarray(self.q_init, history.grid.values.dtype),
                               jnp.full_like(history.origin_id, -1), jnp.asarray(0, jnp.int32),
                               jnp.asarray(-1, jnp.int32))

    def update(self, state: AccountingState, history: HistoryState, t) -> AccountingUpdate:
        return _update_accounting(state, history, t, self.beta, self.eta_q)

    def protect(self, state: AccountingState, proposal) -> SafetyDecision:
        q = state.q
        proposal = jnp.asarray(proposal, q.dtype)
        low = q <= self.q_lower
        high = q >= self.q_upper
        valid = jnp.isfinite(q) & (low | high | (jnp.isfinite(proposal) & (proposal >= 0) & (proposal <= 1)))
        # Invalid ordinary proposals fail closed; the host must still check error.
        conservative = low | ~valid
        mode = jnp.where(conservative, ActionMode.CONSERVATIVE_EXTREME,
                         jnp.where(high, ActionMode.AGGRESSIVE_EXTREME, ActionMode.ORDINARY))
        budget = jnp.where(conservative, 0, jnp.where(high, 1, proposal))
        return SafetyDecision(budget, mode.astype(jnp.int32), conservative | high,
                              jnp.where(valid, UpdateError.OK, UpdateError.INVALID_INPUT).astype(jnp.int32))


class TangentAccountingGuard(ExactAccountingGuard):
    """Tangent-based calibration safeguard (overleaf note section 2.5, eqs. 24-25).

    v = logit(b) + K tan(pi/2 clip(q/(eta_q B_t), -c, c)), with the proposal b
    clipped to [eps, 1-eps]. The issued budget is sigmoid(v) strictly inside
    (-L, L); at or beyond +/-L the extreme actions take over, and q = 0 leaves b
    unchanged. c is chosen so that |q/eta_q| >= c B_t forces an extreme for every
    proposal, which gives -c B_t - H(1-beta) <= q_t/eta_q <= c B_t + H beta (eq. 26).
    """

    def __init__(self, beta: float, config: SafetyConfig):
        super().__init__(beta, config)
        margin = log((1-config.tangent_eps)/config.tangent_eps) if 0 < config.tangent_eps < 0.5 else None
        if margin is None or not (isfinite(config.tangent_limit) and config.tangent_limit > margin):
            raise ValueError("the tangent safeguard requires eps in (0, 0.5) and L > log((1-eps)/eps)")
        if not (isfinite(config.tangent_gain) and config.tangent_gain > 0 and
                isfinite(config.tangent_envelope) and config.tangent_envelope > 0 and
                0 <= config.tangent_envelope_power < 1 and config.q_init == 0 and
                0 <= config.tangent_budget_floor < 1):
            raise ValueError("the tangent safeguard requires K, B > 0, power and floor in [0, 1), and q_init = 0")
        self.gain, self.limit, self.eps = config.tangent_gain, config.tangent_limit, config.tangent_eps
        self.envelope, self.power = config.tangent_envelope, config.tangent_envelope_power
        self.floor = config.tangent_budget_floor
        self.saturation = 2/pi*atan((self.limit + margin)/self.gain)

    def envelope_at(self, t, dtype=None):
        """B_t; time is the physical round whose labels q already includes."""
        return self.envelope*jnp.maximum(jnp.asarray(t, dtype or jnp.asarray(0.0).dtype), 1)**self.power

    def protect(self, state: AccountingState, proposal) -> SafetyDecision:
        q = state.q
        proposal = jnp.asarray(proposal, q.dtype)
        b = jnp.clip(proposal, self.eps, 1-self.eps)
        ratio = q/(self.eta_q*self.envelope_at(state.last_observe_time, q.dtype))
        v = jnp.log(b/(1-b)) + self.gain*jnp.tan(pi/2*jnp.clip(ratio, -self.saturation, self.saturation))
        valid = jnp.isfinite(q) & jnp.isfinite(proposal) & (proposal >= 0) & (proposal <= 1)
        # Saturation implies |v| >= L in exact arithmetic; test it directly so rounding at
        # the boundary cannot escape eq. (26). Invalid inputs fail closed; the host must
        # still check error.
        conservative = (v <= -self.limit) | (ratio <= -self.saturation) | ~valid
        high = (v >= self.limit) | (ratio >= self.saturation)
        mode = jnp.where(conservative, ActionMode.CONSERVATIVE_EXTREME,
                         jnp.where(high, ActionMode.AGGRESSIVE_EXTREME, ActionMode.ORDINARY))
        budget = jnp.where(conservative, 0, jnp.where(high, 1, jnp.maximum(jax.nn.sigmoid(v), self.floor)))
        return SafetyDecision(budget, mode.astype(jnp.int32), conservative | high,
                              jnp.where(valid, UpdateError.OK, UpdateError.INVALID_INPUT).astype(jnp.int32))


def _update_accounting(state, history, t, beta, eta_q):
    time = jnp.asarray(t, jnp.int32)
    batch = resolved_labels_at(history, time)
    unseen = history.valid_origin & (state.accounted_origin != history.origin_id)
    late = unseen & history.trajectory_resolved & (history.reveal_time < time)
    overdue = unseen & (history.origin_id + history.issued_alpha.shape[1] <= time) & ~(
        history.trajectory_resolved & (history.reveal_time <= time))
    valid_labels = jnp.all(~batch.mask | ((batch.labels == 0) | (batch.labels == 1)))
    error = jnp.where(time == state.last_observe_time, UpdateError.DUPLICATE,
            jnp.where((time != state.last_observe_time + 1) | jnp.any(late), UpdateError.OUT_OF_ORDER,
            jnp.where(~valid_labels | jnp.any(overdue), UpdateError.INVALID_INPUT, UpdateError.OK))).astype(jnp.int32)

    def apply(_):
        slots = jnp.mod(batch.origin_id, history.origin_id.size)
        mask = batch.mask & (state.accounted_origin[slots] != batch.origin_id)
        increments = jnp.where(mask, jnp.asarray(beta, state.q.dtype) - batch.labels, 0)
        accounted = state.accounted_origin.at[slots].set(jnp.where(mask, batch.origin_id,
                                                                   state.accounted_origin[slots]))
        updated = replace(state, q=state.q + eta_q * jnp.sum(increments),
                          accounted_origin=accounted,
                          resolved_count=state.resolved_count + jnp.sum(mask, dtype=jnp.int32),
                          last_observe_time=time)
        return AccountingUpdate(updated, ResolvedLabels(batch.origin_id, batch.labels, mask), error)

    def reject(_):
        return AccountingUpdate(state, ResolvedLabels(batch.origin_id, jnp.full_like(batch.labels, -1),
                                                      jnp.zeros_like(batch.mask)), error)

    return jax.lax.cond(error == UpdateError.OK, apply, reject, operand=None)


def build_accounting_guard(config: SafetyConfig, beta: float) -> ExactAccountingGuard:
    choices = {"threshold": ExactAccountingGuard, "tangent": TangentAccountingGuard}
    if config.kind not in choices:
        raise ValueError(f"unknown safety guard: {config.kind!r}")
    return choices[config.kind](beta, config)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ControllerState:
    budget: jax.Array
    integral: jax.Array
    previous_error: jax.Array
    last_time: jax.Array
    updates: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ControllerUpdate:
    state: ControllerState
    error: jax.Array

    @property
    def accepted(self):
        return self.error == UpdateError.OK


class OuterBudgetController(ABC):
    @abstractmethod
    def init_state(self, dtype=None) -> ControllerState: ...

    @abstractmethod
    def update(self, state: ControllerState, increment, t) -> ControllerUpdate: ...

    @abstractmethod
    def propose_budget(self, state: ControllerState) -> jax.Array: ...

    def record_budget(self, state, budget, t):
        """Note the budget actually handed to the allocator for origin t, after the guard."""
        return state


def _controller_error(state, increment, t):
    return jnp.where(t == state.last_time, UpdateError.DUPLICATE,
        jnp.where(t != state.last_time + 1, UpdateError.OUT_OF_ORDER,
        jnp.where(~jnp.isfinite(increment), UpdateError.INVALID_INPUT, UpdateError.OK))).astype(jnp.int32)


class ACIController(OuterBudgetController):
    """One projected update b <- clip(b + gamma*g_pre) per physical round."""

    def __init__(self, config: ControllerConfig):
        if not isfinite(config.budget_init) or not 0 <= config.budget_init <= 1:
            raise ValueError("controller.budget_init must be in [0, 1]")
        if not isfinite(config.gamma) or config.gamma <= 0:
            raise ValueError("controller.gamma must be finite and positive")
        self.initial, self.gamma = config.budget_init, config.gamma

    def init_state(self, dtype=None):
        dtype = jnp.asarray(0.0).dtype if dtype is None else dtype
        return ControllerState(jnp.asarray(self.initial, dtype), jnp.asarray(0, dtype),
                               jnp.asarray(0, dtype), jnp.asarray(-1, jnp.int32), jnp.asarray(0, jnp.int32))

    def update(self, state, increment, t):
        increment = jnp.asarray(increment, state.budget.dtype)
        if increment.shape != ():
            raise ValueError("controller input must be one aggregate scalar per physical round")
        time = jnp.asarray(t, jnp.int32)
        error = _controller_error(state, increment, time)
        budget = jnp.clip(state.budget + self.gamma * increment, 0, 1)
        error = jnp.where((error == 0) & ~jnp.isfinite(budget), UpdateError.INVALID_INPUT, error).astype(jnp.int32)
        updated = replace(state, budget=budget, previous_error=increment, last_time=time, updates=state.updates + 1)
        return ControllerUpdate(jax.lax.cond(error == 0, lambda _: updated, lambda _: state, operand=None), error)

    def propose_budget(self, state):
        return state.budget


class PIDBudgetController(ACIController):
    """Budget b0 + Kp*e + Ki*I + Kd*(e-e_previous), projected into [0,1].

    Optional integral_limit clips the integral symmetrically. Zero-feedback
    rounds still advance the derivative clock and preserve accumulated integral.
    """

    def __init__(self, config: ControllerConfig):
        super().__init__(config)
        if any(not isfinite(gain) or gain < 0 for gain in (config.kp, config.ki, config.kd)):
            raise ValueError("PID gains must be finite and nonnegative")
        if config.integral_limit is not None and (not isfinite(config.integral_limit) or config.integral_limit <= 0):
            raise ValueError("controller.integral_limit must be finite and positive")
        self.kp, self.ki, self.kd = config.kp, config.ki, config.kd
        self.integral_limit = config.integral_limit

    def update(self, state, increment, t):
        increment = jnp.asarray(increment, state.budget.dtype)
        if increment.shape != ():
            raise ValueError("controller input must be one aggregate scalar per physical round")
        time = jnp.asarray(t, jnp.int32)
        error = _controller_error(state, increment, time)
        integral = state.integral + increment
        if self.integral_limit is not None:  # Construction-time option.
            integral = jnp.clip(integral, -self.integral_limit, self.integral_limit)
        raw = self.initial + self.kp*increment + self.ki*integral + self.kd*(increment-state.previous_error)
        budget = jnp.clip(raw, 0, 1)
        valid = jnp.isfinite(raw) & jnp.isfinite(integral)
        error = jnp.where((error == 0) & ~valid, UpdateError.INVALID_INPUT, error).astype(jnp.int32)
        updated = replace(state, budget=budget, integral=integral, previous_error=increment,
                          last_time=time, updates=state.updates + 1)
        return ControllerUpdate(jax.lax.cond(error == 0, lambda _: updated, lambda _: state, operand=None), error)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DtACIState:
    budget: jax.Array
    expert_budgets: jax.Array  # (J,)
    expert_weights: jax.Array  # (J,), normalized
    issued_budget: jax.Array  # (H+1,) ring of the budget proposed for origin t, indexed t mod (H+1)
    unit_allocation: jax.Array  # (H,) alpha per unit budget of the last scored ordinary origin
    critical_budget: jax.Array  # Last B_k^* used (NaN when no usable origin)
    last_time: jax.Array
    updates: jax.Array
    expert_updates: jax.Array


class DtACIController(OuterBudgetController):
    """Gibbs & Candes DtACI on the scalar trajectory budget, with delay H.

    Nestedness comes from a scaled counterfactual family alpha_k(b) = b * u_k,
    where u_k = alpha_k / b_k for an ordinary origin (so alpha_k(b_k) is the
    issued allocation) and, for an extreme origin, the unit allocation of the
    last scored ordinary origin (initially uniform 1/H). Then Z_k(b) = 1{B_k^* < b}
    exactly, with critical budget
        B_k^* = min_h critical_{k,h} / u_{k,h},
    and on ordinary origins Z_k(b_k) equals the actual label. Extremes carry the
    ordinary family's counterfactual critical levels, so they are scored too;
    skipping them would freeze the experts whenever an infeasible budget forces
    conservative fallbacks. B_k^* needs every horizon's critical level, so origin
    k = t - H is scored at physical time t; origins with a missing level are
    skipped. Each expert j
    tracks b_j <- clip(b_j + gamma_j (beta - 1{B^* < b_j}), 0, 1); weights use the
    pinball loss of b_j against B^* with exponential weighting and fixed share.
    The proposed budget is the weight-averaged expert budget. The aggregate
    predict-and-correct increment is not used; q accounting is unaffected.
    """

    requires_critical_budget = True
    requires_history = True

    def __init__(self, config: ControllerConfig, beta: float, horizon: int):
        if not isfinite(config.budget_init) or not 0 <= config.budget_init <= 1:
            raise ValueError("controller.budget_init must be in [0, 1]")
        gammas = tuple(config.dtaci_gammas)
        if not gammas or any(not isfinite(g) or g <= 0 for g in gammas):
            raise ValueError("controller.dtaci_gammas must be nonempty and positive")
        if not isfinite(config.dtaci_eta) or config.dtaci_eta <= 0:
            raise ValueError("controller.dtaci_eta must be positive")
        if not isfinite(config.dtaci_sigma) or not 0 <= config.dtaci_sigma < 1:
            raise ValueError("controller.dtaci_sigma must be in [0, 1)")
        self.initial, self.gammas, self.beta, self.horizon = config.budget_init, gammas, beta, horizon
        self.eta, self.sigma = config.dtaci_eta, config.dtaci_sigma

    def init_state(self, dtype=None):
        dtype = jnp.asarray(0.0).dtype if dtype is None else dtype
        j = len(self.gammas)
        return DtACIState(jnp.asarray(self.initial, dtype), jnp.full(j, self.initial, dtype),
                          jnp.full(j, 1/j, dtype), jnp.full(self.horizon+1, jnp.nan, dtype),
                          jnp.full(self.horizon, 1/self.horizon, dtype),
                          jnp.asarray(jnp.nan, dtype), jnp.asarray(-1, jnp.int32),
                          jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32))

    def update(self, state, increment, t, history: HistoryState):
        dtype = state.budget.dtype
        increment = jnp.asarray(increment, dtype)
        time = jnp.asarray(t, jnp.int32)
        error = _controller_error(state, increment, time)
        h = self.horizon
        k = time - h
        slot = jnp.mod(k, history.origin_id.size)
        b_k = state.issued_budget[jnp.mod(k, h+1)]
        alpha = history.issued_alpha[slot]
        crit = history.critical_level[slot]
        present = (k >= 0) & history.valid_origin[slot] & (history.origin_id[slot] == k)
        ordinary = (present & (history.action_mode[slot] == ActionMode.ORDINARY) &
                    jnp.isfinite(b_k) & (b_k > 0) & jnp.all(jnp.isfinite(alpha) & (alpha > 0)))
        unit = jnp.where(ordinary, alpha / jnp.where(ordinary, b_k, 1), state.unit_allocation)
        usable = present & jnp.all(history.critical_resolved[slot])
        critical_budget = jnp.where(usable, jnp.min(crit / unit), jnp.nan)
        gammas = jnp.asarray(self.gammas, dtype)
        experts = state.expert_budgets
        miss = (critical_budget < experts).astype(dtype)
        u = critical_budget - experts
        loss = self.beta * u - jnp.minimum(u, 0)  # pinball loss, as in DtACI.R
        moved = jnp.clip(experts + gammas * (self.beta - miss), 0, 1)
        raw = state.expert_weights * jnp.exp(-self.eta * (loss - jnp.min(loss)))
        raw = raw / jnp.sum(raw)
        shared = (1 - self.sigma) * raw + self.sigma / experts.size
        experts = jnp.where(usable, moved, experts)
        weights = jnp.where(usable, shared, state.expert_weights)
        budget = jnp.sum(weights * experts)
        ring = state.issued_budget.at[jnp.mod(time, h+1)].set(budget)
        error = jnp.where((error == 0) & ~(jnp.isfinite(budget) & jnp.all(jnp.isfinite(weights))),
                          UpdateError.INVALID_INPUT, error).astype(jnp.int32)
        updated = DtACIState(budget, experts, weights, ring,
                             jnp.where(usable & ordinary, unit, state.unit_allocation),
                             jnp.where(usable, critical_budget, state.critical_budget), time,
                             state.updates + 1, state.expert_updates + usable.astype(jnp.int32))
        return ControllerUpdate(jax.lax.cond(error == 0, lambda _: updated, lambda _: state, operand=None), error)

    def propose_budget(self, state):
        return state.budget

    def record_budget(self, state, budget, t):
        # A supervised budget differs from the proposal. Scoring origin t against the
        # budget that produced its alphas keeps Z_k(b_k) equal to the actual label.
        slot = jnp.mod(jnp.asarray(t, jnp.int32), self.horizon+1)
        return replace(state, issued_budget=state.issued_budget.at[slot].set(budget))


class FixedShareGainController(OuterBudgetController):
    requires_critical_budget = False
    requires_counterfactual_loss = True

    def __init__(self, *args, **kwargs):
        self.init_state()

    def init_state(self, *args, **kwargs):
        raise NotImplementedError("Fixed-share gain aggregation is deferred until a principled "
                                  "counterfactual expert loss is specified; actual Z labels only "
                                  "evaluate the issued allocation")

    def update(self, *args, **kwargs):
        self.init_state()

    def propose_budget(self, *args, **kwargs):
        self.init_state()


def build_controller(config: ControllerConfig, beta: float = None, horizon: int = None) -> OuterBudgetController:
    choices = {"aci": ACIController, "pid": PIDBudgetController,
               "dtaci": DtACIController, "fixed_share_gain": FixedShareGainController}
    if config.kind not in choices:
        raise ValueError(f"unknown budget controller: {config.kind!r}")
    if config.kind == "dtaci":
        if beta is None or horizon is None:
            raise ValueError("DtACI requires beta and the horizon to score critical budgets")
        return DtACIController(config, beta, horizon)
    return choices[config.kind](config)


def react(controller: OuterBudgetController, state, increment, t, history: HistoryState):
    """Dispatch one controller update; history-reading controllers receive it."""
    if getattr(controller, "requires_history", False):
        return controller.update(state, increment, t, history)
    return controller.update(state, increment, t)
