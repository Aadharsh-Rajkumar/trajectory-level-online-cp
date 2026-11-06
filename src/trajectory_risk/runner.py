"""Complete numerical online rounds and a Python forecasting-adapter boundary.

Python adapters own forecasting geometry and marginal conformal assumptions.
They supply only predictable issue-time inputs and revealed membership data.
No arbitrary prediction-set object is stored in compiled numerical state.
"""

from dataclasses import dataclass, fields, is_dataclass, replace
from copy import deepcopy
from enum import Enum
import json
from math import isfinite
from pathlib import Path
from typing import Optional, Protocol
from collections.abc import Callable

import jax
import jax.numpy as jnp

from .configs import ExperimentConfig, configure_jax, parse_config
from .history import (ActionMode, AlphaGrid, HistoryState, UpdateError, add_origin, init_history,
                      resolve_horizon, resolve_trajectory)
from .controllers import AccountingState, ControllerState, SafetyDecision, build_accounting_guard, build_controller, react as react_controller
from .signals import SignalPacket, SignalState, build_signal, empty_packet
from .dependence import DependenceEstimate, DependenceState, build_dependence_estimator
from .utility_weights import WeightState, WeightUpdate, build_weight_updater
from .solver import (
    FrozenRiskModel, SolverResult, SolverState, SolverStatus, acceptance_feasible,
    build_risk_model, build_solver, empty_solver_result,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ForecastInputs:
    """Current ordinary-family curves (H,M), availability (H,), covariates (P,).

    The adapter prepares these before solving, using information through time t.
    Availability is explicit; curves need not be supplied for static weights.
    """

    set_size_grid: jax.Array
    set_size_available: jax.Array
    covariates: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class HorizonObservationBatch:
    """H fixed slots, one for each horizon potentially revealed at physical t.

    An active slot supplies actual membership even when its ordinary-family
    critical level is unavailable. Inactive slots have event_mask=False.
    Membership must describe the ACTUAL sets, including extreme actions.
    """

    origin_id: jax.Array
    horizon: jax.Array
    actual_violation: jax.Array
    critical_level: jax.Array
    critical_available: jax.Array
    event_mask: jax.Array


class TrajectoryAdapter(Protocol):
    """External forecasting boundary; methods are called outside JIT.

    On conservative extremes issue whole-space sets at every horizon. On
    aggressive extremes issue an empty horizon-1 set. Their labels are still
    processed at the prescribed physical reveal times, not booked at issuance.
    Whenever possible, supply counterfactual critical levels for the ordinary
    family on these rounds. This package does not certify the adapter's margins.
    """

    def observe(self, t: int) -> HorizonObservationBatch: ...

    def prepare(self, t: int, grid: AlphaGrid) -> ForecastInputs: ...

    def issue(self, t: int, action_mode: ActionMode,
              alpha: Optional[jax.Array]) -> None:
        """alpha is None for extremes; no synthetic ordinary allocation is implied."""
        ...


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ExperimentTotals:
    issued: jax.Array
    resolved: jax.Array
    failures: jax.Array
    ordinary: jax.Array
    conservative: jax.Array
    aggressive: jax.Array
    uncertified_ordinary: jax.Array
    fallback_rounds: jax.Array
    solver_failures: jax.Array
    unconverged_ordinary: jax.Array
    atomic_signal: jax.Array
    controller_signal: jax.Array
    ordinary_set_size: jax.Array
    measured_sizes: jax.Array
    q_min: jax.Array
    q_max: jax.Array


def _init_totals(dtype, q_init):
    zero = jnp.asarray(0, jnp.int32)
    real = jnp.asarray(0, dtype)
    return ExperimentTotals(*([zero]*10), real, real, real, zero,
                            jnp.asarray(q_init, dtype), jnp.asarray(q_init, dtype))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RunnerState:
    """Immutable online numerical state, suitable for downstream JIT/scan code."""

    history: HistoryState
    alpha_reference: jax.Array
    queued_issue_increment: jax.Array  # Consumed once by the next physical round.
    last_issued_time: jax.Array
    accounting: Optional[AccountingState] = None
    signal: Optional[SignalState] = None
    dependence: Optional[DependenceState] = None
    controller: Optional[ControllerState] = None
    utility: Optional[WeightState] = None
    solver: Optional[SolverState] = None
    totals: Optional[ExperimentTotals] = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RunnerUpdate:
    state: RunnerState
    error: jax.Array

    @property
    def accepted(self) -> jax.Array:
        return self.error == UpdateError.OK


def init_runner_state(config: ExperimentConfig) -> RunnerState:
    """Startup boundary: validate, configure precision/device, then allocate."""
    config.validate()
    configure_jax(config.jax)
    h = config.history
    dtype = jnp.float64 if config.jax.enable_x64 else jnp.float32
    values = (jnp.asarray(h.grid_values, dtype) if h.grid_values else
              jnp.linspace(h.alpha_min, h.alpha_max, h.grid_size, dtype=dtype))
    grid = AlphaGrid.create(values)
    history = init_history(grid, h.horizon, h.capacity, h.alpha_min, h.alpha_max, h.covariate_size)
    reference = (jnp.asarray(h.alpha_reference_init, dtype) if h.alpha_reference_init else
                 jnp.full(h.horizon, jnp.clip(config.beta / h.horizon, h.alpha_min, h.alpha_max), dtype))
    guard = build_accounting_guard(config.safety, config.beta)
    signal = build_signal(config.signal, config.beta)
    dependence = build_dependence_estimator(config.dependence, config.t_max)
    controller = build_controller(config.controller, config.beta, config.history.horizon)
    utility = build_weight_updater(config.weights)
    solver = build_solver(config.solver)
    return RunnerState(history, reference, jnp.asarray(0, dtype), jnp.asarray(-1, jnp.int32),
                       guard.init_state(history), signal.init_state(history), dependence.init_state(history),
                       controller.init_state(dtype), utility.init_state(history), solver.init_state(history, reference),
                       _init_totals(dtype, config.safety.q_init))


def record_issuance(state: RunnerState, origin_id, action_mode,
                    inputs: ForecastInputs, alpha=None) -> RunnerUpdate:
    """Record after actual issuance. Extreme actions preserve alpha_reference.

    This only records metadata. The full runner verifies solver acceptance before
    calling the adapter. queued_issue_increment remains explicit and unchanged.
    """
    update = add_origin(state.history, origin_id, action_mode, alpha,
                        inputs.set_size_grid, inputs.set_size_available, inputs.covariates)
    slot = jnp.mod(jnp.asarray(origin_id, jnp.int32), update.state.origin_id.size)
    ordinary = jnp.asarray(action_mode) == ActionMode.ORDINARY
    reference = jnp.where(update.accepted & ordinary,
                           update.state.issued_alpha[slot], state.alpha_reference)
    origin = jnp.asarray(origin_id, state.last_issued_time.dtype)
    result = replace(state, history=update.state, alpha_reference=reference,
                     last_issued_time=jnp.where(update.accepted, origin, state.last_issued_time))
    return RunnerUpdate(result, update.error)


def require_accepted(update) -> None:
    """Host-only check for an operation expected to succeed; call outside JIT."""
    error = UpdateError(int(update.error))
    if error != UpdateError.OK:
        raise ValueError(f"history operation rejected: {error.name}")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FeedbackUpdate:
    state: RunnerState
    packet: SignalPacket
    pre_issuance_increment: jax.Array
    error: jax.Array

    @property
    def accepted(self):
        return self.error == UpdateError.OK


@dataclass(frozen=True)
class FeedbackKernels:
    """Callables selected once; use jax.jit or compose into a downstream scan."""

    observe: Callable
    issue: Callable


def build_feedback_kernels(config: ExperimentConfig) -> FeedbackKernels:
    """Phase-2 feedback orchestration without an ordinary budget controller.

    observe must follow all history resolution at t. Its scalar output includes
    the previous issue queue, and clears that queue exactly once. issue follows
    record_issuance and queues the NEW contribution. Neither operation updates
    an ordinary controller or chooses the current allocation.
    """
    config.validate()
    guard = build_accounting_guard(config.safety, config.beta)
    signal = build_signal(config.signal, config.beta)
    # Resolve concrete methods before defining compiled-callable kernels.
    update_accounting, observe_signal, issue_signal = guard.update, signal.observe, signal.issue

    def observe(state: RunnerState, t):
        if state.accounting is None or state.signal is None:
            raise ValueError("initialize accounting and signals with init_runner_state")
        accounting = update_accounting(state.accounting, state.history, t)
        feedback = observe_signal(state.signal, state.history, t)
        error = jnp.where(accounting.accepted, feedback.error, accounting.error)

        def apply(_):
            updated = replace(state, accounting=accounting.state, signal=feedback.state,
                              queued_issue_increment=jnp.zeros_like(state.queued_issue_increment))
            return FeedbackUpdate(updated, feedback.packet,
                                  feedback.packet.total + state.queued_issue_increment, error)

        return jax.lax.cond(error == UpdateError.OK, apply,
            lambda _: FeedbackUpdate(state, empty_packet(state.history),
                                     jnp.zeros_like(state.queued_issue_increment), error), operand=None)

    def issue(state: RunnerState, t):
        if state.signal is None:
            raise ValueError("initialize signals with init_runner_state")
        feedback = issue_signal(state.signal, state.history, t)

        def apply(_):
            updated = replace(state, signal=feedback.state, queued_issue_increment=feedback.packet.total)
            # Issuance supplies NO input to the already chosen round-t budget.
            return FeedbackUpdate(updated, feedback.packet,
                                  jnp.zeros_like(state.queued_issue_increment), feedback.error)

        return jax.lax.cond(feedback.accepted, apply,
            lambda _: FeedbackUpdate(state, empty_packet(state.history),
                                     jnp.zeros_like(state.queued_issue_increment), feedback.error), operand=None)

    return FeedbackKernels(observe, issue)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PreAllocationUpdate:
    state: RunnerState
    packet: SignalPacket
    pre_issuance_increment: jax.Array
    dependence: DependenceEstimate
    utility: WeightUpdate
    proposal: jax.Array
    safety: SafetyDecision
    error: jax.Array

    @property
    def accepted(self):
        return self.error == UpdateError.OK


def build_preallocation_kernel(config: ExperimentConfig):
    """Phases 2–4 in the specified order, before freezing/solving/issuance.

    Use this instead of a separate feedback.observe call when operating the
    combined layers. Return state changes only if every layer accepted the round.
    Diagnostics from a rejected round must not be used to issue an allocation.
    The complete round kernel adds history resolution, allocation and issuance.
    """
    config.validate()
    feedback = build_feedback_kernels(config)
    controller = build_controller(config.controller, config.beta, config.history.horizon)
    dependence = build_dependence_estimator(config.dependence, config.t_max)
    utility = build_weight_updater(config.weights)
    guard = build_accounting_guard(config.safety, config.beta)
    observe_feedback = feedback.observe
    update_dependence, estimate_dependence = dependence.update, dependence.estimate
    update_weights, propose, protect = utility.update, controller.propose_budget, guard.protect
    record_budget = controller.record_budget

    def before_allocation(state: RunnerState, t, current: ForecastInputs):
        if state.controller is None or state.dependence is None or state.utility is None:
            raise ValueError("initialize all numerical layers with init_runner_state")
        signal = observe_feedback(state, t)
        reaction = react_controller(controller, state.controller, signal.pre_issuance_increment, t,
                                    state.history)
        learned_dependence = update_dependence(state.dependence, state.history, t)
        estimate = estimate_dependence(learned_dependence.state, state.history, t)
        weights = update_weights(state.utility, state.history, state.alpha_reference, t,
                                 current.set_size_grid, current.set_size_available)
        proposal = propose(reaction.state)
        safety = protect(signal.state.accounting, proposal)
        recorded = record_budget(reaction.state, safety.budget, t)
        errors = jnp.stack((signal.error, reaction.error, learned_dependence.error,
                            estimate.error, weights.error, safety.error))
        error = errors[jnp.argmax(errors != 0)]
        updated = replace(signal.state, controller=recorded,
                          dependence=learned_dependence.state, utility=weights.state)
        result = jax.lax.cond(error == 0, lambda _: updated, lambda _: state, operand=None)
        return PreAllocationUpdate(result, signal.packet, signal.pre_issuance_increment,
                                   estimate, weights, proposal, safety, error)

    return before_allocation


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AllocationUpdate:
    state: RunnerState
    risk_model: FrozenRiskModel
    result: SolverResult
    action_mode: jax.Array
    error: jax.Array

    @property
    def accepted(self):
        """A safe action is ready; check action_mode before using result.alpha."""
        return self.error == UpdateError.OK


def build_allocation_kernel(config: ExperimentConfig, history: HistoryState):
    """Freeze Phase-5 inputs and solve Phase 6 on ORDINARY rounds only.

    A failed ordinary solve requests the conservative extreme. A formal safety
    override always takes priority, including the aggressive extreme at high q.
    Neither this kernel nor the solver records/pretends to issue the allocation;
    alpha_reference changes only after actual issuance via record_issuance.
    """
    config.validate()
    freeze = build_risk_model(config.solver, config.risk_mode, history)
    solve = build_solver(config.solver).solve

    def allocate(preallocation: PreAllocationUpdate):
        state = preallocation.state
        if state.solver is None or state.dependence is None or state.utility is None:
            raise ValueError("initialize the solver and numerical layers with init_runner_state")
        model = freeze(state.dependence, preallocation.dependence, state.alpha_reference)
        requested = preallocation.safety.action_mode

        def ordinary(_):
            numerical, result = solve(state.solver, preallocation.safety.budget, state.utility.values, model)
            return replace(state, solver=numerical), result, result.required_action

        def extreme(_):
            result = empty_solver_result(state.solver, SolverStatus.SKIPPED_EXTREME, requested)
            return state, result, requested

        def ready(_):
            updated, result, action = jax.lax.cond(requested == ActionMode.ORDINARY, ordinary, extreme, operand=None)
            return AllocationUpdate(updated, model, result, action, preallocation.error)

        return jax.lax.cond(preallocation.accepted, ready,
            lambda _: AllocationUpdate(state, model, empty_solver_result(state.solver, SolverStatus.INVALID_INPUT),
                                       jnp.asarray(ActionMode.CONSERVATIVE_EXTREME, jnp.int32), preallocation.error), operand=None)

    return allocate


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RoundDiagnostics:
    time: jax.Array
    issued: jax.Array
    action_mode: jax.Array  # -1 during drainage.
    alpha: jax.Array
    reference_before: jax.Array
    reference_after: jax.Array
    q: jax.Array
    proposal: jax.Array
    budget: jax.Array
    weights: jax.Array
    forest_mask: jax.Array
    solver: SolverResult
    edge_sample_count: jax.Array
    edge_certified: jax.Array
    edge_ready: jax.Array
    edge_missing_count: jax.Array
    edge_lifetime_missing_count: jax.Array
    adaptive_regret_valid: jax.Array
    expert_weights: jax.Array
    fallback_mask: jax.Array
    observations: SignalPacket
    issuance: SignalPacket
    controller_increment: jax.Array
    queued_increment: jax.Array
    ordinary_set_size: jax.Array
    horizon_observations: HorizonObservationBatch
    safety_overridden: jax.Array
    utility_ready: jax.Array
    utility_static_fallback: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RoundUpdate:
    state: RunnerState
    diagnostics: RoundDiagnostics
    error: jax.Array

    @property
    def accepted(self):
        return self.error == UpdateError.OK


@dataclass(frozen=True)
class RoundKernels:
    step: Callable
    drain: Callable


def resolve_observations(history: HistoryState, batch: HorizonObservationBatch, t):
    """Require exactly the actual memberships due now; critical data may be missing.

    Slots may be permuted. Missing/duplicate/future actual observations reject the
    whole batch. Early-failed origins still supply their later horizon records.
    """
    h = history.issued_alpha.shape[1]
    if any(value.shape != (h,) for value in jax.tree_util.tree_leaves(batch)):
        raise ValueError("observation arrays must all have shape (H,)")
    horizons = jnp.arange(1, h + 1, dtype=jnp.int32)
    expected = (t-horizons >= 0) & (t-horizons < history.next_origin_id)
    counts = jnp.sum(batch.event_mask[:, None] & (batch.horizon[:, None] == horizons), axis=0)
    valid = (jnp.all(counts == expected.astype(jnp.int32)) &
             jnp.all(~batch.event_mask | ((batch.origin_id == t-batch.horizon) &
                     (batch.horizon >= 1) & (batch.horizon <= h))))
    error = jnp.where(valid, UpdateError.OK, UpdateError.INVALID_INPUT).astype(jnp.int32)

    def event(i, carry):
        current, error = carry

        def apply(_):
            observed = resolve_horizon(current, batch.origin_id[i], batch.horizon[i], t,
                                       batch.actual_violation[i], batch.critical_level[i],
                                       batch.critical_available[i])
            terminal = resolve_trajectory(observed.state, batch.origin_id[i], t)
            terminal_ok = ((terminal.error == UpdateError.OK) |
                           (terminal.error == UpdateError.DUPLICATE) |
                           (terminal.error == UpdateError.NOT_YET_RESOLVED))
            next_error = jnp.where(observed.accepted,
                                   jnp.where(terminal_ok, 0, terminal.error), observed.error)
            return terminal.state, next_error

        return jax.lax.cond(batch.event_mask[i] & (error == 0), apply,
                            lambda _: carry, operand=None)

    updated, error = jax.lax.fori_loop(0, h, event, (history, error))
    return HistoryBatchUpdate(jax.lax.cond(error == 0, lambda _: updated,
                                         lambda _: history, operand=None), error)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class HistoryBatchUpdate:
    state: HistoryState
    error: jax.Array

    @property
    def accepted(self):
        return self.error == UpdateError.OK


def _first_error(*errors):
    values = jnp.stack(errors)
    return values[jnp.argmax(values != 0)]


def _accumulate(state, diagnostics):
    totals, d = state.totals, diagnostics
    resolved = jnp.sum(d.observations.resolved_mask, dtype=jnp.int32)
    failed = jnp.sum(jnp.where(d.observations.resolved_mask, d.observations.resolved_labels, 0), dtype=jnp.int32)
    ordinary = d.issued & (d.action_mode == ActionMode.ORDINARY)
    measured = ordinary & jnp.isfinite(d.ordinary_set_size)
    fallback = d.issued & jnp.any(d.forest_mask & d.fallback_mask)
    solver_failed = d.issued & (d.solver.status != SolverStatus.SKIPPED_EXTREME) & ~d.solver.feasible
    q = state.accounting.q
    updated = replace(totals,
        issued=totals.issued+d.issued.astype(jnp.int32), resolved=totals.resolved+resolved,
        failures=totals.failures+failed, ordinary=totals.ordinary+ordinary.astype(jnp.int32),
        conservative=totals.conservative+(d.issued & (d.action_mode == ActionMode.CONSERVATIVE_EXTREME)),
        aggressive=totals.aggressive+(d.issued & (d.action_mode == ActionMode.AGGRESSIVE_EXTREME)),
        uncertified_ordinary=totals.uncertified_ordinary+(ordinary & ~d.solver.certified),
        fallback_rounds=totals.fallback_rounds+fallback.astype(jnp.int32),
        solver_failures=totals.solver_failures+solver_failed.astype(jnp.int32),
        unconverged_ordinary=totals.unconverged_ordinary+(ordinary & ~d.solver.converged),
        atomic_signal=totals.atomic_signal+d.observations.total+d.issuance.total,
        controller_signal=totals.controller_signal+d.controller_increment,
        ordinary_set_size=totals.ordinary_set_size+jnp.where(measured, d.ordinary_set_size, 0),
        measured_sizes=totals.measured_sizes+measured.astype(jnp.int32),
        q_min=jnp.minimum(totals.q_min, q), q_max=jnp.maximum(totals.q_max, q))
    return replace(state, totals=updated)


def build_round_kernels(config: ExperimentConfig, history: HistoryState) -> RoundKernels:
    """Complete pure rounds; Python issuance commits the returned immutable state.

    Previewing metadata and provisional signals makes validation transactional:
    no bad numerical record can trigger external issuance. The live state is
    committed only AFTER adapter.issue succeeds. No callback is traced by JAX.
    """
    config.validate()
    before = build_preallocation_kernel(config)
    allocate = build_allocation_kernel(config, history)
    feedback = build_feedback_kernels(config)
    controller = build_controller(config.controller, config.beta, config.history.horizon)
    dependence = build_dependence_estimator(config.dependence, config.t_max)
    h = history.issued_alpha.shape[1]
    e = h*(h-1)//2
    dtype = history.grid.values.dtype

    def step(state: RunnerState, batch: HorizonObservationBatch, current: ForecastInputs, t):
        clock_ok = (t >= 0) & (t < config.t_max) & (t == state.history.next_origin_id)
        observed = resolve_observations(state.history, batch, t)
        ready = replace(state, history=observed.state)
        pre = before(ready, t, current)
        allocated = allocate(pre)
        ordinary = allocated.action_mode == ActionMode.ORDINARY
        independently_checked = acceptance_feasible(allocated.result.alpha, pre.safety.budget,
                                                    allocated.risk_model, config.solver.feasibility_tol)
        acceptance_error = jnp.where(ordinary & ~(allocated.result.feasible & independently_checked),
                                      UpdateError.INVALID_INPUT, 0)
        alpha = jnp.where(ordinary, allocated.result.alpha, jnp.full(h, jnp.nan, dtype))
        recorded = record_issuance(allocated.state, t, allocated.action_mode, current, alpha)
        issued = feedback.issue(recorded.state, t)
        error = _first_error(jnp.where(clock_ok, 0, UpdateError.OUT_OF_ORDER), observed.error,
                             pre.error, allocated.error, acceptance_error, recorded.error, issued.error)
        sizes = jax.vmap(lambda a, curve: jnp.interp(a, history.grid.values, curve))(alpha, current.set_size_grid)
        measured = jnp.where(ordinary & jnp.all(current.set_size_available), jnp.sum(sizes), jnp.nan)
        est, model = pre.dependence, allocated.risk_model
        d = RoundDiagnostics(jnp.asarray(t, jnp.int32), jnp.asarray(True), allocated.action_mode,
            alpha, state.alpha_reference, issued.state.alpha_reference, issued.state.accounting.q,
            pre.proposal, pre.safety.budget, issued.state.utility.values, model.forest_mask,
            allocated.result, est.sample_count, est.certified, est.ready, est.missing_count,
            est.lifetime_missing_count, est.adaptive_regret_valid, est.expert_weights, model.fallback_mask,
            pre.packet, issued.packet, pre.pre_issuance_increment, issued.state.queued_issue_increment,
            measured, batch, pre.safety.overridden, pre.utility.ready, pre.utility.used_static_fallback)
        updated = _accumulate(issued.state, d)
        return RoundUpdate(jax.lax.cond(error == 0, lambda _: updated, lambda _: state, operand=None), d, error)

    def drain(state: RunnerState, batch: HorizonObservationBatch, t):
        clock_ok = ((state.history.next_origin_id == config.t_max) & (t >= config.t_max) &
                    (t < config.t_max+h))
        observed = resolve_observations(state.history, batch, t)
        ready = replace(state, history=observed.state)
        signal = feedback.observe(ready, t)
        reaction = react_controller(controller, state.controller, signal.pre_issuance_increment, t,
                                    ready.history)
        learned = dependence.update(state.dependence, ready.history, t)
        est = dependence.estimate(learned.state, ready.history, t)
        error = _first_error(jnp.where(clock_ok, 0, UpdateError.OUT_OF_ORDER), observed.error,
                             signal.error, reaction.error, learned.error, est.error)
        updated = replace(signal.state, controller=reaction.state, dependence=learned.state)
        proposal = controller.propose_budget(reaction.state)
        d = RoundDiagnostics(jnp.asarray(t, jnp.int32), jnp.asarray(False), jnp.asarray(-1, jnp.int32),
            jnp.full(h, jnp.nan, dtype), state.alpha_reference, state.alpha_reference, updated.accounting.q,
            proposal, jnp.asarray(jnp.nan, dtype), updated.utility.values, jnp.zeros(e, bool),
            empty_solver_result(state.solver, SolverStatus.SKIPPED_EXTREME),
            est.sample_count, est.certified, est.ready, est.missing_count, est.lifetime_missing_count,
            est.adaptive_regret_valid, est.expert_weights, jnp.zeros(e, bool), signal.packet,
            empty_packet(history), signal.pre_issuance_increment, updated.queued_issue_increment,
            jnp.asarray(jnp.nan, dtype), batch, jnp.asarray(False), jnp.asarray(False), jnp.asarray(False))
        updated = _accumulate(updated, d)
        return RoundUpdate(jax.lax.cond(error == 0, lambda _: updated, lambda _: state, operand=None), d, error)

    return RoundKernels(step, drain)


def _json_value(value):
    """Strict portable JSON: unavailable/undefined numerical diagnostics are null."""
    if isinstance(value, HorizonObservationBatch):
        result = {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
        # Infinite critical levels are valid family endpoints, not missing data.
        result["critical_level"] = [
            level if isfinite(level) else "+inf" if level == float("inf") else
            "-inf" if level == -float("inf") else None
            for level in value.critical_level.tolist()]
        return result
    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


def summarize(state: RunnerState, config: ExperimentConfig) -> dict:
    counts = _json_value(state.totals)
    n, resolved, failed = counts["issued"], counts["resolved"], counts["failures"]
    pending = n-resolved
    q = float(state.accounting.q)
    accounting_sum = config.beta*resolved-failed
    bound = (abs(q-config.safety.q_init)/config.safety.eta_q + config.history.horizon)/n if n else None
    return {
        "risk_mode": config.risk_mode.value,
        "seed": config.runner.seed,
        "issued": n, "resolved": resolved, "failures": failed, "pending": pending,
        "failure_rate_interval": [failed/n, (failed+pending)/n] if n else None,
        "failure_rate": failed/n if n and pending == 0 else None,
        "calibration_error": abs(failed/n-config.beta) if n and pending == 0 else None,
        "accounting_bound": bound,
        "q": q,
        "accounting_identity_error": q-config.safety.q_init-config.safety.eta_q*accounting_sum,
        "signal_identity_error": counts["atomic_signal"]-accounting_sum if pending == 0 else None,
        "queue_identity_error": counts["atomic_signal"]-counts["controller_signal"]-float(state.queued_issue_increment),
        "ordinary": counts["ordinary"], "conservative_extreme": counts["conservative"],
        "aggressive_extreme": counts["aggressive"],
        "uncertified_ordinary": counts["uncertified_ordinary"],
        "fallback_rounds": counts["fallback_rounds"], "solver_failures": counts["solver_failures"],
        "unconverged_ordinary": counts["unconverged_ordinary"],
        "mean_ordinary_set_size": counts["ordinary_set_size"]/counts["measured_sizes"] if counts["measured_sizes"] else None,
        "measured_sizes": counts["measured_sizes"], "q_min": counts["q_min"], "q_max": counts["q_max"],
        "missing_critical_by_horizon": _json_value(state.history.missing_critical_count),
        "validity_scope": "Acceptance certificates require the stated marginal and stable conditional-law assumptions; model KKT diagnostics do not imply global optimality.",
    }


@dataclass(frozen=True)
class ExperimentResult:
    state: RunnerState
    summary: dict
    trace: tuple[dict, ...]


class OnlineRunner:
    """Host adapter boundary with compiled numerical rounds and bounded-memory logs.

    A callback failure aborts this runner: an external side effect cannot safely
    be retried automatically. Wrong physical times are rejected before callbacks.
    """

    def __init__(self, config: ExperimentConfig, adapter: TrajectoryAdapter):
        self.config = deepcopy(config)
        self.state = init_runner_state(self.config)
        self.adapter = adapter
        kernels = build_round_kernels(self.config, self.state.history)
        self._step = jax.jit(kernels.step) if config.runner.jit else kernels.step
        self._drain = jax.jit(kernels.drain) if config.runner.jit else kernels.drain
        self.next_time = 0
        self.aborted = False
        self._trace = []

    def step(self, t: Optional[int] = None) -> RoundDiagnostics:
        cfg = self.config
        t = self.next_time if t is None else t
        if self.aborted:
            raise RuntimeError("runner aborted after a callback or numerical failure")
        end = cfg.t_max + (cfg.history.horizon if cfg.runner.drain else 0)
        if type(t) is not int or t != self.next_time or not 0 <= t < end:
            raise ValueError("physical rounds must be consecutive and within the configured run")
        try:
            observations = self.adapter.observe(t)
            if t < cfg.t_max:
                current = self.adapter.prepare(t, self.state.history.grid)
                update = self._step(self.state, observations, current, t)
            else:
                update = self._drain(self.state, observations, t)
            require_accepted(update)
            d = update.diagnostics
            if bool(d.issued):
                mode = ActionMode(int(d.action_mode))
                # The numerical round already independently checked acceptance.
                self.adapter.issue(t, mode, d.alpha if mode == ActionMode.ORDINARY else None)
            self.state = update.state
            self.next_time += 1
            if cfg.runner.collect_trace:
                self._trace.append(_json_value(d))
            return d
        except Exception:
            self.aborted = True
            raise

    def run(self) -> ExperimentResult:
        cfg = self.config
        output = Path(cfg.runner.output_dir) if cfg.runner.output_dir else None
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            # Refuse an existing transcript rather than silently destroying a run.
            if any((output/name).exists() for name in ("config.yaml", "trace.jsonl", "summary.json")):
                raise FileExistsError("output directory already contains experiment artifacts")
            if self.next_time:
                raise ValueError("artifact logging must begin before the first physical round")
            import pyrallis
            with (output/"config.yaml").open("x") as stream:
                pyrallis.dump(cfg, stream)
        end = cfg.t_max + (cfg.history.horizon if cfg.runner.drain else 0)
        stream = (output/"trace.jsonl").open("x") if output else None
        try:
            while self.next_time < end:
                diagnostics = self.step()
                if stream:
                    stream.write(json.dumps(_json_value(diagnostics), allow_nan=False)+"\n")
        finally:
            if stream:
                stream.close()
        summary = summarize(self.state, cfg)
        if output:
            with (output/"summary.json").open("x") as stream:
                json.dump(summary, stream, indent=2, allow_nan=False)
                stream.write("\n")
        return ExperimentResult(self.state, summary, tuple(self._trace))


def run_experiment(config: ExperimentConfig, adapter: TrajectoryAdapter) -> ExperimentResult:
    return OnlineRunner(config, adapter).run()


def main(args=None):
    """Pyrallis CLI for the synthetic interval fixture; adapters stay replaceable."""
    cfg = parse_config(args)
    configure_jax(cfg.jax)
    from .examples import SyntheticIntervalAdapter
    result = run_experiment(cfg, SyntheticIntervalAdapter(cfg))
    print(json.dumps(result.summary, indent=2, allow_nan=False))
