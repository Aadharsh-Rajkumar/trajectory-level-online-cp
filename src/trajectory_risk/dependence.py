"""Delayed lower-orthant overlap estimators, not correlation estimators.

Update once per physical round before recording the new origin. update scores
delayed predictions FIRST, then snapshots predictions for origin t. estimate is
a read-only query of that updated state at t. Missing pairs never backfill the
specified origin windows or create invented losses. Stable certificates and
available-observation point diagnostics have deliberately separate fields.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from math import isfinite

import jax
import jax.numpy as jnp

from .configs import DependenceConfig, RiskMode
from .history import AlphaGrid, HistoryState, UpdateError, lower_orthant_indicators


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DependenceState:
    grid: AlphaGrid
    edges: jax.Array
    delays: jax.Array
    counts: jax.Array  # Lifetime (E,M,M) CDF counts, even after history eviction.
    sample_count: jax.Array
    required_count: jax.Array
    missing_count: jax.Array
    expert_counts: jax.Array  # (E,J,M,M), rolling CDF counts for window experts.
    expert_sample_count: jax.Array
    expert_missing_count: jax.Array
    missing_pattern: jax.Array  # (E,W_max), newest to oldest, no backfill.
    weights: jax.Array  # (E,H,J), initially 1; normalize before prediction.
    residue_mask: jax.Array
    n_max: jax.Array
    share: jax.Array
    pending_origin: jax.Array  # (H+1,E); zero slots for nonadaptive estimators.
    pending_valid: jax.Array
    pending_predictions: jax.Array  # (H+1,E,J,M,M)
    pending_weights: jax.Array
    expert_losses: jax.Array
    mixture_losses: jax.Array
    loss_mask: jax.Array
    cumulative_expert_loss: jax.Array
    cumulative_mixture_loss: jax.Array
    scored_count: jax.Array
    adaptive_regret_valid: jax.Array  # False after any eligible missing prediction/outcome.
    last_time: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DependenceEstimate:
    point_surface: jax.Array
    lower_surface: jax.Array
    radius: jax.Array
    sample_count: jax.Array
    certified: jax.Array  # The LOWER surface has a current-time stable certificate.
    ready: jax.Array
    missing_count: jax.Array  # Missing entries in the relevant history/window.
    lifetime_missing_count: jax.Array
    missing_pattern: jax.Array
    expert_surfaces: jax.Array
    expert_sample_count: jax.Array
    expert_weights: jax.Array
    expert_losses: jax.Array
    mixture_losses: jax.Array
    loss_mask: jax.Array
    adaptive_regret_valid: jax.Array
    error: jax.Array

    @property
    def mixture_surface(self):
        return self.point_surface


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DependenceUpdate:
    state: DependenceState
    error: jax.Array

    @property
    def accepted(self):
        return self.error == UpdateError.OK


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class OverlapCredit:
    surface: jax.Array
    certified: jax.Array  # Credit policy, not a marginal-validity claim.
    fallback_mask: jax.Array


class PairwiseDependenceEstimator(ABC):
    @abstractmethod
    def init_state(self, history: HistoryState, edges=None) -> DependenceState: ...

    @abstractmethod
    def update(self, state: DependenceState, history: HistoryState, t) -> DependenceUpdate: ...

    @abstractmethod
    def estimate(self, state: DependenceState, history: HistoryState, t) -> DependenceEstimate: ...


def candidate_edges(horizon: int):
    return jnp.asarray([(s, r) for s in range(1, horizon + 1) for r in range(s + 1, horizon + 1)],
                       jnp.int32).reshape((-1, 2))


def _initialize(history, edges, windows, t_max, adaptive):
    h, m = history.issued_alpha.shape[1], history.grid.values.size
    edges = candidate_edges(h) if edges is None else jnp.asarray(edges, jnp.int32)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("candidate edges must have shape (E,2)")
    pairs = edges.tolist()  # Boundary validation only, never inside JIT.
    if len(set(tuple(pair) for pair in pairs)) != len(pairs) or any(not 1 <= s < r <= h for s, r in pairs):
        raise ValueError("candidate edges must be distinct pairs with 1 <= s < r <= H")
    if windows and history.origin_id.size < h + max(windows):
        raise ValueError("history capacity must retain H + largest dependence window origins")
    e, j = edges.shape[0], max(len(windows), 1)
    dtype = history.grid.values.dtype
    delays = edges[:, 1]
    n_max = (t_max + delays - 1) // delays
    if adaptive and bool(jnp.any(n_max < 2)):
        raise ValueError("adaptive dependence requires ceil(t_max / d_e) >= 2")
    pending = h + 1 if adaptive else 0
    return DependenceState(
        history.grid, edges, delays, jnp.zeros((e, m, m), jnp.int32),
        jnp.zeros(e, jnp.int32), jnp.zeros(e, jnp.int32), jnp.zeros(e, jnp.int32),
        jnp.zeros((e, j, m, m), jnp.int32), jnp.zeros((e, j), jnp.int32),
        jnp.zeros((e, j), jnp.int32), jnp.zeros((e, max(windows, default=0)), bool),
        jnp.ones((e, h, j), dtype), jnp.arange(h)[None, :] < delays[:, None], n_max,
        1 / n_max.astype(dtype), jnp.full((pending, e), -1, jnp.int32),
        jnp.zeros((pending, e), bool), jnp.zeros((pending, e, j, m, m), dtype),
        jnp.zeros((pending, e, j), dtype), jnp.full((e, j), jnp.nan, dtype),
        jnp.full(e, jnp.nan, dtype), jnp.zeros(e, bool), jnp.zeros((e, j), dtype),
        jnp.zeros(e, dtype), jnp.zeros(e, jnp.int32), jnp.ones(e, bool), jnp.asarray(-1, jnp.int32))


def _pairs(history, state, origins, t, t_max):
    """origins has shape (E,) or (E,N); pair gathering stays edge-specific."""
    slots = jnp.mod(origins, history.origin_id.size)
    indices = state.edges - 1
    if origins.ndim == 2:  # Static shape branch.
        indices = indices[:, None, :]
    levels = history.critical_level[slots[..., None], indices]
    available = history.critical_resolved[slots[..., None], indices]
    reveal = history.horizon_reveal_time[slots[..., None], indices]
    present = history.valid_origin[slots] & (history.origin_id[slots] == origins)
    required = (origins >= 0) & (origins < history.next_origin_id) & (origins < t_max)
    mask = required & present & jnp.all(available & (reveal <= t), axis=-1)
    return levels, mask, required, present


def _incoming(state, history, t, t_max):
    origins = jnp.asarray(t, jnp.int32) - state.delays
    levels, available, required, _ = _pairs(history, state, origins, t, t_max)
    events = jnp.where(available[:, None, None], lower_orthant_indicators(levels, state.grid), False).astype(jnp.int32)
    return origins, events, available, required


def _clock_error(state, t, t_max, horizon):
    time = jnp.asarray(t, jnp.int32)
    return jnp.where(time == state.last_time, UpdateError.DUPLICATE,
                     jnp.where((time != state.last_time + 1) | (time >= t_max + horizon),
                               UpdateError.OUT_OF_ORDER, UpdateError.OK)).astype(jnp.int32)


def _add_lifetime(state, events, available, required, t):
    return replace(state, counts=state.counts + events,
        sample_count=state.sample_count + available.astype(jnp.int32),
        required_count=state.required_count + required.astype(jnp.int32),
        missing_count=state.missing_count + (required & ~available).astype(jnp.int32),
        last_time=jnp.asarray(t, jnp.int32))


def fixed_share_update(weights, losses, eta, share):
    """The note's exact unnormalized multiplicative-then-share equations."""
    share = jnp.asarray(share, weights.dtype)
    multiplied = weights * jnp.exp(-eta * losses)
    return (1 - share[..., None]) * multiplied + share[..., None] * jnp.sum(multiplied, axis=-1, keepdims=True) / weights.shape[-1]


def _probabilities(weights):
    return weights / jnp.sum(weights, axis=-1, keepdims=True)


def _score_delayed(state, origins, events, available, required):
    e = jnp.arange(state.edges.shape[0], dtype=jnp.int32)
    slots = jnp.mod(origins, state.pending_origin.shape[0])
    stored = state.pending_origin[slots, e] == origins
    valid = stored & state.pending_valid[slots, e] & available & required
    predictions = state.pending_predictions[slots, e]
    losses = jnp.mean((events[:, None].astype(predictions.dtype) - predictions) ** 2, axis=(-2, -1))
    past_mixture = jnp.sum(state.pending_weights[slots, e, :, None, None] * predictions, axis=1)
    mixture_loss = jnp.mean((events.astype(predictions.dtype) - past_mixture) ** 2, axis=(-2, -1))
    residue = jnp.mod(origins, state.delays)
    old_weights = state.weights[e, residue]
    return valid, losses, mixture_loss, e, slots, residue, old_weights


def _update_windows(state, history, t, windows, t_max, eta, adaptive):
    error = _clock_error(state, t, t_max, history.issued_alpha.shape[1])
    origins, incoming, available, required = _incoming(state, history, t, t_max)
    window = jnp.asarray(windows, jnp.int32)
    outgoing_origins = origins[:, None] - window[None, :]
    levels, outgoing_available, outgoing_required, outgoing_present = _pairs(history, state, outgoing_origins, t, t_max)
    outgoing = jnp.where(outgoing_available[..., None, None], lower_orthant_indicators(levels, state.grid), False).astype(jnp.int32)
    # The oldest outgoing origin is still retained BEFORE round-t issuance.
    unsafe = jnp.any(outgoing_required & ~outgoing_present)
    error = jnp.where((error == 0) & unsafe, UpdateError.UNSAFE_OVERWRITE, error).astype(jnp.int32)

    def apply(_):
        updated = _add_lifetime(state, incoming, available, required, t)
        if adaptive:  # Factory-selected static structure, not runtime dispatch.
            valid, losses, mixture_loss, e, pending_slots, residue, old_weights = _score_delayed(
                state, origins, incoming, available, required)
            shared = fixed_share_update(old_weights, losses, eta, state.share)
            # Homogeneity permits common normalization without changing the note's pi.
            weights = state.weights.at[e, residue].set(jnp.where(valid[:, None], _probabilities(shared), old_weights))
            eligible = required & (origins >= state.delays + max(windows) - 1)
            stored = state.pending_origin[pending_slots, e] == origins
            updated = replace(updated, weights=weights,
                pending_valid=state.pending_valid.at[pending_slots, e].set(jnp.where(
                    stored, False, state.pending_valid[pending_slots, e])),
                expert_losses=jnp.where(valid[:, None], losses, jnp.nan),
                mixture_losses=jnp.where(valid, mixture_loss, jnp.nan), loss_mask=valid,
                cumulative_expert_loss=state.cumulative_expert_loss + jnp.where(valid[:, None], losses, 0),
                cumulative_mixture_loss=state.cumulative_mixture_loss + jnp.where(valid, mixture_loss, 0),
                scored_count=state.scored_count + valid.astype(jnp.int32),
                adaptive_regret_valid=state.adaptive_regret_valid & (~eligible | valid))
        expert_counts = state.expert_counts + incoming[:, None] - outgoing
        expert_n = state.expert_sample_count + available[:, None].astype(jnp.int32) - outgoing_available.astype(jnp.int32)
        ages = jnp.arange(max(windows), dtype=jnp.int32)
        cohort_origins = origins[:, None] - ages[None, :]
        _, cohort_available, cohort_required, _ = _pairs(history, state, cohort_origins, t, t_max)
        missing = cohort_required & ~cohort_available
        expert_missing = jnp.sum(missing[:, None, :] & (ages[None, None, :] < window[None, :, None]), axis=-1, dtype=jnp.int32)
        updated = replace(updated, expert_counts=expert_counts, expert_sample_count=expert_n,
                          expert_missing_count=expert_missing, missing_pattern=missing)
        if adaptive:
            ready = jnp.all(expert_n == window[None, :], axis=-1)
            e = jnp.arange(state.edges.shape[0], dtype=jnp.int32)
            residue = jnp.mod(jnp.asarray(t, jnp.int32), state.delays)
            pi = _probabilities(updated.weights[e, residue])
            surfaces = expert_counts.astype(state.grid.values.dtype) / jnp.maximum(expert_n[..., None, None], 1)
            pending_slot = jnp.mod(jnp.asarray(t, jnp.int32), history.issued_alpha.shape[1] + 1)
            predict_origin = t < t_max
            snapshot_valid = ready & predict_origin
            eligible_prediction = predict_origin & (t >= state.delays + max(windows) - 1)
            updated = replace(updated,
                pending_origin=updated.pending_origin.at[pending_slot].set(jnp.where(predict_origin, t, -1).astype(jnp.int32)),
                pending_valid=updated.pending_valid.at[pending_slot].set(snapshot_valid),
                pending_predictions=updated.pending_predictions.at[pending_slot].set(surfaces),
                pending_weights=updated.pending_weights.at[pending_slot].set(pi),
                adaptive_regret_valid=updated.adaptive_regret_valid & (~eligible_prediction | ready))
        valid_counts = jnp.all((expert_n >= 0) & (expert_n <= window[None, :])) & jnp.all(expert_counts >= 0)
        return jax.lax.cond(valid_counts,
            lambda _: DependenceUpdate(updated, error),
            lambda _: DependenceUpdate(state, jnp.asarray(UpdateError.INVALID_INPUT, jnp.int32)), operand=None)

    return jax.lax.cond(error == 0, apply, lambda _: DependenceUpdate(state, error), operand=None)


def _estimate(state, t, delta, stable, t_max, windows, adaptive):
    dtype = state.grid.values.dtype
    same_time = jnp.asarray(t, jnp.int32) == state.last_time
    complexity = state.grid.values.size ** 2 * jnp.sum(state.delays.astype(dtype))
    if windows:  # Static estimator choice.
        expert_n = state.expert_sample_count
        experts = state.expert_counts.astype(dtype) / jnp.maximum(expert_n[..., None, None], 1)
        e = jnp.arange(state.edges.shape[0], dtype=jnp.int32)
        pi = _probabilities(state.weights[e, jnp.mod(jnp.asarray(t, jnp.int32), state.delays)])
        point = jnp.sum(pi[..., None, None] * experts, axis=1)
        w = jnp.asarray(windows, dtype)
        log_term = jnp.log(jnp.maximum(complexity, 1)) + 2 * jnp.log(jnp.asarray(t_max, dtype)) - jnp.log(jnp.asarray(delta, dtype))
        expert_radius = jnp.sqrt(state.delays[:, None].astype(dtype) * log_term / (2 * w[None, :]))
        radius = jnp.sum(pi * expert_radius, axis=-1)
        ready = jnp.all(expert_n == w[None, :], axis=-1)
        missing = state.expert_missing_count[:, -1]
        certified = ready & stable & (missing == 0) & (t < t_max)
        sample = expert_n[:, -1]
    else:
        sample = state.sample_count
        point = state.counts.astype(dtype) / jnp.maximum(sample[:, None, None], 1)
        n = jnp.maximum(sample, 1).astype(dtype)
        log_term = jnp.log(jnp.asarray(jnp.pi ** 2 / 3, dtype)) + jnp.log(jnp.maximum(complexity, 1)) + 2 * jnp.log(n) - jnp.log(jnp.asarray(delta, dtype))
        radius = jnp.sqrt(state.delays.astype(dtype) * log_term / (2 * n))
        ready = sample > 0
        missing = state.missing_count
        certified = ready & stable & (missing == 0)
        experts, expert_n, pi = point[:, None], sample[:, None], jnp.ones((sample.size, 1), dtype)
    ready &= same_time
    certified &= same_time
    lower = jnp.where(certified[:, None, None], jnp.maximum(point - radius[:, None, None], 0), 0)
    return DependenceEstimate(
        jnp.where(same_time, point, 0), lower, jnp.where(ready, radius, jnp.inf),
        jnp.where(same_time, sample, 0), certified, ready,
        jnp.where(same_time, missing, 0), jnp.where(same_time, state.missing_count, 0),
        state.missing_pattern & same_time, jnp.where(same_time, experts, 0),
        jnp.where(same_time, expert_n, 0), jnp.where(same_time, pi, 0),
        jnp.where(same_time, state.expert_losses, jnp.nan), jnp.where(same_time, state.mixture_losses, jnp.nan),
        state.loss_mask & same_time, state.adaptive_regret_valid & same_time & adaptive,
        jnp.where(same_time, UpdateError.OK, UpdateError.OUT_OF_ORDER).astype(jnp.int32))


class FullHistoryDependenceEstimator(PairwiseDependenceEstimator):
    def __init__(self, config: DependenceConfig, t_max: int):
        if type(t_max) is not int or t_max < 1 or not isfinite(config.delta) or not 0 < config.delta < 1:
            raise ValueError("dependence requires positive t_max and delta in (0, 1)")
        if config.assumption not in ("stable_certified", "unassumed"):
            raise ValueError("v1 supports stable_certified or unassumed dependence")
        self.delta, self.stable, self.t_max = config.delta, config.assumption == "stable_certified", t_max

    def init_state(self, history, edges=None):
        return _initialize(history, edges, (), self.t_max, False)

    def update(self, state, history, t):
        error = _clock_error(state, t, self.t_max, history.issued_alpha.shape[1])
        _, events, available, required = _incoming(state, history, t, self.t_max)
        return DependenceUpdate(jax.lax.cond(error == 0,
            lambda s: _add_lifetime(s, events, available, required, t), lambda s: s, state), error)

    def estimate(self, state, history, t):
        return _estimate(state, t, self.delta, self.stable, self.t_max, (), False)


class FixedWindowDependenceEstimator(FullHistoryDependenceEstimator):
    def __init__(self, config: DependenceConfig, t_max: int):
        super().__init__(config, t_max)
        if type(config.window) is not int or config.window < 1:
            raise ValueError("dependence.window must be a positive integer")
        self.windows = (config.window,)

    def init_state(self, history, edges=None):
        return _initialize(history, edges, self.windows, self.t_max, False)

    def update(self, state, history, t):
        return _update_windows(state, history, t, self.windows, self.t_max, 0, False)

    def estimate(self, state, history, t):
        return _estimate(state, t, self.delta, self.stable, self.t_max, self.windows, False)


class AdaptiveWindowDependenceEstimator(FixedWindowDependenceEstimator):
    def __init__(self, config: DependenceConfig, t_max: int):
        super().__init__(config, t_max)
        if (not config.windows or any(type(w) is not int or w < 1 for w in config.windows)
                or any(a >= b for a, b in zip(config.windows, config.windows[1:]))):
            raise ValueError("adaptive dependence requires strictly increasing positive windows")
        if not isfinite(config.meta_eta) or not 0 < config.meta_eta <= 0.5:
            raise ValueError("dependence.meta_eta must be in (0, 0.5]")
        self.windows, self.eta = tuple(config.windows), config.meta_eta

    def init_state(self, history, edges=None):
        return _initialize(history, edges, self.windows, self.t_max, True)

    def update(self, state, history, t):
        return _update_windows(state, history, t, self.windows, self.t_max, self.eta, True)

    def estimate(self, state, history, t):
        return _estimate(state, t, self.delta, self.stable, self.t_max, self.windows, True)


def build_dependence_estimator(config: DependenceConfig, t_max: int) -> PairwiseDependenceEstimator:
    choices = {"full_history": FullHistoryDependenceEstimator,
               "fixed_window": FixedWindowDependenceEstimator, "adaptive": AdaptiveWindowDependenceEstimator}
    if config.kind not in choices:
        raise ValueError(f"unknown dependence estimator: {config.kind!r}")
    return choices[config.kind](config, t_max)


def build_overlap_credit_selector(mode: RiskMode):
    if not isinstance(mode, RiskMode):
        raise ValueError("risk_mode must be a RiskMode")
    experimental = mode is RiskMode.EXPERIMENTAL_PLUGIN

    def select(estimate: DependenceEstimate) -> OverlapCredit:
        usable = estimate.certified & estimate.ready & (estimate.error == 0)
        point = jnp.where(estimate.error == 0, estimate.point_surface, 0)
        surface = point if experimental else jnp.where(usable[:, None, None], estimate.lower_surface, 0)
        return OverlapCredit(surface, jnp.asarray(not experimental),
                             jnp.zeros_like(usable) if experimental else ~usable)

    return select
