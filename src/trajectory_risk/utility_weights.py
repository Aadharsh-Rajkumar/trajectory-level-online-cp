"""Predictable horizon value of risk, independent of ordinary budget control.

Targets are L1-normalized (writeup eq. 17), projected onto the bounded simplex,
then smoothed and step-limited. Current ordinary-family size curves determine
efficiency; mature origin cohorts determine fragility. Missing curves/cohorts
use the static application target.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from math import isfinite

import jax
import jax.numpy as jnp

from .configs import WeightConfig
from .history import HistoryState, UpdateError, first_failure_histogram, marginal_set_efficiency


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class WeightState:
    values: jax.Array
    application: jax.Array
    last_time: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class WeightUpdate:
    state: WeightState
    target: jax.Array
    marginal_efficiency: jax.Array
    first_failure_mass: jax.Array
    ready: jax.Array
    used_static_fallback: jax.Array
    error: jax.Array

    @property
    def accepted(self):
        return self.error == UpdateError.OK


class UtilityWeightUpdater(ABC):
    @abstractmethod
    def init_state(self, history: HistoryState) -> WeightState: ...

    @abstractmethod
    def update(self, state: WeightState, history: HistoryState, alpha_reference, t,
               current_size_grid=None, current_available=None) -> WeightUpdate: ...

    @abstractmethod
    def weights(self, state: WeightState) -> jax.Array: ...


def bounded_simplex_projection(values, lower, upper):
    """Euclidean projection using fixed-iteration scalar-dual bisection."""
    values = jnp.asarray(values)
    if not jnp.issubdtype(values.dtype, jnp.floating):
        values = values.astype(jnp.asarray(0.0).dtype)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bounded-simplex projection requires a nonempty vector")
    lower, upper = jnp.asarray(lower, values.dtype), jnp.asarray(upper, values.dtype)
    centered = values - jnp.max(values)  # Translation invariance improves precision.
    lo, hi = jnp.min(centered)-upper, jnp.max(centered)-lower

    def body(_, bounds):
        lo, hi = bounds
        mid = lo + (hi-lo)/2
        total = jnp.sum(jnp.clip(centered-mid, lower, upper))
        return jnp.where(total > 1, mid, lo), jnp.where(total > 1, hi, mid)

    lo, hi = jax.lax.fori_loop(0, 64, body, (lo, hi))
    return jnp.clip(centered - (lo + (hi-lo)/2), lower, upper)


def smooth_and_cap(current, target, smoothing, max_step):
    ema = (1-smoothing)*current + smoothing*target
    if max_step is None:  # Construction-time configuration.
        return ema
    difference = ema-current
    epsilon = jnp.finfo(current.dtype).eps
    factor = jnp.minimum(1, max_step/(jnp.linalg.norm(difference)+epsilon))
    return current + factor*difference


def _clock_error(state, t):
    return jnp.where(t == state.last_time, UpdateError.DUPLICATE,
        jnp.where(t != state.last_time + 1, UpdateError.OUT_OF_ORDER, UpdateError.OK)).astype(jnp.int32)


class StaticWeightUpdater(UtilityWeightUpdater):
    def __init__(self, config: WeightConfig):
        if not (isfinite(config.min_weight) and isfinite(config.max_weight)
                and 0 < config.min_weight <= config.max_weight <= 1):
            raise ValueError("weight bounds must satisfy 0 < min <= max <= 1")
        if not isfinite(config.smoothing) or not 0 < config.smoothing <= 1:
            raise ValueError("weight smoothing must be in (0, 1]")
        if config.max_step_norm is not None and (not isfinite(config.max_step_norm) or config.max_step_norm < 0):
            raise ValueError("max_step_norm must be finite and nonnegative")
        self.application = tuple(config.application_weights)
        self.lower, self.upper = config.min_weight, config.max_weight
        self.smoothing, self.max_step = config.smoothing, config.max_step_norm

    def init_state(self, history):
        h = history.issued_alpha.shape[1]
        if not h*self.lower <= 1 <= h*self.upper:
            raise ValueError("weight bounds make the normalized simplex infeasible")
        if self.application and (len(self.application) != h or any(not isfinite(x) or x <= 0 for x in self.application)):
            raise ValueError("application weights must contain H finite positive entries")
        raw = jnp.asarray(self.application, history.grid.values.dtype) if self.application else jnp.ones(h, history.grid.values.dtype)
        application = raw/jnp.sum(raw)
        if not bool(jnp.all((application >= self.lower) & (application <= self.upper))):
            raise ValueError("normalized application weights must satisfy the configured weight bounds")
        return WeightState(application, application, jnp.asarray(-1, jnp.int32))

    def update(self, state, history, alpha_reference, t, current_size_grid=None, current_available=None):
        time = jnp.asarray(t, jnp.int32)
        error = _clock_error(state, time)
        updated = replace(state, values=state.application, last_time=time)
        new = jax.lax.cond(error == 0, lambda _: updated, lambda _: state, operand=None)
        return WeightUpdate(new, state.application, jnp.zeros_like(state.values),
                            jnp.zeros_like(state.values), jnp.asarray(True), jnp.asarray(False), error)

    def weights(self, state):
        return state.values


def _efficiency(history, alpha_reference, curves, supplied_mask):
    h, m = history.issued_alpha.shape[1], history.grid.values.size
    curves = jnp.full((h, m), jnp.nan, history.grid.values.dtype) if curves is None else jnp.asarray(curves, history.grid.values.dtype)
    if curves.shape != (h, m):
        raise ValueError("current predictable size curves must have shape (H,M)")
    efficiency, available = marginal_set_efficiency(history.grid, curves, alpha_reference)
    if supplied_mask is not None:
        mask = jnp.asarray(supplied_mask, bool)
        if mask.shape != (h,):
            raise ValueError("size availability mask must have shape (H,)")
        available &= mask
    return jnp.where(available, efficiency, 0), jnp.all(available)


def _adapt(state, history, alpha_reference, t, curves, supplied_mask, settings, failure_aware):
    time = jnp.asarray(t, jnp.int32)
    alpha = jnp.asarray(alpha_reference, state.values.dtype)
    if alpha.shape != state.values.shape:
        raise ValueError("alpha_reference must contain H ordinary allocations")
    error = _clock_error(state, time)
    valid_reference = jnp.all(jnp.isfinite(alpha) & (alpha >= history.alpha_min) & (alpha <= history.alpha_max))
    efficiency, curve_ready = _efficiency(history, alpha, curves, supplied_mask)
    raw = state.application*alpha*efficiency
    failure = jnp.zeros_like(state.values)
    ready = curve_ready
    if failure_aware:  # Factory-selected algorithm, static at compilation.
        failure, cohort_ready = first_failure_histogram(history, time, settings.failure_window)
        ready &= cohort_ready
        raw = (raw + settings.eps_efficiency)/(settings.eps_failure + failure)
    # Normalize across horizons before projecting (writeup eq. 17). Without it the
    # target carries the data's units (e.g. W/m^2 for solar), and float64 rounding
    # in the projection then breaks sum(w)=1 beyond the validity tolerance.
    total = jnp.sum(raw)
    raw = jnp.where(total > 0, raw/jnp.where(total > 0, total, 1), raw)
    target = jnp.where(ready, bounded_simplex_projection(raw, settings.lower, settings.upper), state.application)
    proposed = smooth_and_cap(state.values, target, settings.smoothing, settings.max_step)
    tolerance = 64*jnp.finfo(state.values.dtype).eps*state.values.size
    valid = (valid_reference & jnp.all(jnp.isfinite(proposed)) &
             jnp.all(proposed >= settings.lower-tolerance) & jnp.all(proposed <= settings.upper+tolerance) &
             (jnp.abs(jnp.sum(proposed)-1) <= tolerance))
    error = jnp.where((error == 0) & ~valid, UpdateError.INVALID_INPUT, error).astype(jnp.int32)
    updated = replace(state, values=proposed, last_time=time)
    result = jax.lax.cond(error == 0, lambda _: updated, lambda _: state, operand=None)
    return WeightUpdate(result, target, efficiency, failure, ready, ~ready, error)


class MarginalEfficiencyWeightUpdater(StaticWeightUpdater):
    def update(self, state, history, alpha_reference, t, current_size_grid=None, current_available=None):
        return _adapt(state, history, alpha_reference, t, current_size_grid, current_available, self, False)


class FailureAwareEfficiencyWeightUpdater(MarginalEfficiencyWeightUpdater):
    def __init__(self, config: WeightConfig):
        super().__init__(config)
        if type(config.failure_window) is not int or config.failure_window < 1:
            raise ValueError("failure_window must be a positive integer")
        if any(not isfinite(x) or x <= 0 for x in (config.eps_failure, config.eps_efficiency)):
            raise ValueError("efficiency/failure regularizers must be finite and positive")
        self.failure_window = config.failure_window
        self.eps_failure, self.eps_efficiency = config.eps_failure, config.eps_efficiency

    def init_state(self, history):
        if history.origin_id.size < history.issued_alpha.shape[1] + self.failure_window:
            raise ValueError("history must retain H + failure window origins")
        return super().init_state(history)

    def update(self, state, history, alpha_reference, t, current_size_grid=None, current_available=None):
        return _adapt(state, history, alpha_reference, t, current_size_grid, current_available, self, True)


def build_weight_updater(config: WeightConfig) -> UtilityWeightUpdater:
    choices = {"static": StaticWeightUpdater, "marginal_efficiency": MarginalEfficiencyWeightUpdater,
               "failure_aware": FailureAwareEfficiencyWeightUpdater}
    if config.kind not in choices:
        raise ValueError(f"unknown utility weight updater: {config.kind!r}")
    return choices[config.kind](config)
