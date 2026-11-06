"""Fixed-capacity numerical history with explicit, JIT-safe update results.

Origins and physical time start at zero; horizons and edge endpoints are 1-based.
All updates return an error code rather than throwing inside JIT. A rejected
update leaves the state unchanged. The Python runner must check these codes.
Extreme actions have NaN alpha storage and an explicit action mode; they never
borrow an ordinary allocation to describe the actual issued sets.
"""

from dataclasses import dataclass, replace
from enum import IntEnum

import jax
import jax.numpy as jnp


class ActionMode(IntEnum):
    ORDINARY = 0
    CONSERVATIVE_EXTREME = 1
    AGGRESSIVE_EXTREME = 2  # Horizon 1 is empty; later sets belong to the adapter.


class UpdateError(IntEnum):
    OK = 0
    UNKNOWN_ORIGIN = 1
    WRONG_REVEAL_TIME = 2
    DUPLICATE = 3
    INVALID_INPUT = 4
    UNSAFE_OVERWRITE = 5
    OUT_OF_ORDER = 6
    NOT_YET_RESOLVED = 7


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AlphaGrid:
    values: jax.Array

    @classmethod
    def create(cls, values) -> "AlphaGrid":
        """Validate an external grid once, outside compiled kernels."""
        values = jnp.asarray(values)
        if not jnp.issubdtype(values.dtype, jnp.floating):
            values = values.astype(jnp.asarray(0.0).dtype)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("alpha grid must be a vector with at least two entries")
        valid = (jnp.all(jnp.isfinite(values)) & jnp.all(values > 0) &
                 jnp.all(values < 1) & jnp.all(jnp.diff(values) > 0))
        if not bool(valid):
            raise ValueError("alpha grid requires 0 < a_1 < ... < a_M < 1")
        return cls(values)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class HistoryState:
    grid: AlphaGrid
    alpha_min: jax.Array
    alpha_max: jax.Array
    origin_id: jax.Array
    valid_origin: jax.Array
    action_mode: jax.Array
    issued_alpha: jax.Array
    horizon_resolved: jax.Array
    horizon_violation: jax.Array
    horizon_reveal_time: jax.Array
    critical_level: jax.Array
    critical_resolved: jax.Array  # Availability mask: an actual critical level is known.
    trajectory_label: jax.Array
    trajectory_resolved: jax.Array
    first_failure: jax.Array  # H encodes success ONLY jointly with label == 0.
    reveal_time: jax.Array
    set_size_grid: jax.Array  # Ordinary-family sizes, even on extreme rounds.
    set_size_available: jax.Array
    issue_covariates: jax.Array
    next_origin_id: jax.Array
    missing_critical_count: jax.Array  # Lifetime per-horizon counts, including evicted origins.

    @property
    def critical_missing(self) -> jax.Array:
        return self.horizon_resolved & ~self.critical_resolved


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class HistoryUpdate:
    state: HistoryState
    error: jax.Array
    newly_resolved: jax.Array

    @property
    def accepted(self) -> jax.Array:
        return self.error == UpdateError.OK


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CompletedOrigins:
    origin_id: jax.Array
    slots: jax.Array
    mask: jax.Array
    labels: jax.Array
    first_failure: jax.Array

    @property
    def ready(self) -> jax.Array:
        return jnp.all(self.mask)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EdgeCriticalLevels:
    levels: jax.Array  # (N,E,2); unavailable pairs are NaN.
    mask: jax.Array  # (N,E), both levels available by time t.
    missing_mask: jax.Array  # (N,E), due origin pairs that cannot be supplied.
    pending_mask: jax.Array  # (N,E), valid issued origin pairs not yet due.


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ResolvedLabels:
    """H fixed slots in origin order, containing only labels revealed at t."""

    origin_id: jax.Array
    labels: jax.Array
    mask: jax.Array


def init_history(grid: AlphaGrid, horizon: int, capacity: int,
                 alpha_min: float, alpha_max: float, covariate_size: int = 0) -> HistoryState:
    """Initialize static shapes at the Python boundary, after configuring JAX."""
    if horizon < 1 or capacity <= horizon or covariate_size < 0:
        raise ValueError("history requires H >= 1, capacity > H, covariate_size >= 0")
    if not 0 < alpha_min < alpha_max < 1:
        raise ValueError("ordinary alpha bounds require 0 < alpha_min < alpha_max < 1")
    dtype = grid.values.dtype
    lower, upper = jnp.asarray(alpha_min, dtype), jnp.asarray(alpha_max, dtype)
    if not bool((grid.values[0] <= lower) & (grid.values[-1] >= upper)):
        raise ValueError("alpha grid must cover the ordinary alpha box")
    if not bool((lower > 0) & (lower < upper) & (upper < 1)):
        raise ValueError("ordinary alpha box is not representable at the configured precision")
    ch = (capacity, horizon)
    return HistoryState(
        grid=grid, alpha_min=jnp.asarray(alpha_min, dtype), alpha_max=jnp.asarray(alpha_max, dtype),
        origin_id=jnp.full(capacity, -1, jnp.int32), valid_origin=jnp.zeros(capacity, bool),
        action_mode=jnp.full(capacity, -1, jnp.int32), issued_alpha=jnp.full(ch, jnp.nan, dtype),
        horizon_resolved=jnp.zeros(ch, bool), horizon_violation=jnp.zeros(ch, bool),
        horizon_reveal_time=jnp.full(ch, -1, jnp.int32), critical_level=jnp.full(ch, jnp.nan, dtype),
        critical_resolved=jnp.zeros(ch, bool), trajectory_label=jnp.full(capacity, -1, jnp.int32),
        trajectory_resolved=jnp.zeros(capacity, bool), first_failure=jnp.full(capacity, -1, jnp.int32),
        reveal_time=jnp.full(capacity, -1, jnp.int32),
        set_size_grid=jnp.full((capacity, horizon, grid.values.size), jnp.nan, dtype),
        set_size_available=jnp.zeros(ch, bool),
        issue_covariates=jnp.zeros((capacity, covariate_size), dtype),
        next_origin_id=jnp.asarray(0, jnp.int32), missing_critical_count=jnp.zeros(horizon, jnp.int32),
    )


def _slots(state: HistoryState, origin_id):
    origins = jnp.asarray(origin_id, jnp.int32)
    slots = jnp.mod(origins, state.origin_id.size)
    present = (origins >= 0) & state.valid_origin[slots] & (state.origin_id[slots] == origins)
    return slots, present


def add_origin(state: HistoryState, origin_id, action_mode, alpha=None,
               set_size_grid=None, set_size_available=None, covariates=None) -> HistoryUpdate:
    """Record an issuance; overwrite only after all actual horizons are processed.

    alpha must be supplied for ordinary issuance and omitted for extremes.
    Sizes describe the ordinary family, not the formal extreme sets. Missing
    curves occupy fixed-shape NaN rows with an availability mask.
    """
    h = state.issued_alpha.shape[1]
    dtype = state.grid.values.dtype
    origin = jnp.asarray(origin_id, jnp.int32)
    mode = jnp.asarray(action_mode, jnp.int32)
    slot = jnp.mod(origin, state.origin_id.size)
    ordinary = mode == ActionMode.ORDINARY
    alpha_values = jnp.full(h, jnp.nan, dtype) if alpha is None else jnp.asarray(alpha, dtype)
    sizes = (jnp.full((h, state.grid.values.size), jnp.nan, dtype) if set_size_grid is None
             else jnp.asarray(set_size_grid, dtype))
    size_mask = (jnp.all(jnp.isfinite(sizes), axis=-1) if set_size_available is None
                 else jnp.asarray(set_size_available, bool))
    covariates = (jnp.zeros(state.issue_covariates.shape[1], dtype) if covariates is None
                  else jnp.asarray(covariates, dtype))
    if (alpha_values.shape != (h,) or sizes.shape != (h, state.grid.values.size) or
            size_mask.shape != (h,) or covariates.shape != state.issue_covariates.shape[1:]):
        raise ValueError("issuance array shapes must match the static history shapes")
    valid_alpha = jnp.all(jnp.isfinite(alpha_values) & (alpha_values >= state.alpha_min) &
                          (alpha_values <= state.alpha_max))
    valid_sizes = jnp.all(jnp.where(size_mask,
        jnp.all(jnp.isfinite(sizes) & (sizes >= 0), axis=-1) &
        jnp.all(jnp.diff(sizes, axis=-1) <= 0, axis=-1), True))
    reusable = (~state.valid_origin[slot] | (state.trajectory_resolved[slot] &
                                             jnp.all(state.horizon_resolved[slot])))
    valid = ((mode >= ActionMode.ORDINARY) & (mode <= ActionMode.AGGRESSIVE_EXTREME) &
             jnp.where(ordinary, valid_alpha, jnp.all(jnp.isnan(alpha_values))) &
             valid_sizes & jnp.all(jnp.isfinite(covariates)))
    error = jnp.where(origin != state.next_origin_id, UpdateError.OUT_OF_ORDER,
            jnp.where(~reusable, UpdateError.UNSAFE_OVERWRITE,
            jnp.where(~valid, UpdateError.INVALID_INPUT, UpdateError.OK))).astype(jnp.int32)

    def write(s):
        return replace(s,
            origin_id=s.origin_id.at[slot].set(origin), valid_origin=s.valid_origin.at[slot].set(True),
            action_mode=s.action_mode.at[slot].set(mode),
            issued_alpha=s.issued_alpha.at[slot].set(jnp.where(ordinary, alpha_values, jnp.nan)),
            horizon_resolved=s.horizon_resolved.at[slot].set(False),
            horizon_violation=s.horizon_violation.at[slot].set(False),
            horizon_reveal_time=s.horizon_reveal_time.at[slot].set(-1),
            critical_level=s.critical_level.at[slot].set(jnp.nan),
            critical_resolved=s.critical_resolved.at[slot].set(False),
            trajectory_label=s.trajectory_label.at[slot].set(-1),
            trajectory_resolved=s.trajectory_resolved.at[slot].set(False),
            first_failure=s.first_failure.at[slot].set(-1), reveal_time=s.reveal_time.at[slot].set(-1),
            set_size_grid=s.set_size_grid.at[slot].set(jnp.where(size_mask[:, None], sizes, jnp.nan)),
            set_size_available=s.set_size_available.at[slot].set(size_mask),
            issue_covariates=s.issue_covariates.at[slot].set(covariates), next_origin_id=origin + 1)

    return HistoryUpdate(jax.lax.cond(error == 0, write, lambda s: s, state), error, jnp.asarray(False))


def resolve_horizon(state: HistoryState, origin_id, horizon, t, actual_violation,
                    critical_level=jnp.nan, critical_available=False) -> HistoryUpdate:
    """Process actual membership and an optional ordinary-family critical level.

    Process horizons in temporal order at exactly k+h. A missing level is
    explicit, not a zero critical value. At an ordinary action, supplied
    critical levels must agree with the strict actual-membership convention.
    """
    slot, present = _slots(state, origin_id)
    h = jnp.asarray(horizon, jnp.int32)
    index = jnp.clip(h - 1, 0, state.issued_alpha.shape[1] - 1)
    time = jnp.asarray(t, jnp.int32)
    violation = jnp.asarray(actual_violation, bool)
    available = jnp.asarray(critical_available, bool)
    level = jnp.asarray(critical_level, state.grid.values.dtype)
    mode = state.action_mode[slot]
    # +/- infinity represent observations outside every/inside every ordinary set.
    valid_level = ~available | ~jnp.isnan(level)
    ordinary_match = (~available | (violation == (level < state.issued_alpha[slot, index])))
    actual_match = jnp.where(mode == ActionMode.ORDINARY, ordinary_match,
                   jnp.where(mode == ActionMode.CONSERVATIVE_EXTREME, ~violation,
                             (h != 1) | violation))
    previous_done = jnp.all(state.horizon_resolved[slot] |
                           (jnp.arange(state.issued_alpha.shape[1]) >= index))
    valid = (h >= 1) & (h <= state.issued_alpha.shape[1]) & valid_level & actual_match
    error = jnp.where(~present, UpdateError.UNKNOWN_ORIGIN,
            jnp.where(~valid, UpdateError.INVALID_INPUT,
            jnp.where(time != jnp.asarray(origin_id) + h, UpdateError.WRONG_REVEAL_TIME,
            jnp.where(state.horizon_resolved[slot, index], UpdateError.DUPLICATE,
            jnp.where(~previous_done, UpdateError.OUT_OF_ORDER, UpdateError.OK))))).astype(jnp.int32)

    def write(s):
        return replace(s,
            horizon_resolved=s.horizon_resolved.at[slot, index].set(True),
            horizon_violation=s.horizon_violation.at[slot, index].set(violation),
            horizon_reveal_time=s.horizon_reveal_time.at[slot, index].set(time),
            critical_level=s.critical_level.at[slot, index].set(jnp.where(available, level, jnp.nan)),
            critical_resolved=s.critical_resolved.at[slot, index].set(available),
            missing_critical_count=s.missing_critical_count.at[index].add((~available).astype(jnp.int32)))

    return HistoryUpdate(jax.lax.cond(error == 0, write, lambda s: s, state), error, jnp.asarray(False))


def resolve_trajectory(state: HistoryState, origin_id, t) -> HistoryUpdate:
    """Resolve from ACTUAL observed horizons, independently of counterfactual data."""
    slot, present = _slots(state, origin_id)
    observed = state.horizon_resolved[slot] & (state.horizon_reveal_time[slot] <= t)
    failed = observed & state.horizon_violation[slot]
    any_failed = jnp.any(failed)
    all_observed = jnp.all(observed)
    horizon = state.issued_alpha.shape[1]
    first = jnp.min(jnp.where(failed, jnp.arange(1, horizon + 1, dtype=jnp.int32), horizon))
    origin = jnp.asarray(origin_id, jnp.int32)
    reveal = jnp.where(any_failed, origin + first, origin + horizon)
    error = jnp.where(~present, UpdateError.UNKNOWN_ORIGIN,
            jnp.where(state.trajectory_resolved[slot], UpdateError.DUPLICATE,
            jnp.where(~(any_failed | all_observed), UpdateError.NOT_YET_RESOLVED, UpdateError.OK)))

    def write(s):
        return replace(s,
            trajectory_label=s.trajectory_label.at[slot].set(any_failed.astype(jnp.int32)),
            trajectory_resolved=s.trajectory_resolved.at[slot].set(True),
            first_failure=s.first_failure.at[slot].set(first), reveal_time=s.reveal_time.at[slot].set(reveal))

    return HistoryUpdate(jax.lax.cond(error == 0, write, lambda s: s, state),
                         error.astype(jnp.int32), error == UpdateError.OK)


def recent_completed_origins(state: HistoryState, t, window: int) -> CompletedOrigins:
    """Fixed origin cohort [t-H-W+1, t-H], never a reveal-order window.

    window is a static argument when this query is JIT compiled.
    """
    if window < 1:
        raise ValueError("window must be positive")
    end = jnp.asarray(t, jnp.int32) - state.issued_alpha.shape[1]
    origins = end - window + 1 + jnp.arange(window, dtype=jnp.int32)
    slots, present = _slots(state, origins)
    mask = present & state.trajectory_resolved[slots] & (state.reveal_time[slots] <= t)
    return CompletedOrigins(origins, slots, mask,
                            jnp.where(mask, state.trajectory_label[slots], -1),
                            jnp.where(mask, state.first_failure[slots], -1))


def resolved_labels_at(state: HistoryState, t) -> ResolvedLabels:
    """Actual trajectory labels newly revealed at physical time t, at most H."""
    time = jnp.asarray(t, jnp.int32)
    origins = time - state.issued_alpha.shape[1] + jnp.arange(state.issued_alpha.shape[1], dtype=jnp.int32)
    slots, present = _slots(state, origins)
    mask = present & state.trajectory_resolved[slots] & (state.reveal_time[slots] == time)
    return ResolvedLabels(origins, jnp.where(mask, state.trajectory_label[slots], -1), mask)


def edge_critical_levels(state: HistoryState, origin_ids, edges, t) -> EdgeCriticalLevels:
    """Query ordinary-family pairs with origin/reveal/availability masks.

    edges has static shape (E,2) and uses 1-based horizon numbers. Missing due
    pairs retain their position; a window must never backfill from older origins.
    """
    origins = jnp.asarray(origin_ids, jnp.int32)
    edges = jnp.asarray(edges, jnp.int32)
    if origins.ndim != 1 or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("expected origin vector and edge array of shape (E,2)")
    slots, present = _slots(state, origins)
    valid_edge = jnp.all((edges >= 1) & (edges <= state.issued_alpha.shape[1]), axis=-1)
    valid_edge &= edges[:, 0] < edges[:, 1]
    indices = jnp.clip(edges - 1, 0, state.issued_alpha.shape[1] - 1)
    levels = state.critical_level[slots[:, None, None], indices[None, :, :]]
    available = state.critical_resolved[slots[:, None, None], indices[None, :, :]]
    reveal = state.horizon_reveal_time[slots[:, None, None], indices[None, :, :]]
    due = (origins[:, None] >= 0) & (origins[:, None] + jnp.max(edges, axis=-1) <= t)
    due &= valid_edge[None, :]
    mask = present[:, None] & due & jnp.all(available & (reveal <= t), axis=-1)
    pending = present[:, None] & valid_edge[None, :] & (origins[:, None] <= t) & ~due
    return EdgeCriticalLevels(jnp.where(mask[..., None], levels, jnp.nan), mask, due & ~mask, pending)


def lower_orthant_indicators(levels: jax.Array, grid: AlphaGrid) -> jax.Array:
    """Strict lower-orthant events; supports arbitrary leading pair dimensions."""
    left = levels[..., 0, None] < grid.values
    right = levels[..., 1, None] < grid.values
    return left[..., :, None] & right[..., None, :]


def first_failure_histogram(state: HistoryState, t, window: int):
    cohort = recent_completed_origins(state, t, window)
    horizons = jnp.arange(1, state.issued_alpha.shape[1] + 1)
    # label == 1 prevents successful H sentinels being counted as horizon-H failures.
    events = (cohort.mask & (cohort.labels == 1))[:, None] & (cohort.first_failure[:, None] == horizons)
    counts = jnp.sum(events, axis=0)
    return counts / window, cohort.ready


def set_size_curve(state: HistoryState, origin_id, t):
    slot, present = _slots(state, origin_id)
    mask = present & (jnp.asarray(origin_id) <= t) & state.set_size_available[slot]
    return jnp.where(mask[:, None], state.set_size_grid[slot], jnp.nan), mask


def marginal_set_efficiency(grid: AlphaGrid, curves, alpha_reference):
    """Centered grid slopes (one-sided endpoints), interpolated at reference alpha.

    Receives CURRENT predictable curves from the adapter; it does not average
    historical curves. Returns (nonnegative efficiency, per-horizon availability).
    """
    curves = jnp.asarray(curves, grid.values.dtype)
    alpha = jnp.asarray(alpha_reference, grid.values.dtype)
    values = grid.values
    mid = -(curves[:, 2:] - curves[:, :-2]) / (values[2:] - values[:-2])
    first = -(curves[:, 1] - curves[:, 0]) / (values[1] - values[0])
    last = -(curves[:, -1] - curves[:, -2]) / (values[-1] - values[-2])
    slopes = jnp.concatenate((first[:, None], mid, last[:, None]), axis=-1)
    available = (jnp.all(jnp.isfinite(curves) & (curves >= 0), axis=-1) &
                 jnp.all(jnp.diff(curves, axis=-1) <= 0, axis=-1) &
                 jnp.isfinite(alpha) & (alpha >= values[0]) & (alpha <= values[-1]))
    efficiency = jax.vmap(lambda slope, a: jnp.interp(a, values, slope))(slopes, alpha)
    return jnp.where(available, jnp.maximum(efficiency, 0), 0), available
