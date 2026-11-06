"""Predict-and-correct trajectory feedback, independent of budget control.

Select a concrete signal before compilation. Bound methods invoke pure JAX
kernels with fixed Python callables; there is no runtime ABC/string dispatch.
The host calls observe after resolving history, then issue after recording the
new origin. Atomic packets retain origin identities; controllers will consume
one aggregate per round, with issuance queued explicitly by RunnerState.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import IntEnum
from math import isfinite

import jax
import jax.numpy as jnp

from .configs import SignalConfig
from .history import ActionMode, HistoryState, UpdateError, recent_completed_origins, resolved_labels_at


class SignalEvent(IntEnum):
    NONE = 0
    ISSUE = 1
    TERMINAL = 2
    REFRESH = 3


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SignalPacket:
    origin_id: jax.Array  # (H+1,); final slot reserved for issuance.
    increments: jax.Array
    event_mask: jax.Array
    event_kind: jax.Array
    resolved_origin_id: jax.Array  # (H,), diagnostics only; not the q input.
    resolved_labels: jax.Array
    resolved_mask: jax.Array

    @property
    def total(self):
        return jnp.sum(jnp.where(self.event_mask, self.increments, 0))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SignalState:
    beta: jax.Array
    prior: jax.Array
    origin_id: jax.Array
    pending: jax.Array
    provisional: jax.Array
    features: jax.Array  # Frozen issue-time [1, ordinary alpha, covariates].
    theta: jax.Array
    trained_examples: jax.Array
    cumulative_increment: jax.Array
    last_observe_time: jax.Array
    next_issue_origin: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SignalUpdate:
    state: SignalState
    packet: SignalPacket
    error: jax.Array

    @property
    def accepted(self):
        return self.error == UpdateError.OK


class TrajectorySignal(ABC):
    @abstractmethod
    def init_state(self, history: HistoryState) -> SignalState: ...

    @abstractmethod
    def observe(self, state: SignalState, history: HistoryState, t) -> SignalUpdate: ...

    @abstractmethod
    def issue(self, state: SignalState, history: HistoryState, t) -> SignalUpdate: ...


def empty_packet(history: HistoryState) -> SignalPacket:
    h = history.issued_alpha.shape[1]
    return SignalPacket(jnp.full(h + 1, -1, jnp.int32),
                        jnp.zeros(h + 1, history.grid.values.dtype), jnp.zeros(h + 1, bool),
                        jnp.zeros(h + 1, jnp.int32), jnp.full(h, -1, jnp.int32),
                        jnp.full(h, -1, jnp.int32), jnp.zeros(h, bool))


def _init_state(history, beta, prior, learned=False):
    dtype = history.grid.values.dtype
    capacity, horizon = history.issued_alpha.shape
    features = 1 + horizon + history.issue_covariates.shape[1]
    theta = jnp.zeros(features, dtype)
    if learned:  # Construction time, before compilation.
        initial_probability = jnp.asarray(prior, dtype)
        theta = theta.at[0].set(jnp.log(initial_probability) - jnp.log1p(-initial_probability))
    return SignalState(jnp.asarray(beta, dtype), jnp.asarray(prior, dtype),
                       jnp.full(capacity, -1, jnp.int32), jnp.zeros(capacity, bool),
                       jnp.zeros(capacity, dtype), jnp.zeros((capacity, features), dtype), theta,
                       jnp.asarray(0, jnp.int32), jnp.asarray(0, dtype),
                       jnp.asarray(-1, jnp.int32), jnp.asarray(0, jnp.int32))


def _validate_beta(beta):
    if not isfinite(beta) or not 0 < beta < 1:
        raise ValueError("beta must be in (0, 1)")


def _unchanged_model(theta, features, labels, mask):
    return theta, jnp.asarray(0, jnp.int32)


def _observe(state, history, t, predict, train, refresh):
    time = jnp.asarray(t, jnp.int32)
    batch = resolved_labels_at(history, time)
    slots = jnp.mod(batch.origin_id, history.origin_id.size)
    present = ((batch.origin_id >= 0) & history.valid_origin[slots] &
               (history.origin_id[slots] == batch.origin_id))
    registered = state.origin_id[slots] == batch.origin_id
    active = present & registered & state.pending[slots]
    terminal = active & batch.mask
    overwritten = state.pending & (~history.valid_origin | (state.origin_id != history.origin_id))
    late = (state.pending & history.trajectory_resolved &
            (state.origin_id == history.origin_id) & (history.reveal_time < time))
    unregistered = history.valid_origin & (history.origin_id < time) & (history.origin_id >= state.next_issue_origin)
    expired = state.pending & (state.origin_id + history.issued_alpha.shape[1] <= time) & ~(
        history.trajectory_resolved & (history.reveal_time <= time))
    error = jnp.where(time == state.last_observe_time, UpdateError.DUPLICATE,
            jnp.where((time != state.last_observe_time + 1) | jnp.any(late | unregistered), UpdateError.OUT_OF_ORDER,
            jnp.where(jnp.any(overwritten), UpdateError.UNSAFE_OVERWRITE,
            jnp.where(jnp.any(expired), UpdateError.INVALID_INPUT, UpdateError.OK)))).astype(jnp.int32)

    def apply(_):
        # Terminal corrections use the LAST BOOKED probability, not a new fit.
        old = state.provisional[slots]
        increments = jnp.where(terminal, old - batch.labels, 0)
        ordinary = history.action_mode[slots] == ActionMode.ORDINARY
        training_mask = terminal & ordinary
        theta, trained = train(state.theta, state.features[slots], batch.labels, training_mask)
        remaining = active & ~terminal
        refresh_mask = remaining & ordinary & jnp.asarray(refresh)
        predicted = predict(replace(state, theta=theta), history, time, state.features[slots])
        refreshed = jnp.where(refresh_mask, predicted, old)
        increments += jnp.where(refresh_mask, old - refreshed, 0)
        new = jnp.where(terminal, batch.labels, refreshed)
        pending = state.pending.at[slots].set(jnp.where(active, ~terminal, state.pending[slots]))
        provisional = state.provisional.at[slots].set(jnp.where(active, new, old))
        event_mask = (terminal | refresh_mask) & (increments != 0)
        packet = SignalPacket(jnp.concatenate((batch.origin_id, jnp.asarray([-1], jnp.int32))),
            jnp.concatenate((increments, jnp.zeros(1, state.beta.dtype))),
            jnp.concatenate((event_mask, jnp.zeros(1, bool))),
            jnp.concatenate((jnp.where(event_mask, jnp.where(terminal, SignalEvent.TERMINAL,
                                                         SignalEvent.REFRESH), SignalEvent.NONE).astype(jnp.int32),
                             jnp.zeros(1, jnp.int32))),
            batch.origin_id, jnp.where(terminal, batch.labels, -1), terminal)
        updated = replace(state, pending=pending, provisional=provisional, theta=theta,
                          trained_examples=state.trained_examples + trained,
                          cumulative_increment=state.cumulative_increment + packet.total,
                          last_observe_time=time)
        finite_model = jnp.all(jnp.isfinite(theta))
        valid_old = jnp.all(~active | (jnp.isfinite(old) & (old >= 0) & (old <= 1)))
        valid_prediction = jnp.all(~refresh_mask | (jnp.isfinite(predicted) & (predicted >= 0) & (predicted <= 1)))
        return jax.lax.cond(finite_model & valid_old & valid_prediction,
            lambda _: SignalUpdate(updated, packet, error),
            lambda _: SignalUpdate(state, empty_packet(history), jnp.asarray(UpdateError.INVALID_INPUT, jnp.int32)),
            operand=None)

    return jax.lax.cond(error == UpdateError.OK, apply,
                        lambda _: SignalUpdate(state, empty_packet(history), error), operand=None)


def _issue(state, history, t, predict):
    time = jnp.asarray(t, jnp.int32)
    slot = jnp.mod(time, history.origin_id.size)
    present = history.valid_origin[slot] & (history.origin_id[slot] == time)
    ordinary = history.action_mode[slot] == ActionMode.ORDINARY
    # Extreme features are all zero and are excluded from fitting and refresh.
    features = jnp.concatenate((jnp.ones(1, state.beta.dtype),
                               jnp.where(ordinary, history.issued_alpha[slot], 0),
                               history.issue_covariates[slot]))
    features = jnp.where(ordinary, features, 0)
    error = jnp.where(~present, UpdateError.UNKNOWN_ORIGIN,
            jnp.where(time < state.next_issue_origin, UpdateError.DUPLICATE,
            jnp.where((time != state.next_issue_origin) | (time != state.last_observe_time), UpdateError.OUT_OF_ORDER,
            jnp.where(state.pending[slot], UpdateError.UNSAFE_OVERWRITE,
            jnp.where(jnp.any(history.horizon_resolved[slot]), UpdateError.INVALID_INPUT, UpdateError.OK)))))

    def apply(_):
        learned = predict(state, history, time, features[None, :])[0]
        deterministic = (history.action_mode[slot] == ActionMode.AGGRESSIVE_EXTREME).astype(state.beta.dtype)
        provisional = jnp.where(ordinary, learned, deterministic)
        increment = state.beta - provisional
        packet = empty_packet(history)
        packet = replace(packet, origin_id=packet.origin_id.at[-1].set(time),
                         increments=packet.increments.at[-1].set(increment),
                         event_mask=packet.event_mask.at[-1].set(increment != 0),
                         event_kind=packet.event_kind.at[-1].set(jnp.where(increment != 0, SignalEvent.ISSUE,
                                                                                     SignalEvent.NONE).astype(jnp.int32)))
        updated = replace(state, origin_id=state.origin_id.at[slot].set(time),
                          pending=state.pending.at[slot].set(True),
                          provisional=state.provisional.at[slot].set(provisional),
                          features=state.features.at[slot].set(features),
                          next_issue_origin=time + 1,
                          cumulative_increment=state.cumulative_increment + increment)
        valid_prediction = ~ordinary | (jnp.isfinite(learned) & (learned >= 0) & (learned <= 1))
        return jax.lax.cond(valid_prediction,
            lambda _: SignalUpdate(updated, packet, error.astype(jnp.int32)),
            lambda _: SignalUpdate(state, empty_packet(history), jnp.asarray(UpdateError.INVALID_INPUT, jnp.int32)),
            operand=None)

    return jax.lax.cond(error == UpdateError.OK, apply,
                        lambda _: SignalUpdate(state, empty_packet(history), error.astype(jnp.int32)), operand=None)


def _reveal_probability(state, history, t, features):
    return jnp.full(features.shape[0], state.beta, state.beta.dtype)


def _pending_valid_probability(state, history, t, features):
    return jnp.zeros(features.shape[0], state.beta.dtype)


class RevealOnlySignal(TrajectorySignal):
    def __init__(self, beta: float):
        _validate_beta(beta)
        self.beta = beta

    def init_state(self, history):
        return _init_state(history, self.beta, self.beta)

    def observe(self, state, history, t):
        return _observe(state, history, t, _reveal_probability, _unchanged_model, False)

    def issue(self, state, history, t):
        return _issue(state, history, t, _reveal_probability)


class PendingValidSignal(RevealOnlySignal):
    def observe(self, state, history, t):
        return _observe(state, history, t, _pending_valid_probability, _unchanged_model, False)

    def issue(self, state, history, t):
        return _issue(state, history, t, _pending_valid_probability)


class CompletedWindowProbabilitySignal(RevealOnlySignal):
    def __init__(self, beta: float, window: int = 100, refresh_pending: bool = True,
                 prior_probability: float | None = None):
        super().__init__(beta)
        prior = beta if prior_probability is None else prior_probability
        if type(window) is not int or window <= 0 or not isfinite(prior) or not 0 <= prior <= 1:
            raise ValueError("completed-window signal requires a positive window and a prior in [0, 1]")
        self.window = window
        self.refresh_pending = refresh_pending
        self.prior = prior

    def init_state(self, history):
        if history.origin_id.size < history.issued_alpha.shape[1] + self.window:
            raise ValueError("history capacity must retain H + signal window origins")
        return _init_state(history, self.beta, self.prior)

    def _predict(self, state, history, t, features):
        cohort = recent_completed_origins(history, t, self.window)
        mean = jnp.sum(jnp.where(cohort.mask, cohort.labels, 0)) / self.window
        probability = jnp.where(cohort.ready, mean, state.prior)
        return jnp.full(features.shape[0], probability, state.beta.dtype)

    def observe(self, state, history, t):
        return _observe(state, history, t, self._predict, _unchanged_model, self.refresh_pending)

    def issue(self, state, history, t):
        return _issue(state, history, t, self._predict)


def logistic_probability(theta, features):
    return jax.nn.sigmoid(features @ theta)


def _logistic_loss(theta, features, labels, mask):
    logits = features @ theta
    losses = jnp.logaddexp(0, logits) - jnp.where(mask, labels, 0) * logits
    return jnp.sum(jnp.where(mask, losses, 0)) / jnp.maximum(jnp.sum(mask), 1)


def train_logistic(theta, features, labels, mask, learning_rate):
    """One mean-batch SGD step per round with newly resolved ordinary labels."""
    count = jnp.sum(mask, dtype=jnp.int32)
    updated = jax.lax.cond(count > 0,
        lambda value: value - learning_rate * jax.grad(_logistic_loss)(value, features, labels, mask),
        lambda value: value, theta)
    return updated, count


class LearnedProbabilitySignal(RevealOnlySignal):
    def __init__(self, beta: float, learning_rate: float = 0.01, refresh_pending: bool = True,
                 prior_probability: float | None = None):
        super().__init__(beta)
        prior = beta if prior_probability is None else prior_probability
        if not isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("logistic learning_rate must be finite and positive")
        if not isfinite(prior) or not 0 < prior < 1:
            raise ValueError("a finite logistic initialization requires prior_probability in (0, 1)")
        self.learning_rate = learning_rate
        self.refresh_pending = refresh_pending
        self.prior = prior

    def init_state(self, history):
        return _init_state(history, self.beta, self.prior, learned=True)

    def _predict(self, state, history, t, features):
        return logistic_probability(state.theta, features)

    def _train(self, theta, features, labels, mask):
        return train_logistic(theta, features, labels, mask, self.learning_rate)

    def observe(self, state, history, t):
        return _observe(state, history, t, self._predict, self._train, self.refresh_pending)

    def issue(self, state, history, t):
        return _issue(state, history, t, self._predict)


def build_signal(config: SignalConfig, beta: float) -> TrajectorySignal:
    """Select the concrete implementation once at the Python boundary."""
    if config.kind == "reveal_only":
        return RevealOnlySignal(beta)
    if config.kind == "pending_valid":
        return PendingValidSignal(beta)
    if config.kind == "completed_window":
        return CompletedWindowProbabilitySignal(beta, config.window, config.refresh_pending,
                                                config.prior_probability)
    if config.kind == "learned":
        return LearnedProbabilitySignal(beta, config.learning_rate, config.refresh_pending,
                                        config.prior_probability)
    raise ValueError(f"unknown trajectory signal: {config.kind!r}")
