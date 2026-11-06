"""Synthetic scalar interval geometry for testing orchestration, not forecasting.

Y_t are IID Uniform[-1,1]. The ordinary horizon family is
C_h(a)=[-(1-a), 1-a], with width 2(1-a) and critical level 1-|Y_t|.
Distinct horizons of one origin see distinct independent future draws, hence
its ordinary conditional joint event probability is a*b. Shared physical draws
across origins exercise the delayed, dependent stream. No future draw is returned
through the adapter boundary. Replace this fixture with an application adapter.
"""

from copy import deepcopy
from typing import Optional

import jax
import jax.numpy as jnp

from .configs import ExperimentConfig, configure_jax
from .history import ActionMode, AlphaGrid
from .runner import ForecastInputs, HorizonObservationBatch


class SyntheticIntervalAdapter:
    def __init__(self, config: ExperimentConfig, outcomes=None, critical_available=None):
        config.validate()
        self.config = deepcopy(config)
        configure_jax(config.jax)
        self.dtype = jnp.float64 if config.jax.enable_x64 else jnp.float32
        self._pending = {}
        self._last_observe = -1
        self._last_prepare = -1
        self.issued_count = 0
        self._current_outcome = 0.0
        key = jax.random.key(config.runner.seed)
        self._draw = jax.jit(lambda t: jax.random.uniform(jax.random.fold_in(key, t), (),
                             dtype=self.dtype, minval=-1, maxval=1))
        self._outcomes = None
        if outcomes is not None:
            values = jnp.asarray(outcomes, self.dtype)
            if values.ndim != 1 or values.size < config.t_max+config.history.horizon:
                raise ValueError("outcomes must cover issuance and all H tail rounds")
            if not bool(jnp.all(jnp.isfinite(values) & (jnp.abs(values) <= 1))):
                raise ValueError("synthetic interval outcomes must be finite in [-1,1]")
            self._outcomes = values
        self._available = None
        if critical_available is not None:
            values = jnp.asarray(critical_available, bool)
            if values.shape != (config.t_max, config.history.horizon):
                raise ValueError("critical availability must have shape (t_max,H)")
            self._available = values.tolist()

    def observe(self, t: int) -> HorizonObservationBatch:
        if t != self._last_observe+1 or (t < self.config.t_max and self.issued_count != t):
            raise ValueError("synthetic observations must follow consecutive completed rounds")
        outcome = self._draw(t) if self._outcomes is None else self._outcomes[t]
        self._current_outcome = float(outcome)
        level = float(jnp.asarray(1, self.dtype)-jnp.abs(outcome))
        hmax = self.config.history.horizon
        origins, violations, levels, available, active = [], [], [], [], []
        for h in range(1, hmax+1):
            origin = t-h
            present = 0 <= origin < self.issued_count
            mode, alpha = self._pending[origin] if present else (None, None)
            fail = (level < alpha[h-1] if mode == ActionMode.ORDINARY else
                    mode == ActionMode.AGGRESSIVE_EXTREME and h == 1)
            supplied = present and (self._available is None or self._available[origin][h-1])
            origins.append(origin)
            violations.append(bool(fail))
            levels.append(level if supplied else float("nan"))
            available.append(supplied)
            active.append(present)
            if present and h == hmax:
                del self._pending[origin]
        self._last_observe = t
        return HorizonObservationBatch(jnp.asarray(origins, jnp.int32), jnp.arange(1, hmax+1),
            jnp.asarray(violations, bool), jnp.asarray(levels, self.dtype),
            jnp.asarray(available, bool), jnp.asarray(active, bool))

    def prepare(self, t: int, grid: AlphaGrid) -> ForecastInputs:
        if t != self._last_observe or t >= self.config.t_max or t == self._last_prepare:
            raise ValueError("prepare current geometry once after observing and before issuing")
        h, p = self.config.history.horizon, self.config.history.covariate_size
        curves = jnp.broadcast_to(2*(1-grid.values), (h, grid.values.size))
        covariates = jnp.zeros(p, self.dtype)
        if p:
            covariates = covariates.at[0].set(abs(self._current_outcome))
        self._last_prepare = t
        return ForecastInputs(curves, jnp.ones(h, bool), covariates)

    def issue(self, t: int, action_mode: ActionMode, alpha: Optional[jax.Array]) -> None:
        if t != self.issued_count or t != self._last_prepare or t >= self.config.t_max:
            raise ValueError("issue exactly once after preparing current geometry")
        action_mode = ActionMode(action_mode)
        if action_mode == ActionMode.ORDINARY:
            values = jnp.asarray(alpha, self.dtype) if alpha is not None else jnp.array([])
            if values.shape != (self.config.history.horizon,) or not bool(jnp.all(
                jnp.isfinite(values) & (values >= self.config.history.alpha_min) &
                (values <= self.config.history.alpha_max))):
                raise ValueError("ordinary issuance requires a finite bounded alpha vector")
            stored = values.tolist()
        else:
            if alpha is not None:
                raise ValueError("extreme issuance must receive alpha=None")
            stored = None
        self._pending[t] = (action_mode, stored)
        self.issued_count += 1
