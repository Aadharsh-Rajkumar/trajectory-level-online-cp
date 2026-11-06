"""Real-data 1-D adapter: forecaster + per-horizon split-conformal intervals.

Pipeline (mirrors the D1/D2 split of the earlier ALPaCA/DtACI work):
  * D1 trains a forecaster offline (AR ridge, or ALPaCA from alpaca.py); its
    parameters are then frozen. A forecaster returns a mean and a scale per
    horizon; AR's scale is 1, ALPaCA's is its predictive standard deviation.
  * A held-out calibration block supplies initial standardized residual scores
    |y - mean| / scale per horizon, so intervals exist from the first round.
  * Online, horizon h of origin k is mean_{k,h} +/- scale_{k,h} Q_h(alpha_{k,h}),
    where Q_h is the conformal quantile of the last ``window`` revealed h-step
    scores at issue time (no lookahead). Revealed scores are appended afterwards.

The ordinary family is nested in alpha. Its critical level is the conformal
p-value (1 + #{S >= s}) / (n + 1); the interval contains y iff p >= alpha, which
matches the library's convention (failure iff critical < alpha). The half-width
consistent with this rule is the m-th largest score, m = ceil(alpha (n+1) - 1).
"""

from collections import deque
from copy import deepcopy
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .configs import ExperimentConfig, configure_jax
from .history import ActionMode, AlphaGrid
from .runner import ForecastInputs, HorizonObservationBatch


class ARForecaster:
    """Ridge AR(p) model with iterated multi-step forecasts."""

    def __init__(self, order: int = 5, ridge: float = 1e-3):
        self.order, self.ridge = order, ridge

    def fit(self, y: np.ndarray) -> "ARForecaster":
        p = self.order
        X = np.stack([y[i:len(y)-p+i] for i in range(p)], axis=1)
        X = np.hstack([X, np.ones((len(X), 1))])
        target = y[p:]
        self.coef = np.linalg.solve(X.T@X + self.ridge*np.eye(p+1), X.T@target)
        return self

    def predict_all(self, y: np.ndarray, origins: np.ndarray, horizon: int):
        """Mean forecasts from y[:k+1] for each origin k; unit scale."""
        mean = np.array([self.forecast(y[:k+1], horizon) for k in origins])
        return mean, np.ones_like(mean)

    def forecast(self, past: np.ndarray, horizon: int) -> np.ndarray:
        window = list(past[-self.order:])
        out = []
        for _ in range(horizon):
            nxt = float(np.dot(self.coef[:-1], window) + self.coef[-1])
            out.append(nxt)
            window = window[1:] + [nxt]
        return np.asarray(out)


def compute_forecasts(forecaster, series: np.ndarray, n_train: int, window: int,
                      horizon: int, t_max: int):
    """Fit on D1 and forecast every calibration/online origin once.

    Returns (mean, scale) indexed by origin - n_train; reusable across methods.
    """
    y = np.asarray(series, float)
    forecaster.fit(y[:n_train])
    origins = np.arange(n_train, n_train + window + horizon + t_max)
    return forecaster.predict_all(y, origins, horizon)


def _half_width(sorted_desc: np.ndarray, alpha) -> np.ndarray:
    n = sorted_desc.size
    m = np.ceil(np.asarray(alpha)*(n+1) - 1 - 1e-12).astype(int)
    return np.where(m <= 0, np.inf, sorted_desc[np.clip(m, 1, n)-1])


class ConformalIntervalAdapter:
    """TrajectoryAdapter for a scalar series; see module docstring."""

    def __init__(self, config: ExperimentConfig, series: np.ndarray, n_train: int,
                 window: int = 250, forecaster=None, forecasts=None):
        config.validate()
        self.config = deepcopy(config)
        configure_jax(config.jax)
        self.dtype = jnp.float64 if config.jax.enable_x64 else jnp.float32
        H = config.history.horizon
        if config.history.alpha_min <= 1/(window+1):
            raise ValueError("alpha_min must exceed 1/(window+1) for finite intervals")
        self.y = np.asarray(series, float)
        self.base = n_train + window + H          # series index of physical time 0
        if self.y.size < self.base + config.t_max + H:
            raise ValueError("series too short for train/calibration/online/tail")
        if forecasts is None:
            forecasts = compute_forecasts(forecaster or ARForecaster(), self.y, n_train,
                                          window, H, config.t_max)
        self.mean, self.scale = (np.asarray(a, float) for a in forecasts)
        self.n_train = n_train
        if self.mean.shape != (window + H + config.t_max, H) or self.scale.shape != self.mean.shape:
            raise ValueError("forecasts must cover every calibration and online origin")
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale) & (self.scale > 0)):
            raise ValueError("forecast means must be finite and scales positive")
        self.scores = [deque(maxlen=window) for _ in range(H)]
        for k in range(n_train, self.base - H):   # calibration origins, all revealed before t=0
            yhat, sc = self.mean[k-n_train], self.scale[k-n_train]
            for h in range(H):
                self.scores[h].append(abs(self.y[k+1+h] - yhat[h]) / sc[h])
        self._pending, self.records = {}, {}
        self._last_observe = self._last_prepare = -1
        self.issued_count = 0

    def _snapshot(self, t):
        i = self.base + t - self.n_train
        return (self.mean[i], self.scale[i],
                [np.sort(np.fromiter(s, float))[::-1] for s in self.scores])

    def observe(self, t: int) -> HorizonObservationBatch:
        if t != self._last_observe+1 or (t < self.config.t_max and self.issued_count != t):
            raise ValueError("observations must follow consecutive completed rounds")
        H = self.config.history.horizon
        y_now = self.y[self.base + t]
        origins, viol, levels, avail, active = [], [], [], [], []
        new_scores = []
        for h in range(1, H+1):
            k = t-h
            present = 0 <= k < self.issued_count
            fail, level = False, float("nan")
            if present:
                mode, alpha, yhat, scale, sdesc = self._pending[k]
                s = abs(y_now - yhat[h-1]) / scale[h-1]
                level = (1 + np.sum(sdesc[h-1] >= s)) / (sdesc[h-1].size + 1)
                fail = (level < alpha[h-1] if mode == ActionMode.ORDINARY else
                        mode == ActionMode.AGGRESSIVE_EXTREME and h == 1)
                new_scores.append((h-1, s))
                self.records[k]["y"][h-1] = y_now
                self.records[k]["viol"][h-1] = bool(fail)
                self.records[k]["level"][h-1] = level
                if h == H:
                    del self._pending[k]
            origins.append(k); viol.append(bool(fail)); levels.append(level)
            avail.append(present); active.append(present)
        for h, s in new_scores:                       # reveal after scoring this round
            self.scores[h].append(s)
        self._last_observe = t
        return HorizonObservationBatch(jnp.asarray(origins, jnp.int32), jnp.arange(1, H+1),
            jnp.asarray(viol, bool), jnp.asarray(levels, self.dtype),
            jnp.asarray(avail, bool), jnp.asarray(active, bool))

    def prepare(self, t: int, grid: AlphaGrid) -> ForecastInputs:
        if t != self._last_observe or t >= self.config.t_max or t == self._last_prepare:
            raise ValueError("prepare once after observing and before issuing")
        self._current = self._snapshot(t)
        g = np.asarray(grid.values)
        _, scale, sdesc = self._current
        curves = np.stack([2*scale[h]*_half_width(sdesc[h], g) for h in range(len(sdesc))])
        self._last_prepare = t
        return ForecastInputs(jnp.asarray(curves, self.dtype),
                              jnp.ones(self.config.history.horizon, bool),
                              jnp.zeros(self.config.history.covariate_size, self.dtype))

    def issue(self, t: int, action_mode: ActionMode, alpha: Optional[jax.Array]) -> None:
        if t != self.issued_count or t != self._last_prepare:
            raise ValueError("issue exactly once after preparing current geometry")
        H = self.config.history.horizon
        mode = ActionMode(action_mode)
        yhat, scale, sdesc = self._current
        if mode == ActionMode.ORDINARY:
            a = np.asarray(alpha, float)
            hw = np.array([scale[h]*_half_width(sdesc[h], a[h]) for h in range(H)], float)
        else:
            a = np.full(H, np.nan)
            hw = np.full(H, np.inf if mode == ActionMode.CONSERVATIVE_EXTREME else np.nan)
        self._pending[t] = (mode, a, yhat, scale, sdesc)
        self.records[t] = dict(mode=int(mode), alpha=a, yhat=yhat, half_width=hw,
                               y=np.full(H, np.nan), viol=np.zeros(H, bool), level=np.full(H, np.nan),
                               scale=scale, sdesc=sdesc)
        self.issued_count += 1
