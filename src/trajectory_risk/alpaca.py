"""ALPaCA (Harrison et al., 2018) in JAX for scalar multi-step forecasting.

Port of the PyTorch model used in DtACI_ALPaCA. A feature network phi maps the
last ``lags`` values to a basis; the last layer K has a matrix-normal prior
(K0, Lambda0^{-1}) with diagonal noise Sigma_eps. Conditioning on n context
pairs (x_j, y_j), with y_j the H-step future of x_j, gives the closed form

    Lambda_n = Phi^T Phi + Lambda0,  K_n = Lambda_n^{-1}(Phi^T Y + Lambda0 K0),
    y | x ~ N(K_n^T phi(x), (1 + phi^T Lambda_n^{-1} phi) Sigma_eps).

Meta-training maximizes the posterior predictive likelihood on random D1
windows with a random number of context pairs. Online, the context is the
last ``n_context`` pairs whose futures are fully revealed (no lookahead).

With ``noise_adapt`` the noise Sigma_eps is rescaled by the context's mean
standardized prior-predictive residual, so the predictive spread follows local
volatility (fixed Sigma_eps cannot). Conformal calibration happens downstream.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


def _init_mlp(key, sizes):
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        params.append((jax.random.normal(sub, (a, b)) * jnp.sqrt(1.0/a), jnp.zeros(b)))
    return params


def _features(mlp, x):
    for w, b in mlp:
        x = jnp.tanh(x @ w + b)
    return x


def _prior(params):
    phi_dim = params["K0"].shape[0]
    lam0 = params["L"] @ params["L"].T + 1e-3*jnp.eye(phi_dim)
    return params["K0"], lam0, jnp.exp(params["log_sig2"])


def _posterior(params, phi_c, y_c, mask):
    k0, lam0, _ = _prior(params)
    phi_m = phi_c * mask[:, None]
    lam_n = phi_m.T @ phi_c + lam0
    k_n = jnp.linalg.solve(lam_n, phi_m.T @ y_c + lam0 @ k0)
    return k_n, lam_n


def _noise_scale(params, phi_c, y_c, mask):
    """Mean standardized prior-predictive residual of the context (1 if empty)."""
    k0, lam0, sig2 = _prior(params)
    spread = 1 + jnp.einsum("nd,nd->n", phi_c, jnp.linalg.solve(lam0, phi_c.T).T)
    z2 = jnp.mean((y_c - phi_c @ k0)**2 / sig2, axis=1) / spread
    n = jnp.sum(mask)
    s2 = jnp.where(n > 0, jnp.sum(z2*mask) / jnp.maximum(n, 1), 1.0)
    return jnp.clip(s2, 0.05, 20.0)


def _predict(params, x_c, y_c, mask, x, noise_adapt):
    phi_c = _features(params["mlp"], x_c)
    phi = _features(params["mlp"], x)
    k_n, lam_n = _posterior(params, phi_c, y_c, mask)
    _, _, sig2 = _prior(params)
    mean = phi @ k_n
    spread = 1 + jnp.einsum("...d,...d->...", phi, jnp.linalg.solve(lam_n, phi.T).T)
    scale = _noise_scale(params, phi_c, y_c, mask) if noise_adapt else 1.0
    var = spread[..., None] * sig2 * scale
    return mean, var


def _nll(params, x_c, y_c, mask, x, y, noise_adapt):
    mean, var = _predict(params, x_c, y_c, mask, x, noise_adapt)
    return jnp.mean(jnp.log(var) + (y - mean)**2 / var)


@dataclass
class ALPaCAConfig:
    lags: int = 10
    n_context: int = 20
    n_test: int = 5
    layers: tuple = (64, 32)
    lr: float = 1e-3
    iterations: int = 1500
    batch: int = 32
    noise_adapt: bool = True
    seed: int = 0


class ALPaCAForecaster:
    """Same interface as ARForecaster: fit on D1, then predict_all(y, origins, H)."""

    def __init__(self, config: ALPaCAConfig = None, horizon: int = 10):
        self.cfg = config or ALPaCAConfig()
        self.horizon = horizon

    def _pairs(self, z, origins):
        """x = lags ending at origin j, y = the H values after it (normalized)."""
        p, h = self.cfg.lags, self.horizon
        xi = origins[:, None] + np.arange(-p+1, 1)[None, :]
        yi = origins[:, None] + np.arange(1, h+1)[None, :]
        return z[xi], z[yi]

    def fit(self, y_train: np.ndarray) -> "ALPaCAForecaster":
        cfg, h = self.cfg, self.horizon
        self.mu, self.sd = float(np.mean(y_train)), float(np.std(y_train) + 1e-12)
        z = (np.asarray(y_train, float) - self.mu) / self.sd
        origins = np.arange(cfg.lags-1, len(z)-h)
        X, Y = self._pairs(z, origins)
        X, Y = jnp.asarray(X), jnp.asarray(Y)
        span = cfg.n_context + h + cfg.n_test  # test x starts after the context futures
        if len(origins) <= span:
            raise ValueError("D1 too short for ALPaCA meta-training windows")
        key = jax.random.key(cfg.seed)
        key, k1, k2, k3 = jax.random.split(key, 4)
        phi_dim = cfg.layers[-1]
        params = {"mlp": _init_mlp(k1, (cfg.lags, *cfg.layers)),
                  "K0": jax.random.normal(k2, (phi_dim, h)) * 0.01,
                  "L": jax.random.normal(k3, (phi_dim, phi_dim)) * 0.1 + jnp.eye(phi_dim),
                  "log_sig2": jnp.zeros(h)}
        ctx_idx = jnp.arange(cfg.n_context)
        test_idx = cfg.n_context + h - 1 + jnp.arange(cfg.n_test)
        noise_adapt = cfg.noise_adapt

        def batch_loss(params, starts, counts):
            def one(s, n):
                mask = (ctx_idx < n).astype(X.dtype)
                return _nll(params, X[s+ctx_idx], Y[s+ctx_idx], mask, X[s+test_idx], Y[s+test_idx],
                            noise_adapt)
            return jnp.mean(jax.vmap(one)(starts, counts))

        @jax.jit
        def step(carry, key):
            params, m, v, i = carry
            k1, k2 = jax.random.split(key)
            starts = jax.random.randint(k1, (cfg.batch,), 0, len(origins) - span)
            counts = jax.random.randint(k2, (cfg.batch,), 0, cfg.n_context + 1)
            loss, g = jax.value_and_grad(batch_loss)(params, starts, counts)
            i = i + 1
            m = jax.tree.map(lambda a, b: 0.9*a + 0.1*b, m, g)
            v = jax.tree.map(lambda a, b: 0.999*a + 0.001*b*b, v, g)
            params = jax.tree.map(lambda p, a, b: p - cfg.lr*(a/(1-0.9**i))/(jnp.sqrt(b/(1-0.999**i))+1e-8),
                                  params, m, v)
            return (params, m, v, i), loss

        zeros = jax.tree.map(jnp.zeros_like, params)
        carry = (params, zeros, zeros, jnp.asarray(0, jnp.float32))
        carry, losses = jax.lax.scan(step, carry, jax.random.split(key, cfg.iterations))
        self.params, self.losses = carry[0], np.asarray(losses)
        return self

    def predict_all(self, y: np.ndarray, origins: np.ndarray, horizon: int):
        """Forecast mean/std for each origin k using y[:k+1] only."""
        if horizon != self.horizon:
            raise ValueError("ALPaCA was meta-trained for a different horizon")
        cfg, h = self.cfg, self.horizon
        z = (np.asarray(y, float) - self.mu) / self.sd
        origins = np.asarray(origins)
        ctx = origins[:, None] - h - cfg.n_context + 1 + np.arange(cfg.n_context)[None, :]
        if ctx.min() < cfg.lags - 1:
            raise ValueError("origins too early for a full ALPaCA context")
        # Context futures end at j + H <= k: everything used is observed at origin k.
        Xc, Yc = self._pairs(z, ctx.reshape(-1))
        Xc = Xc.reshape(len(origins), cfg.n_context, cfg.lags)
        Yc = Yc.reshape(len(origins), cfg.n_context, h)
        x = z[origins[:, None] + np.arange(-cfg.lags+1, 1)[None, :]]
        mask = jnp.ones(cfg.n_context)
        predict = jax.jit(jax.vmap(lambda a, b, c: _predict(self.params, a, b, mask, c, cfg.noise_adapt)))
        mean, var = predict(jnp.asarray(Xc), jnp.asarray(Yc), jnp.asarray(x))
        return (np.asarray(mean)*self.sd + self.mu, np.sqrt(np.asarray(var))*self.sd)
