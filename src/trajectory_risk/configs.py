"""Python-only configuration. Validate and configure JAX before array creation.

Annotations deliberately remain evaluated types: Pyrallis inspects dataclass
fields directly. RiskMode is serialized by its value, not by its member name.
"""

from dataclasses import dataclass, field
from enum import Enum
from math import ceil, isfinite, log
from pathlib import Path
from typing import Optional

import pyrallis
from pyrallis import cfgparsing, utils
from pyrallis.argparsing import ArgumentParser


class RiskMode(str, Enum):
    CERTIFIED = "certified"
    EXPERIMENTAL_PLUGIN = "experimental_plugin"


pyrallis.decode.register(RiskMode, RiskMode)
pyrallis.encode.register(RiskMode, lambda mode: mode.value)


@dataclass
class JaxConfig:
    enable_x64: bool = True
    platform: str = "auto"


@dataclass
class HistoryConfig:
    horizon: int = 10
    alpha_min: float = 0.001
    alpha_max: float = 0.25
    grid_size: int = 51
    grid_values: tuple[float, ...] = ()  # Empty means a linear grid on the box.
    capacity: int = 512
    covariate_size: int = 0
    alpha_reference_init: tuple[float, ...] = ()  # Empty means clip(beta / H).


@dataclass
class DependenceConfig:
    kind: str = "adaptive"
    assumption: str = "stable_certified"
    delta: float = 0.05
    window: int = 100
    windows: tuple[int, ...] = (16, 32, 64, 128)
    meta_eta: float = 0.1
    # Drift certification is deferred; v1 supports stable or unassumed targets.


@dataclass
class SignalConfig:
    kind: str = "completed_window"
    window: int = 100
    refresh_pending: bool = True
    prior_probability: Optional[float] = None  # None means beta.
    learning_rate: float = 0.01


@dataclass
class ControllerConfig:
    kind: str = "aci"
    budget_init: float = 0.1
    gamma: float = 0.01
    # DtACI (Gibbs & Candes), defaults from the earlier DtACI_ALPaCA system.
    dtaci_gammas: tuple[float, ...] = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128)
    dtaci_eta: float = 2.72
    dtaci_sigma: float = 0.001
    kp: float = 0.01
    ki: float = 0.0
    kd: float = 0.0
    integral_limit: Optional[float] = None


@dataclass
class WeightConfig:
    kind: str = "failure_aware"
    application_weights: tuple[float, ...] = ()  # Empty means uniform c_h.
    smoothing: float = 0.05
    max_step_norm: Optional[float] = 0.05
    min_weight: float = 0.01
    max_weight: float = 0.8
    failure_window: int = 100
    eps_failure: float = 0.001
    eps_efficiency: float = 0.000001


@dataclass
class SolverConfig:
    mode: str = "rti"
    qp_backend: str = "admm"
    max_sqp_iterations: int = 1
    feasibility_tol: float = 0.000001
    step_tol: float = 0.00001
    hessian_regularization: float = 0.000001
    line_search: bool = True
    warm_start: bool = True
    forest_kind: str = "maximum_spanning_tree"
    fixed_forest: tuple[tuple[int, int], ...] = ()  # Horizon numbers are 1-based.
    qp_max_iterations: int = 400
    qp_abs_tol: float = 1e-8
    qp_rel_tol: float = 1e-7
    qp_initial_penalty: float = 1.0
    qp_min_penalty: float = 1e-4
    qp_max_penalty: float = 1e6
    qp_penalty_update_interval: int = 25
    qp_proximal_regularization: float = 1e-9
    line_search_steps: int = 20
    restoration_steps: int = 24
    armijo: float = 1e-4
    backtrack_factor: float = 0.5
    merit_penalty: float = 10.0


@dataclass
class SafetyConfig:
    eta_q: float = 0.01
    q_init: float = 0.0
    q_lower: float = -1.0
    q_upper: float = 1.0
    # "threshold": hard override at q_lower/q_upper (note eq. 20). "tangent": smooth
    # supervision (note section 2.5, eqs. 24-25); q_lower/q_upper are then unused.
    kind: str = "threshold"
    tangent_gain: float = 1.0  # K
    tangent_limit: float = 6.0  # L; must exceed log((1-eps)/eps)
    tangent_eps: float = 0.01  # eps_b: proposals are clipped to [eps, 1-eps]
    tangent_envelope: float = 50.0  # B_t = envelope * max(t, 1)**power, in trajectories
    tangent_envelope_power: float = 0.0  # 0 gives O(1/T) calibration; < 1 keeps B_t = o(t)
    # Lowest ordinary budget, normally H*alpha_min: eq. (25) issues an ordinary allocation
    # for all v > -L, but none exists below it. Only ordinary rounds change, so eq. (26) holds.
    tangent_budget_floor: float = 0.0


@dataclass
class RunnerConfig:
    jit: bool = True
    drain: bool = True
    seed: int = 0
    collect_trace: bool = False
    output_dir: Optional[str] = None


@dataclass
class ExperimentConfig:
    t_max: int  # Physical issuance horizon: required, no hidden meta-horizon.
    beta: float = 0.1
    risk_mode: RiskMode = RiskMode.CERTIFIED
    jax: JaxConfig = field(default_factory=JaxConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    dependence: DependenceConfig = field(default_factory=DependenceConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)

    def validate(self) -> None:
        """Validate on the Python boundary, never inside a compiled kernel."""
        h, d, s, c, w, sol, guard = (
            self.history, self.dependence, self.signal, self.controller,
            self.weights, self.solver, self.safety,
        )
        _positive_int("t_max", self.t_max)
        _positive_int("history.horizon", h.horizon)
        _check(self.t_max + h.horizon <= 2147483647,
               "t_max + horizon must fit the numerical state's int32 time indices")
        _check(type(self.runner.seed) is int and 0 <= self.runner.seed <= 4294967295,
               "runner.seed must be an integer in [0, 2**32-1]")
        for name in ("jit", "drain", "collect_trace"):
            _check(type(getattr(self.runner, name)) is bool, f"runner.{name} must be boolean")
        if self.runner.output_dir is not None:
            _check(isinstance(self.runner.output_dir, str) and bool(self.runner.output_dir.strip()),
                   "runner.output_dir must be a nonempty string or null")
        _positive_int("history.capacity", h.capacity)
        _positive_int("history.grid_size", h.grid_size)
        _check(h.grid_size >= 2, "history.grid_size must be at least 2")
        _check(type(h.covariate_size) is int and h.covariate_size >= 0,
               "history.covariate_size must be a nonnegative integer")
        _check(_finite(self.beta) and 0 < self.beta < 1, "beta must be in (0, 1)")
        _check(_finite(h.alpha_min, h.alpha_max) and 0 < h.alpha_min < h.alpha_max < 1,
               "history requires 0 < alpha_min < alpha_max < 1")
        _check(isinstance(self.risk_mode, RiskMode), "risk_mode must be a RiskMode enum")
        _choice("jax.platform", self.jax.platform, ("auto", "cpu", "gpu", "tpu"))
        _choice("dependence.kind", d.kind, ("full_history", "fixed_window", "adaptive"))
        _choice("dependence.assumption", d.assumption, ("stable_certified", "unassumed"))
        _check(_finite(d.delta) and 0 < d.delta < 1, "dependence.delta must be in (0, 1)")
        _positive_int("dependence.window", d.window)
        _check(bool(d.windows), "dependence.windows must be nonempty")
        for window in d.windows:
            _positive_int("dependence.windows entry", window)
        _check(all(a < b for a, b in zip(d.windows, d.windows[1:])),
               "dependence.windows must be strictly increasing")
        _check(_finite(d.meta_eta) and 0 < d.meta_eta <= 0.5,
               "dependence.meta_eta must be in (0, 0.5]")
        if d.kind == "adaptive" and h.horizon > 1:
            _check(ceil(self.t_max / h.horizon) >= 2,
                   "adaptive dependence requires ceil(t_max / d_e) >= 2 for every edge")
        _choice("signal.kind", s.kind,
                ("reveal_only", "pending_valid", "completed_window", "learned"))
        _positive_int("signal.window", s.window)
        _positive("signal.learning_rate", s.learning_rate)
        if s.prior_probability is not None:
            _check(_finite(s.prior_probability) and 0 <= s.prior_probability <= 1,
                   "signal.prior_probability must be in [0, 1]")
            if s.kind == "learned":
                _check(0 < s.prior_probability < 1,
                       "a finite logistic initialization requires signal.prior_probability in (0, 1)")
        if c.kind == "fixed_share_gain":
            raise NotImplementedError("fixed_share_gain is deferred until a principled "
                                      "counterfactual expert loss is specified")
        _choice("controller.kind", c.kind, ("aci", "pid", "dtaci"))
        if c.kind == "dtaci":
            _check(len(c.dtaci_gammas) > 0 and all(_finite(g) and g > 0 for g in c.dtaci_gammas),
                   "controller.dtaci_gammas must be nonempty and positive")
            _check(_finite(c.dtaci_eta) and c.dtaci_eta > 0, "controller.dtaci_eta must be positive")
            _check(_finite(c.dtaci_sigma) and 0 <= c.dtaci_sigma < 1,
                   "controller.dtaci_sigma must be in [0, 1)")
        _check(_finite(c.budget_init) and 0 <= c.budget_init <= 1,
               "controller.budget_init must be in [0, 1]")
        _positive("controller.gamma", c.gamma)
        _check(_finite(c.kp, c.ki, c.kd) and min(c.kp, c.ki, c.kd) >= 0,
               "PID gains must be finite and nonnegative")
        if c.integral_limit is not None:
            _positive("controller.integral_limit", c.integral_limit)
        _choice("weights.kind", w.kind, ("static", "marginal_efficiency", "failure_aware"))
        _positive_int("weights.failure_window", w.failure_window)
        _check(_finite(w.smoothing) and 0 < w.smoothing <= 1,
               "weights.smoothing must be in (0, 1]")
        _check(_finite(w.min_weight, w.max_weight) and 0 < w.min_weight <= w.max_weight <= 1,
               "weights require 0 < min_weight <= max_weight <= 1")
        _check(h.horizon * w.min_weight <= 1 <= h.horizon * w.max_weight,
               "weights bounds make the normalized simplex infeasible")
        if w.max_step_norm is not None:
            _check(_finite(w.max_step_norm) and w.max_step_norm >= 0,
                   "weights.max_step_norm must be finite and nonnegative")
        _positive("weights.eps_failure", w.eps_failure)
        _positive("weights.eps_efficiency", w.eps_efficiency)
        if w.application_weights:
            _check(len(w.application_weights) == h.horizon and
                   all(_finite(x) and x > 0 for x in w.application_weights),
                   "weights.application_weights must contain H finite positive entries")
            total = sum(w.application_weights)
            _check(all(w.min_weight <= x / total <= w.max_weight for x in w.application_weights),
                   "normalized application weights must satisfy weights bounds for static burn-in")
        _choice("solver.mode", sol.mode, ("rti", "full_sqp"))
        _choice("solver.qp_backend", sol.qp_backend, ("admm",))
        _positive_int("solver.max_sqp_iterations", sol.max_sqp_iterations)
        _positive("solver.feasibility_tol", sol.feasibility_tol)
        _positive("solver.step_tol", sol.step_tol)
        _positive("solver.hessian_regularization", sol.hessian_regularization)
        validate_solver_numerics(sol)
        _choice("solver.forest_kind", sol.forest_kind, ("maximum_spanning_tree", "fixed"))
        _validate_forest(sol.fixed_forest, h.horizon)
        _check(sol.forest_kind == "fixed" or not sol.fixed_forest,
               "solver.fixed_forest is only used with forest_kind=fixed")
        _positive("safety.eta_q", guard.eta_q)
        _check(_finite(guard.q_lower, guard.q_upper, guard.q_init) and
               guard.q_lower < guard.q_upper and guard.q_lower <= guard.q_init <= guard.q_upper,
               "safety requires q_lower < q_upper and q_init within those thresholds")
        _choice("safety.kind", guard.kind, ("threshold", "tangent"))
        if guard.kind == "tangent":
            _positive("safety.tangent_gain", guard.tangent_gain)
            _positive("safety.tangent_envelope", guard.tangent_envelope)
            _check(_finite(guard.tangent_eps) and 0 < guard.tangent_eps < 0.5,
                   "safety.tangent_eps must be in (0, 0.5)")
            _check(_finite(guard.tangent_limit) and
                   guard.tangent_limit > log((1-guard.tangent_eps)/guard.tangent_eps),
                   "safety.tangent_limit must exceed log((1-eps)/eps)")
            _check(_finite(guard.tangent_envelope_power) and 0 <= guard.tangent_envelope_power < 1,
                   "safety.tangent_envelope_power must be in [0, 1)")
            _check(guard.q_init == 0, "the tangent safeguard requires q_init = 0")
            _check(_finite(guard.tangent_budget_floor) and 0 <= guard.tangent_budget_floor < 1,
                   "safety.tangent_budget_floor must be in [0, 1)")
        if h.grid_values:
            _check(len(h.grid_values) == h.grid_size and
                   all(_finite(x) and 0 < x < 1 for x in h.grid_values) and
                   all(a < b for a, b in zip(h.grid_values, h.grid_values[1:])),
                   "history.grid_values must contain grid_size strictly increasing values in (0, 1)")
            _check(h.grid_values[0] <= h.alpha_min and h.grid_values[-1] >= h.alpha_max,
                   "history.grid_values must cover the ordinary alpha box")
        if h.alpha_reference_init:
            _check(len(h.alpha_reference_init) == h.horizon and
                   all(_finite(x) and h.alpha_min <= x <= h.alpha_max for x in h.alpha_reference_init),
                   "history.alpha_reference_init must contain H entries inside the alpha box")
        dep_window = {"full_history": 0, "fixed_window": d.window, "adaptive": max(d.windows)}[d.kind]
        sig_window = s.window if s.kind == "completed_window" else 0
        util_window = w.failure_window if w.kind == "failure_aware" else 0
        minimum = h.horizon + max(dep_window, sig_window, util_window)
        # One spare slot prevents overwriting an origin while its last query is read.
        _check(h.capacity >= max(h.horizon + 1, minimum),
               f"history.capacity must be at least {max(h.horizon + 1, minimum)} for selected algorithms")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(*values: float) -> bool:
    return all(isfinite(x) for x in values)


def _positive(name: str, value: float) -> None:
    _check(_finite(value) and value > 0, f"{name} must be finite and positive")


def _positive_int(name: str, value: int) -> None:
    _check(type(value) is int and value > 0, f"{name} must be a positive integer")


def _choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    _check(value in choices, f"{name} must be one of {choices}, got {value!r}")


def _validate_forest(edges: tuple[tuple[int, int], ...], horizon: int) -> None:
    parents = list(range(horizon + 1))

    def root(v: int) -> int:
        while parents[v] != v:
            v = parents[v]
        return v

    for edge in edges:
        _check(len(edge) == 2, "fixed forest edges must be horizon pairs")
        s, r = edge
        _check(type(s) is int and type(r) is int and 1 <= s < r <= horizon,
               "fixed forest edges require 1 <= s < r <= H")
        a, b = root(s), root(r)
        _check(a != b, "fixed forest contains a cycle or duplicate edge")
        parents[a] = b


def validate_solver_numerics(config: SolverConfig) -> None:
    """Shared startup validation for the standalone solver and experiments."""
    _choice("solver.mode", config.mode, ("rti", "full_sqp"))
    _choice("solver.qp_backend", config.qp_backend, ("admm",))
    for name in ("max_sqp_iterations", "qp_max_iterations", "qp_penalty_update_interval",
                 "line_search_steps", "restoration_steps"):
        _positive_int(f"solver.{name}", getattr(config, name))
    for name in ("feasibility_tol", "step_tol", "hessian_regularization", "qp_abs_tol", "qp_rel_tol",
                 "qp_initial_penalty", "qp_min_penalty", "qp_max_penalty",
                 "qp_proximal_regularization", "merit_penalty"):
        _positive(f"solver.{name}", getattr(config, name))
    _check(config.qp_min_penalty <= config.qp_initial_penalty <= config.qp_max_penalty,
           "QP penalties require min <= initial <= max")
    _check(_finite(config.armijo, config.backtrack_factor) and
           0 < config.armijo < 1 and 0 < config.backtrack_factor < 1,
           "solver.armijo and backtrack_factor must be in (0, 1)")


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open() as stream:
        config = pyrallis.load(ExperimentConfig, stream)
    config.validate()
    return config


class _ClosedFileArgumentParser(ArgumentParser):
    """Preserve Pyrallis overrides while closing YAML files deterministically.

    Pyrallis 0.3.1 opens its config file without a context manager. Override
    postprocessing locally so CLI runs also pass ResourceWarning-as-error checks.
    """

    def _postprocessing(self, namespace):
        overrides = {name: cfgparsing.parse_string(value) for name, value in vars(namespace).items()}
        path = overrides.pop(utils.CONFIG_ARG, self.config_path)
        if path is not None:
            with Path(path).open() as stream:
                contents = cfgparsing.load_config(stream)
            merged = utils.flatten(contents, sep=".")
            merged.update(overrides)
        else:
            merged = overrides
        return pyrallis.decode(self.config_class, utils.deflatten(merged, sep="."))


def parse_config(args: Optional[list[str]] = None) -> ExperimentConfig:
    config = _ClosedFileArgumentParser(ExperimentConfig).parse_args(args)
    config.validate()
    return config


def configure_jax(config: JaxConfig) -> None:
    """Call before initializing numerical states or compiling any kernels."""
    import jax

    jax.config.update("jax_enable_x64", config.enable_x64)
    if config.platform != "auto":
        jax.config.update("jax_platforms", config.platform)
