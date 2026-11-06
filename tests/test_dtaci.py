import jax.numpy as jnp
import numpy as np
import pytest

from trajectory_risk.configs import ControllerConfig
from trajectory_risk.controllers import DtACIController, build_controller
from trajectory_risk.examples import SyntheticIntervalAdapter
from trajectory_risk.history import ActionMode
from trajectory_risk.runner import run_experiment

from test_end_to_end import assert_accounting_and_transcript, config, outcomes


def dtaci_config(t_max=30):
    c = config("adaptive", "completed_window", t_max=t_max)
    c.controller.kind = "dtaci"
    c.controller.dtaci_gammas = (0.01, 0.05, 0.2)
    c.validate()
    return c


def reference(c, ys, trace):
    """NumPy DtACI from the transcript's issued alphas and synthetic critical levels."""
    h, beta = c.history.horizon, c.beta
    gammas = np.asarray(c.controller.dtaci_gammas)
    experts = np.full(gammas.size, c.controller.budget_init)
    weights = np.full(gammas.size, 1/gammas.size)
    proposals, criticals, unit = [], {}, np.full(h, 1/h)
    for t, record in enumerate(trace):
        k = t-h
        if 0 <= k < c.t_max:
            if trace[k]["action_mode"] == ActionMode.ORDINARY:
                unit = np.asarray(trace[k]["alpha"])/proposals[k]
            crit = np.array([1-abs(ys[k+j]) for j in range(1, h+1)])
            b_star = np.min(crit/unit)
            if trace[k]["action_mode"] == ActionMode.ORDINARY:
                criticals[k] = b_star
            u = b_star-experts
            loss = beta*u-np.minimum(u, 0)
            experts = np.clip(experts+gammas*(beta-(b_star < experts)), 0, 1)
            w = weights*np.exp(-c.controller.dtaci_eta*(loss-loss.min()))
            w /= w.sum()
            weights = (1-c.controller.dtaci_sigma)*w+c.controller.dtaci_sigma/gammas.size
        proposals.append(float(np.sum(weights*experts)))
    return np.asarray(proposals), criticals


def test_dtaci_matches_numpy_reference_and_keeps_exact_accounting():
    c = dtaci_config()
    ys = outcomes(c)
    result = run_experiment(c, SyntheticIntervalAdapter(c, ys))
    assert_accounting_and_transcript(result, c, ys)
    proposals, criticals = reference(c, ys, result.trace)
    np.testing.assert_allclose([r["proposal"] for r in result.trace], proposals, rtol=1e-10, atol=1e-12)
    assert criticals, "the stream must exercise scored origins"


def test_critical_budget_reproduces_actual_label_at_issued_budget():
    c = dtaci_config()
    ys = outcomes(c)
    trace = run_experiment(c, SyntheticIntervalAdapter(c, ys)).trace
    proposals, criticals = reference(c, ys, trace)
    h = c.history.horizon
    for k, b_star in criticals.items():
        alpha = np.asarray(trace[k]["alpha"])
        label = any(1-abs(ys[k+j]) < alpha[j-1] for j in range(1, h+1))
        assert (b_star < proposals[k]) == label


def test_dtaci_scores_extreme_origins_and_recovers_from_infeasible_budgets():
    c = dtaci_config(t_max=60)
    c.controller.budget_init = 0.0  # Infeasible: every early round is a conservative fallback.
    c.validate()
    # IID uniform outcomes: P(B* < b) ~ b, so DtACI must climb above the 3*alpha_min floor.
    ys = np.random.default_rng(0).uniform(-1, 1, c.t_max+3)
    trace = run_experiment(c, SyntheticIntervalAdapter(c, ys)).trace
    assert trace[0]["action_mode"] == ActionMode.CONSERVATIVE_EXTREME
    assert any(r["action_mode"] == ActionMode.ORDINARY for r in trace[:c.t_max])
    proposals, _ = reference(c, ys, trace)
    np.testing.assert_allclose([r["proposal"] for r in trace], proposals, rtol=1e-10, atol=1e-12)


def test_dtaci_skips_unusable_origins_and_validates_configuration():
    c = ControllerConfig(kind="dtaci")
    controller = build_controller(c, beta=0.1, horizon=3)
    assert isinstance(controller, DtACIController)
    with pytest.raises(ValueError, match="horizon"):
        build_controller(c)
    with pytest.raises(ValueError, match="gammas"):
        DtACIController(ControllerConfig(kind="dtaci", dtaci_gammas=()), 0.1, 3)
    from trajectory_risk.runner import init_runner_state
    cfg = dtaci_config()
    state = controller.init_state(jnp.float64)
    history = init_runner_state(cfg).history
    for t in range(5):  # Empty history: clock advances, experts stay put.
        update = controller.update(state, 0.0, t, history)
        assert int(update.error) == 0
        state = update.state
    assert int(state.expert_updates) == 0
    np.testing.assert_allclose(state.budget, c.budget_init)
