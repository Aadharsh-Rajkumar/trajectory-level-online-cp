from dataclasses import replace
from math import atan, log, pi, tan

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajectory_risk.configs import SafetyConfig
from trajectory_risk.controllers import AccountingState, TangentAccountingGuard, build_accounting_guard
from trajectory_risk.controllers import ExactAccountingGuard
from trajectory_risk.examples import SyntheticIntervalAdapter
from trajectory_risk.history import ActionMode, UpdateError
from trajectory_risk.runner import OnlineRunner, run_experiment

from test_end_to_end import assert_accounting_and_transcript, config, outcomes
from test_dtaci import dtaci_config

BETA = 0.1
SAFETY = SafetyConfig(kind="tangent", eta_q=0.01, tangent_gain=1.0, tangent_limit=6.0,
                      tangent_eps=0.01, tangent_envelope=50.0)


def account(q, t=5):
    return AccountingState(jnp.asarray(q, jnp.asarray(0.0).dtype), jnp.full(1, -1), jnp.asarray(0, jnp.int32),
                           jnp.asarray(t, jnp.int32))


def reference_v(q, b, s=SAFETY):
    margin = log((1-s.tangent_eps)/s.tangent_eps)
    c = 2/pi*atan((s.tangent_limit+margin)/s.tangent_gain)
    ratio = max(-c, min(c, q/(s.eta_q*s.tangent_envelope)))
    b = max(s.tangent_eps, min(1-s.tangent_eps, b))
    return log(b/(1-b)) + s.tangent_gain*tan(pi/2*ratio), c


def test_factory_selects_guard_and_rejects_unknown():
    assert isinstance(build_accounting_guard(SAFETY, BETA), TangentAccountingGuard)
    assert type(build_accounting_guard(SafetyConfig(), BETA)) is ExactAccountingGuard
    with pytest.raises(ValueError):
        build_accounting_guard(SafetyConfig(kind="nope"), BETA)


@pytest.mark.parametrize("bad", [dict(tangent_limit=4.0), dict(tangent_gain=0.0), dict(tangent_eps=0.5),
                                 dict(tangent_envelope_power=1.0), dict(q_init=0.1)])
def test_invalid_tangent_settings_rejected_at_construction(bad):
    with pytest.raises(ValueError):
        build_accounting_guard(replace(SAFETY, **bad), BETA)


@pytest.mark.parametrize("b", [0.05, 0.1, 0.5, 0.9])
def test_zero_accounting_state_leaves_the_proposal_unchanged(b):
    decision = jax.jit(build_accounting_guard(SAFETY, BETA).protect)(account(0.0), b)
    assert int(decision.action_mode) == ActionMode.ORDINARY and not bool(decision.overridden)
    np.testing.assert_allclose(decision.budget, b, atol=1e-12)


@pytest.mark.parametrize("q", [-0.45, -0.2, -0.05, 0.05, 0.2, 0.45])
@pytest.mark.parametrize("b", [0.0, 0.1, 0.3, 1.0])
def test_budget_matches_equations_24_and_25(q, b):
    v, _ = reference_v(q, b)
    decision = build_accounting_guard(SAFETY, BETA).protect(account(q), b)
    if v <= -SAFETY.tangent_limit:
        assert int(decision.action_mode) == ActionMode.CONSERVATIVE_EXTREME and float(decision.budget) == 0
    elif v >= SAFETY.tangent_limit:
        assert int(decision.action_mode) == ActionMode.AGGRESSIVE_EXTREME and float(decision.budget) == 1
    else:
        assert int(decision.action_mode) == ActionMode.ORDINARY
        np.testing.assert_allclose(decision.budget, 1/(1+np.exp(-v)), rtol=1e-12)


def test_correction_is_monotone_in_q():
    guard = build_accounting_guard(SAFETY, BETA)
    budgets = [float(guard.protect(account(q), 0.2).budget) for q in np.linspace(-0.3, 0.3, 13)]
    assert np.all(np.diff(budgets) > 0)


@pytest.mark.parametrize("b", [0.0, 0.01, 0.5, 0.99, 1.0])
def test_envelope_forces_the_extremes_for_every_proposal(b):
    guard = build_accounting_guard(SAFETY, BETA)
    a = guard.saturation*SAFETY.tangent_envelope*SAFETY.eta_q*(1+1e-12)
    assert int(guard.protect(account(-a), b).action_mode) == ActionMode.CONSERVATIVE_EXTREME
    assert int(guard.protect(account(a), b).action_mode) == ActionMode.AGGRESSIVE_EXTREME


def test_growing_envelope_weakens_the_correction_over_time():
    guard = build_accounting_guard(replace(SAFETY, tangent_envelope_power=0.5), BETA)
    early, late = (float(guard.protect(account(-0.3, t), 0.2).budget) for t in (1, 400))
    assert early < late < 0.2


@pytest.mark.parametrize("q,b", [(float("nan"), 0.2), (0.0, float("nan")), (0.0, 1.5)])
def test_invalid_inputs_fail_closed(q, b):
    decision = build_accounting_guard(SAFETY, BETA).protect(account(q), b)
    assert int(decision.error) == UpdateError.INVALID_INPUT
    assert int(decision.action_mode) == ActionMode.CONSERVATIVE_EXTREME and float(decision.budget) == 0


def tangent(c, envelope):
    c.safety = replace(c.safety, kind="tangent", eta_q=0.1, tangent_envelope=envelope)
    c.validate()
    return c


@pytest.mark.parametrize("make", [lambda: config("adaptive", "completed_window", t_max=30), lambda: dtaci_config(30)])
def test_exact_accounting_and_theorem_2_bound_hold_end_to_end(make):
    c = tangent(make(), 2.0)
    ys = outcomes(c)
    result = run_experiment(c, SyntheticIntervalAdapter(c, ys))
    assert_accounting_and_transcript(result, c, ys)
    guard, h = build_accounting_guard(c.safety, c.beta), c.history.horizon
    a = guard.saturation*c.safety.tangent_envelope
    q = np.array([r["q"] for r in result.trace])/c.safety.eta_q
    assert np.all(q >= -a-h*(1-c.beta)-1e-9) and np.all(q <= a+h*c.beta+1e-9)  # eq. 26
    bound = (a + 2*h*max(c.beta, 1-c.beta))/c.t_max                            # eq. 27
    assert abs(result.summary["failure_rate"]-c.beta) <= bound


def test_dtaci_scores_origins_against_the_supervised_budget():
    c = tangent(dtaci_config(30), 2.0)
    runner = OnlineRunner(c, SyntheticIntervalAdapter(c, outcomes(c)))
    supervised = False
    for t in range(c.t_max):
        d = runner.step()
        slot = t % (c.history.horizon+1)
        if int(d.action_mode) == ActionMode.ORDINARY:
            np.testing.assert_allclose(runner.state.controller.issued_budget[slot], d.budget)
            supervised |= abs(float(d.budget)-float(d.proposal)) > 1e-6
    assert supervised


def test_budget_floor_keeps_low_budgets_ordinary_without_touching_the_extremes():
    guard = build_accounting_guard(replace(SAFETY, tangent_budget_floor=0.04), BETA)
    low = guard.protect(account(-0.3), 0.05)
    assert int(low.action_mode) == ActionMode.ORDINARY
    np.testing.assert_allclose(low.budget, 0.04)
    a = guard.saturation*SAFETY.tangent_envelope*SAFETY.eta_q*(1+1e-12)
    saturated = guard.protect(account(-a), 0.05)
    assert int(saturated.action_mode) == ActionMode.CONSERVATIVE_EXTREME and float(saturated.budget) == 0
