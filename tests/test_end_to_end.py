from dataclasses import replace
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trajectory_risk.configs import ExperimentConfig, RiskMode, load_config, parse_config
from trajectory_risk.examples import SyntheticIntervalAdapter
from trajectory_risk.history import ActionMode, UpdateError
from trajectory_risk.runner import (
    ForecastInputs, HorizonObservationBatch, OnlineRunner, build_round_kernels,
    init_runner_state, resolve_observations, run_experiment,
    _json_value,
)
from trajectory_risk.solver import SolverStatus
from test_signals import assert_tree_equal


def config(dep="adaptive", signal="learned", t_max=12):
    c = ExperimentConfig(t_max=t_max, beta=.2)
    c.history.horizon = 3
    c.history.capacity = 8  # Reuse slots while older origins still contribute to totals.
    c.history.grid_size = 4
    c.history.grid_values = (.05, .1, .2, .3)
    c.history.alpha_min, c.history.alpha_max = .05, .25
    c.history.covariate_size = 1
    c.dependence.kind, c.dependence.window, c.dependence.windows = dep, 2, (1, 2)
    c.signal.kind, c.signal.window, c.signal.learning_rate = signal, 3, .2
    c.weights.failure_window = 2
    c.controller.budget_init, c.controller.gamma = .3, .02
    c.controller.kp, c.controller.ki, c.controller.kd = .02, .003, .004
    c.solver.qp_max_iterations = 150
    c.runner.collect_trace = True
    c.validate()
    return c


def outcomes(c):
    return np.resize([0., .99, .2, -.95, .4, .85, -.7], c.t_max+3)


def assert_accounting_and_transcript(result, c, ys):
    assert len(result.trace) == c.t_max+3
    issued = result.trace[:c.t_max]
    actual_labels, reveal_times = [], []
    for k, record in enumerate(issued):
        mode = record["action_mode"]
        if mode == ActionMode.ORDINARY:
            failures = [1-abs(ys[k+h]) < record["alpha"][h-1] for h in range(1, 4)]
            assert record["solver"]["feasible"]
            assert record["solver"]["acceptance_risk"] <= record["budget"]+c.solver.feasibility_tol
            assert record["solver"]["certified"] == (c.risk_mode == RiskMode.CERTIFIED)
            np.testing.assert_allclose(record["reference_after"], record["alpha"])
        else:
            assert record["alpha"] == [None]*3
            np.testing.assert_allclose(record["reference_before"], record["reference_after"])
            failures = [mode == ActionMode.AGGRESSIVE_EXTREME, False, False]
        label = int(any(failures))
        actual_labels.append(label)
        reveal_times.append(k+failures.index(True)+1 if label else k+3)
    contributions = np.zeros(c.t_max)
    prior_queue = 0.
    accumulated_control = 0.
    for t, record in enumerate(result.trace):
        packet = record["observations"]
        g = sum(v for v, active in zip(packet["increments"], packet["event_mask"]) if active)
        np.testing.assert_allclose(record["controller_increment"], g+prior_queue, atol=1e-13)
        accumulated_control += record["controller_increment"]
        for p in [packet, record["issuance"]]:
            for k, increment, active in zip(p["origin_id"], p["increments"], p["event_mask"]):
                if active:
                    assert 0 <= k < c.t_max
                    contributions[k] += increment
            for k, label, active in zip(p["resolved_origin_id"], p["resolved_labels"], p["resolved_mask"]):
                if active:
                    assert label == actual_labels[k] and reveal_times[k] == t
        expected_q = c.safety.q_init+c.safety.eta_q*sum(c.beta-z for z, rt in zip(actual_labels, reveal_times) if rt <= t)
        np.testing.assert_allclose(record["q"], expected_q, atol=1e-13)
        prior_queue = record["queued_increment"]
        assert record["issued"] == (t < c.t_max)
    np.testing.assert_allclose(contributions, c.beta-np.array(actual_labels), atol=1e-13)
    np.testing.assert_allclose(accumulated_control, np.sum(contributions), atol=1e-13)
    assert result.summary["failures"] == sum(actual_labels)
    assert result.summary["pending"] == 0
    assert abs(result.summary["accounting_identity_error"]) < 1e-13
    assert abs(result.summary["signal_identity_error"]) < 1e-13
    assert abs(result.summary["queue_identity_error"]) < 1e-13
    assert int(result.state.controller.updates) == c.t_max+3
    assert int(result.state.utility.last_time) == c.t_max-1
    assert int(result.state.dependence.last_time) == c.t_max+2
    assert bool(jnp.all(result.state.history.horizon_resolved[result.state.history.valid_origin]))


@pytest.mark.parametrize("dep", ["full_history", "fixed_window", "adaptive"])
@pytest.mark.parametrize("signal", ["reveal_only", "pending_valid", "completed_window", "learned"])
def test_algorithm_combinations_accounting_and_per_origin_telescoping(dep, signal):
    c = config(dep, signal)
    c.controller.kind = "pid" if signal == "learned" else "aci"
    c.weights.kind = {"reveal_only": "static", "pending_valid": "marginal_efficiency"}.get(signal, "failure_aware")
    # Full SQP selection uses the same orchestration, with fixed numeric loop limits.
    if dep == "full_history" and signal == "reveal_only":
        c.solver.mode, c.solver.max_sqp_iterations = "full_sqp", 3
    ys = outcomes(c)
    result = run_experiment(c, SyntheticIntervalAdapter(c, ys))
    assert_accounting_and_transcript(result, c, ys)
    assert any(all(record["edge_ready"]) for record in result.trace)


def test_no_lookahead_reproducibility_and_eager_jit_equivalence():
    c = config(t_max=8)
    ys = outcomes(c)
    a = run_experiment(c, SyntheticIntervalAdapter(c, ys))
    future_changed = ys.copy()
    future_changed[5:] = .999
    b = run_experiment(c, SyntheticIntervalAdapter(c, future_changed))
    assert a.trace[:5] == b.trace[:5]
    again = run_experiment(c, SyntheticIntervalAdapter(c, ys))
    assert a.trace == again.trace and a.summary == again.summary
    c.runner.jit = False
    eager = run_experiment(c, SyntheticIntervalAdapter(c, ys))
    for left, right in zip(jax.tree.leaves(a.state), jax.tree.leaves(eager.state)):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=2e-7, equal_nan=True)


def test_extreme_actions_actual_labels_none_alpha_reference_and_solver_bypass():
    c = config(signal="pending_valid", t_max=18)
    c.safety.q_init, c.safety.q_lower, c.safety.q_upper = .015, -.015, .015
    c.safety.eta_q = .1
    ys = np.zeros(c.t_max+3)
    adapter = SyntheticIntervalAdapter(c, ys)
    result = run_experiment(c, adapter)
    assert_accounting_and_transcript(result, c, ys)
    assert result.summary["conservative_extreme"] > 0
    assert result.summary["aggressive_extreme"] > 0
    assert result.summary["ordinary"] > 0
    for record in result.trace[:c.t_max]:
        if record["action_mode"] != ActionMode.ORDINARY:
            assert record["solver"]["status"] == SolverStatus.SKIPPED_EXTREME
    assert adapter.issued_count == c.t_max and not adapter._pending
    # Counterfactual ordinary criticals remain available despite extreme issuance.
    assert result.summary["missing_critical_by_horizon"] == [0]*3


def test_failed_ordinary_solve_issues_conservative_and_drain_keeps_later_criticals():
    c = config(signal="pending_valid", t_max=8)
    c.controller.budget_init, c.controller.gamma = .01, 1e-6
    ys = np.ones(c.t_max+3)
    result = run_experiment(c, SyntheticIntervalAdapter(c, ys))
    assert_accounting_and_transcript(result, c, ys)
    assert result.summary["solver_failures"] == c.t_max
    assert result.summary["conservative_extreme"] == c.t_max
    assert result.summary["failures"] == 0  # All counterfactual ordinary sets fail.
    assert result.summary["mean_ordinary_set_size"] is None
    assert result.summary["missing_critical_by_horizon"] == [0]*3


def test_missing_counterfactuals_keep_actual_accounting_and_flag_adaptive_loss_gaps():
    c = config(t_max=12)
    ys = outcomes(c)
    availability = np.ones((c.t_max, 3), bool)
    availability[3, 1] = False
    clean = run_experiment(c, SyntheticIntervalAdapter(c, ys))
    missing = run_experiment(c, SyntheticIntervalAdapter(c, ys, availability))
    assert_accounting_and_transcript(missing, c, ys)
    assert missing.summary["missing_critical_by_horizon"] == [0, 1, 0]
    assert missing.summary["q"] == clean.summary["q"]  # Zero certified credit here in both runs.
    assert any(any(r["edge_missing_count"]) for r in missing.trace)
    assert not all(missing.trace[-1]["adaptive_regret_valid"])
    assert all(missing.trace[-1]["edge_missing_count"][i] == 0 for i in range(3))


def test_experimental_mode_and_undrained_summary_do_not_claim_completed_coverage():
    c = config(t_max=8)
    c.risk_mode = RiskMode.EXPERIMENTAL_PLUGIN
    c.runner.drain = False
    r = run_experiment(c, SyntheticIntervalAdapter(c, np.zeros(11)))
    assert len(r.trace) == c.t_max
    assert r.summary["pending"] == 3
    assert r.summary["failure_rate"] is None and r.summary["calibration_error"] is None
    assert r.summary["signal_identity_error"] is None
    assert r.summary["uncertified_ordinary"] == r.summary["ordinary"]
    assert r.summary["failure_rate_interval"] == [0., 3/8]
    for record in r.trace:
        assert not record["solver"]["certified"]


def test_callback_order_transactional_state_commit_and_rejection_before_issuance():
    c = config(t_max=8)
    calls = []

    class Instrumented(SyntheticIntervalAdapter):
        def observe(self, t):
            calls.append(("observe", t))
            b = super().observe(t)
            return replace(b, event_mask=jnp.zeros(3, bool)) if t == 1 else b

        def prepare(self, t, grid):
            calls.append(("prepare", t))
            return super().prepare(t, grid)

        def issue(self, t, action_mode, alpha):
            calls.append(("issue", t))
            assert int(runner.state.history.next_origin_id) == t
            assert int(runner.state.controller.updates) == t
            super().issue(t, action_mode, alpha)

    adapter = Instrumented(c)
    runner = OnlineRunner(c, adapter)
    with pytest.raises(ValueError, match="consecutive"):
        runner.step(1)
    assert calls == [] and not runner.aborted
    runner.step()
    previous = runner.state
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        runner.step()
    assert adapter.issued_count == 1 and calls == [("observe", 0), ("prepare", 0), ("issue", 0), ("observe", 1), ("prepare", 1)]
    assert_tree_equal(previous, runner.state)
    assert runner.aborted
    with pytest.raises(RuntimeError, match="aborted"):
        runner.step()


def test_invalid_predictable_metadata_and_callback_failure_never_commit_or_retry():
    c = config(t_max=8)

    class BadMetadata(SyntheticIntervalAdapter):
        def prepare(self, t, grid):
            current = super().prepare(t, grid)
            return replace(current, covariates=jnp.array([jnp.nan]))

    a = BadMetadata(c)
    runner = OnlineRunner(c, a)
    before = runner.state
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        runner.step()
    assert a.issued_count == 0
    assert_tree_equal(before, runner.state)

    class FailingIssue(SyntheticIntervalAdapter):
        def issue(self, *args):
            super().issue(*args)
            raise RuntimeError("external geometry write failed")

    a = FailingIssue(c)
    runner = OnlineRunner(c, a)
    before = runner.state
    with pytest.raises(RuntimeError, match="external geometry"):
        runner.step()
    assert a.issued_count == 1 and runner.aborted
    assert_tree_equal(before, runner.state)
    with pytest.raises(RuntimeError, match="aborted"):
        runner.step()
    assert a.issued_count == 1


def test_batch_rejection_is_transactional_duplicate_missing_and_wrong_reveal_time():
    c = config(t_max=8)
    adapter = SyntheticIntervalAdapter(c, np.zeros(11))
    runner = OnlineRunner(c, adapter)
    runner.step()
    state = runner.state.history
    batch = adapter.observe(1)
    resolve = jax.jit(resolve_observations)
    accepted = resolve(state, batch, 1)
    assert bool(accepted.accepted)
    for wrong in [replace(batch, event_mask=jnp.zeros(3, bool)),
                  replace(batch, origin_id=batch.origin_id.at[0].set(1)),
                  replace(batch, horizon=batch.horizon.at[1].set(1),
                          origin_id=batch.origin_id.at[1].set(0), event_mask=batch.event_mask.at[1].set(True))]:
        rejected = resolve(state, wrong, 1)
        assert not bool(rejected.accepted)
        assert_tree_equal(rejected.state, state)
    shuffled = jax.tree.map(lambda x: x[jnp.array([2, 0, 1])], batch)
    assert_tree_equal(resolve(state, shuffled, 1).state, accepted.state)


def test_complete_pure_rounds_compose_in_scan_without_python_callbacks():
    c = config(t_max=8)
    ys = jnp.asarray(outcomes(c))
    reference = run_experiment(c, SyntheticIntervalAdapter(c, ys))
    initial = init_runner_state(c)
    kernels = build_round_kernels(c, initial.history)
    horizons = jnp.arange(1, 4, dtype=jnp.int32)

    def round(state, t):
        origins = t-horizons
        slots = jnp.mod(origins, c.history.capacity)
        active = (origins >= 0) & (origins < c.t_max)
        modes, alpha = state.history.action_mode[slots], state.history.issued_alpha[slots, horizons-1]
        level = 1-jnp.abs(ys[t])
        violation = jnp.where(modes == ActionMode.ORDINARY, level < alpha,
                             (modes == ActionMode.AGGRESSIVE_EXTREME) & (horizons == 1))
        batch = HorizonObservationBatch(origins, horizons, violation,
                                        jnp.full(3, level), active, active)
        current = ForecastInputs(jnp.broadcast_to(2*(1-state.history.grid.values), (3, 4)),
                                 jnp.ones(3, bool), jnp.array([jnp.abs(ys[t])]))
        update = jax.lax.cond(t < c.t_max,
            lambda _: kernels.step(state, batch, current, t),
            lambda _: kernels.drain(state, batch, t), operand=None)
        return update.state, update.error

    final, errors = jax.jit(lambda s: jax.lax.scan(round, s, jnp.arange(c.t_max+3, dtype=jnp.int32)))(initial)
    np.testing.assert_array_equal(errors, 0)
    assert_tree_equal(final, reference.state)


def test_artifact_streaming_strict_json_seed_reproducibility_and_yaml_choices(tmp_path):
    c = config(t_max=8)
    c.runner.output_dir = str(tmp_path/"first")
    first = run_experiment(c, SyntheticIntervalAdapter(c))
    c.runner.output_dir = str(tmp_path/"second")
    second = run_experiment(c, SyntheticIntervalAdapter(c))
    assert first.trace == second.trace and first.summary == second.summary
    assert (tmp_path/"first"/"trace.jsonl").read_bytes() == (tmp_path/"second"/"trace.jsonl").read_bytes()
    assert load_config(tmp_path/"first"/"config.yaml").t_max == 8
    strict = lambda text: json.loads(text, parse_constant=lambda value: pytest.fail(f"invalid JSON: {value}"))
    assert strict((tmp_path/"first"/"summary.json").read_text()) == first.summary
    for line in (tmp_path/"first"/"trace.jsonl").read_text().splitlines():
        strict(line)
    with pytest.raises(FileExistsError):
        run_experiment(c, SyntheticIntervalAdapter(c))
    expected = {"default": ("adaptive", "aci", "completed_window"),
                "full_history_aci": ("full_history", "aci", "reveal_only"),
                "fixed_window_aci": ("fixed_window", "aci", "pending_valid"),
                "adaptive_window_aci": ("adaptive", "aci", "completed_window"),
                "adaptive_window_pid": ("adaptive", "pid", "learned")}
    for name, choice in expected.items():
        selected = load_config(Path("configs")/(name+".yaml"))
        assert (selected.dependence.kind, selected.controller.kind, selected.signal.kind) == choice
    cli = parse_config(["--config_path", "configs/synthetic_demo.yaml", "--t_max", "8",
                        "--runner.seed", "42", "--dependence.windows", "[1,2]",
                        "--runner.drain", "false"])
    assert cli.t_max == 8 and cli.runner.seed == 42 and not cli.runner.drain


def test_valid_infinite_critical_endpoints_remain_distinct_from_missing_in_json():
    batch = HorizonObservationBatch(jnp.arange(3), jnp.arange(1, 4),
        jnp.array([False, True, False]), jnp.array([jnp.inf, -jnp.inf, jnp.nan]),
        jnp.array([True, True, False]), jnp.ones(3, bool))
    serialized = json.loads(json.dumps(_json_value(batch), allow_nan=False))
    assert serialized["critical_level"] == ["+inf", "-inf", None]
    assert serialized["critical_available"] == [True, True, False]
