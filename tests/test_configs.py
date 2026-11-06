from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pyrallis
import pytest

from trajectory_risk.configs import (
    ExperimentConfig, RiskMode, load_config, parse_config,
)
from trajectory_risk.runner import init_runner_state


def test_example_config_and_yaml_roundtrip(tmp_path):
    cfg = load_config(Path(__file__).parents[1] / "configs/default.yaml")
    assert cfg.t_max == 100000
    assert cfg.risk_mode is RiskMode.CERTIFIED
    assert cfg.dependence.windows == (16, 32, 64, 128)
    path = tmp_path / "roundtrip.yaml"
    with path.open("w") as stream:
        pyrallis.dump(cfg, stream)
    assert "risk_mode: certified" in path.read_text()
    assert load_config(path) == cfg


def test_cli_override_and_experimental_enum():
    cfg = parse_config(["--t_max", "100000", "--risk_mode", "experimental_plugin",
                        "--history.capacity", "1024"])
    assert cfg.risk_mode is RiskMode.EXPERIMENTAL_PLUGIN
    assert cfg.history.capacity == 1024


def test_required_physical_horizon_and_no_legacy_booleans(tmp_path):
    with pytest.raises(TypeError):
        ExperimentConfig()
    path = tmp_path / "legacy.yaml"
    path.write_text("t_max: 100000\nrequire_certificate: true\n")
    with pytest.raises(Exception, match="require_certificate"):
        load_config(path)
    path.write_text("t_max: 100000\ndependence:\n  n_max: 1000\n")
    with pytest.raises(Exception, match="n_max"):
        load_config(path)


@pytest.mark.parametrize("group,changes,message", [
    ("history", {"capacity": 137}, "capacity"),
    ("history", {"alpha_min": 0.3}, "alpha_min"),
    ("history", {"grid_values": (0.1, 0.1)}, "grid_values"),
    ("history", {"alpha_reference_init": (0.1,)}, "alpha_reference_init"),
    ("history", {"covariate_size": -1}, "covariate_size"),
    ("dependence", {"windows": (32, 16)}, "strictly increasing"),
    ("dependence", {"meta_eta": 0.6}, "meta_eta"),
    ("dependence", {"delta": float("nan")}, "delta"),
    ("dependence", {"assumption": "drift_certified"}, "assumption"),
    ("signal", {"prior_probability": 1.1}, "prior_probability"),
    ("controller", {"gamma": -0.1}, "gamma"),
    ("controller", {"budget_init": 2.0}, "budget_init"),
    ("weights", {"min_weight": 0.11}, "simplex"),
    ("weights", {"max_weight": 0.09}, "simplex"),
    ("weights", {"max_step_norm": -1}, "max_step_norm"),
    ("weights", {"application_weights": (0.1,)}, "application_weights"),
    ("solver", {"forest_kind": "fixed", "fixed_forest": ((1, 2), (2, 3), (1, 3))}, "cycle"),
    ("solver", {"forest_kind": "fixed", "fixed_forest": ((0, 1),)}, "1 <= s"),
    ("solver", {"max_sqp_iterations": 0}, "max_sqp_iterations"),
    ("safety", {"q_init": 2.0}, "q_init"),
    ("runner", {"seed": -1}, "seed"),
    ("runner", {"seed": 4294967296}, "seed"),
    ("runner", {"jit": "false"}, "boolean"),
    ("runner", {"drain": 0}, "boolean"),
    ("runner", {"collect_trace": 1}, "boolean"),
    ("runner", {"output_dir": "  "}, "output_dir"),
    ("runner", {"output_dir": 12}, "output_dir"),
])
def test_invalid_startup_combinations(group, changes, message):
    cfg = ExperimentConfig(t_max=100000)
    cfg = replace(cfg, **{group: replace(getattr(cfg, group), **changes)})
    with pytest.raises(ValueError, match=message):
        cfg.validate()


@pytest.mark.parametrize("kind,reason", [("fixed_share_gain", "expert loss")])
def test_deferred_controllers_fail_clearly(kind, reason):
    cfg = ExperimentConfig(t_max=100000)
    cfg.controller.kind = kind
    with pytest.raises(NotImplementedError, match=reason):
        cfg.validate()


def test_unassumed_estimation_allowed_in_certified_mode():
    cfg = ExperimentConfig(t_max=100000)
    cfg.dependence.assumption = "unassumed"
    cfg.validate()  # Later phases use zero credit rather than rejecting this run.


def test_adaptive_meta_horizon_validation():
    with pytest.raises(ValueError, match="ceil"):
        ExperimentConfig(t_max=10).validate()


def test_reference_initialization_and_precision():
    cfg = ExperimentConfig(t_max=100000)
    state = init_runner_state(cfg)
    np.testing.assert_allclose(state.alpha_reference, 0.01)
    assert state.alpha_reference.dtype == jnp.float64
    assert state.history.set_size_grid.shape == (512, 10, 51)
    assert float(state.queued_issue_increment) == 0.0
    assert int(state.last_issued_time) == -1
    cfg.history.alpha_reference_init = (0.02,) * 10
    state = init_runner_state(cfg)
    np.testing.assert_allclose(state.alpha_reference, 0.02)
