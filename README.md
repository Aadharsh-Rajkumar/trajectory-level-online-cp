# Trajectory risk allocation

Research implementation of the specifications in `docs/online_trajectory_cp.pdf`
and `docs/bivariate_theorems.pdf`, with the user's subsequent decisions taking
precedence. **All seven phases are implemented:** configuration, numerical history,
adapter/runner-state interfaces, exact accounting/safety, four trajectory signals,
three dependence estimators, ACI/PID, three utility-weight modes, frozen forest
risk models, dense ADMM-backed SQP/RTI, and an end-to-end runner with seeded
synthetic experiments, streaming diagnostics, and delayed-observation draining.

## Quick start

### 1. Environment

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'   # installs JAX, Pyrallis, pytest
.venv/bin/python -m pip install matplotlib      # only needed for experiments/
```

If the package is not installed in editable mode, prefix commands with
`PYTHONPATH=src` (otherwise imports fail with `No module named 'trajectory_risk'`).

### 2. Run the tests

```sh
.venv/bin/python -m pytest -q                         # full suite (slow: several minutes, JIT compiles)
.venv/bin/python -m pytest -q tests/test_dtaci.py     # one module
.venv/bin/python -m pytest -q -x -k accounting        # stop at first failure, filter by name
.venv/bin/python -m pytest -q -W error                # treat warnings as errors (the project standard)
```

| Test file | What it protects |
|---|---|
| `test_history.py` | Ring buffer, no-lookahead reveal timing, critical-level conventions |
| `test_signals.py` | Every feedback signal telescopes exactly to beta - Z per trajectory |
| `test_dependence.py` | Estimators match brute-force NumPy; certificate radii; fixed-share updates |
| `test_controllers.py`, `test_dtaci.py` | ACI/PID/DtACI sign conventions, budget bounds, safety guard override |
| `test_tangent_guard.py` | Tangent safeguard: eqs. 24-25 against a reference, saturation, budget floor, eqs. 26-27 end to end |
| `test_utility_weights.py` | Weights positive, normalized, bounded, step-capped |
| `test_solver.py` | QP/SQP vs high-accuracy references, feasibility of every issued allocation |
| `test_end_to_end.py`, `test_layer_integration.py` | Exact accounting identity q_T - q_0 over full runs, JIT equivalence |
| `test_alpaca.py` | ALPaCA forecaster fit/predict shapes and posterior updates |

Run the full suite on an otherwise idle machine; running it next to experiments
roughly doubles both runtimes.

### 3. Run the 1-D real-data experiments

`experiments/run_1d.py` compares six methods built from the same runner, each a
config variant of `configs/default.yaml`:

All of them run under the tangent safeguard (`--safeguard tangent`, see below). The three
`TRA cert` controllers are also rerun with the hard thresholds for `5_safeguard.png`.

| Method label | What it is | Why it is there |
|---|---|---|
| `TRA cert + DtACI` | Dependence-aware allocation, only statistically certified overlap credit, DtACI budget | The method the theory covers |
| `TRA cert + ACI` / `TRA cert + PID` | Same allocation, fixed-gain ACI or PID (gains from `adaptive_window_pid.yaml`) budget | Isolates the effect of the outer controller |
| `TRA plug-in + DtACI` | Uses the point estimate of the overlap (`risk_mode=experimental_plugin`) | Upper bound on what overlap credit could buy; no validity guarantee |
| `Boole + DtACI` | Fixed empty forest, static weights: plain Bonferroni split of the budget | Isolates the value of the overlap term |
| `Boole + ACI (previous)` | Bonferroni with the fixed-gain ACI controller | The earlier system |

A method that crashes is skipped and recorded under `crashed` in `summary.json`.
A no-feedback `Static beta/H` baseline is scored from the same conformal p-values.

```sh
PYTHONPATH=src .venv/bin/python -u experiments/run_1d.py --datasets amd nvidia
PYTHONPATH=src .venv/bin/python -u experiments/run_1d.py --forecaster alpaca --datasets solar \
  --out results/my_run/solar
PYTHONPATH=src .venv/bin/python -u experiments/run_1d.py --datasets wind \
  --n_train 150 --window 150 --t_max 440
```

| Flag | Default | Effect / why |
|---|---|---|
| `--datasets` | `amd nvidia regime_shift drone_x` | Any of `amd nvidia amazon blackberry fanniemae regime_shift drone_x solar wind` |
| `--forecaster` | `ar` | `ar`: ridge AR(5), unit scale. `alpaca`: Bayesian meta-learned forecaster that also supplies a predictive std, so intervals scale with its uncertainty |
| `--t_max` | `2000` | Number of online forecast origins scored |
| `--n_train` | `500` | Points used to fit the forecaster (D1), then frozen |
| `--window` | `300` | Each horizon's interval is the conformal quantile of the last `window` revealed residuals. Smaller adapts faster but is noisier; alpha_min is raised to 1.2/(window+1) so quantiles stay finite |
| `--out` | timestamped `results/<time>_1d_<forecaster>/` | Fixed output folder; lets one process per dataset write in parallel |
| `--safeguard` | `tangent` | `tangent`: smooth budget correction (note section 2.5). `threshold`: hard override at q_lower/q_upper |
| `--tangent_K`, `--tangent_B` | `1`, `50` | Gain and constant envelope of eq. 24. B counts excess failures; one shock fails about H trajectories, so B=50 allows about five. Smaller B tightens calibration but reaches whole-space sets sooner |
| `--no_baseline` | off | Skip the hard-threshold reruns (and `5_safeguard.png`); about a third faster |

Series length must be at least `n_train + window + t_max + 20`; shorten
`--t_max`/`--window` for short series (as done for `wind`). Also keep
`H * alpha_min` below beta: alpha_min is about 1.2/(window+1), and with H=10 a window
of 100 makes the target budget infeasible, so nearly every round falls back to
whole-space intervals.

Per dataset the script writes (see `results/11_06-tangent_safeguard/README.md` for a reading guide):
- `1_overview.png`: rolling miscoverage, width, controller budget, fallback rate,
  accounting state q_t vs q_lower/q_upper, and the raw data (prices for stocks).
- `2_per_horizon_budget.png`: stacked per-step α_h over time per controller, and
  mean issued α_h vs realized per-step miss rate.
- `3_blowup_<i>_t<a>-<b>.png`: zooms on the two worst miscoverage windows (prices,
  tubes, per-origin and per-horizon failures, budget and α_h in the window).
- `4_overlap.png`: Boole charge vs overlap credit, claimed credit vs realized Boole
  slack, failure-dependence heatmap, credit-tree edges.
- `5_safeguard.png`: per controller, proposal vs issued budget, excess outcomes
  q_t/η against the envelope, and rolling miscoverage, tangent vs hard thresholds.
- `data.npz` (raw arrays) and `summary.json`.

`--scale` divides the series by its training-segment std (optional for
large-unit data). `experiments/summarize_suite.py results/11_06-tangent_safeguard` writes a
cross-dataset `SUMMARY.md` and `summary_all.png`.

`experiments/compare_forecasters.py RESULTS_AR RESULTS_ALPACA` overlays AR and
ALPaCA runs of the same datasets.

The full suite (nine datasets) is `experiments/run_suite.sh [out_dir]` and writes
`<out_dir>/<group>/<dataset>/`. A 2000-origin method takes under a minute, so the suite
needs about 40 minutes on 8 GB of RAM. It runs `JOBS=2` datasets at a time on purpose:
nine at once swapped so heavily that the same runs took about 2000 seconds per method.

**Data.** `solar` (Palo Alto NSRDB irradiance, hourly), `wind` and `amazon` are
bundled in `data/`. The stock CSVs and the `regime_shift`/`drone_x` generators
come from the earlier DtACI_ALPaCA project; point `DTACI_ALPACA_DIR` at a copy
of it.

### 4. Settings worth toggling

Any field of `configs/*.yaml` can be overridden on the
`python -m trajectory_risk` CLI (for example `--controller.kind dtaci`); in
`run_1d.py` they are set in `make_configs()`.

| Setting | Options | What it does / why |
|---|---|---|
| `beta` | (0, 1) | Target long-run trajectory miscoverage |
| `risk_mode` | `certified`, `experimental_plugin` | Subtract only certified lower bounds on pair overlap (valid), or point estimates (more credit, no guarantee) |
| `controller.kind` | `aci`, `dtaci`, `pid` | How the scalar trajectory budget b_t reacts to failures. `dtaci` runs `dtaci_gammas` experts on the critical budget and weights them (`dtaci_eta`, `dtaci_sigma`) |
| `controller.gamma`, `controller.budget_init` | | ACI step size and starting budget |
| `dependence.kind` | `full_history`, `fixed_window`, `adaptive` | How pairwise overlap is estimated: all history (stable dependence), one window, or fixed-share over `windows` (drifting dependence) |
| `dependence.delta` | (0, 1) | Confidence level of the overlap certificate. Larger delta gives smaller radii and more credit |
| `solver.forest_kind` | `maximum_spanning_tree`, `fixed` | Which horizon pairs receive credit; `fixed` with empty `fixed_forest` reduces to Bonferroni |
| `weights.kind` | `static`, `marginal_efficiency`, `failure_aware` | Where the budget is spent across horizons (log-utility weights) |
| `signal.kind` | `reveal_only`, `pending_valid`, `completed_window`, `learned` | Provisional label for pending trajectories; affects controller smoothness only, never the accounting |
| `safety.kind` | `threshold`, `tangent` | Hard override at q_lower/q_upper, or tangent supervision: `v = logit(b̃) + K tan(π/2 clip(q/(ηB), -c, c))`, issuing budget σ(v) inside (-L, L) and an extreme action outside. `run_1d.py` uses `tangent` |
| `safety.q_lower`, `safety.q_upper`, `safety.eta_q` | | Exact accounting gain and hard thresholds; outside them the runner issues whole-space (or empty) sets, which is what bounds calibration error |
| `safety.tangent_gain`, `tangent_limit`, `tangent_eps`, `tangent_envelope`, `tangent_envelope_power` | | K, L, ε_b and B_t = B·t^power of eq. 24. L must exceed log((1-ε)/ε). Power 0 gives O(1/T) calibration |
| `safety.tangent_budget_floor` | | Lowest ordinary budget, H·alpha_min in `run_1d.py`. Eq. 25 issues an ordinary allocation for all v > -L, but none exists below this budget; without the floor those rounds become whole-space sets |
| `history.horizon`, `history.alpha_min`, `history.alpha_max` | | H and the box for per-horizon alpha |
| `solver.mode` | `rti`, `full_sqp` | One warm-started SQP step per round, or iterate to convergence |

**Known limitation.** With H=10, a 51-point alpha grid and all 45 pairs, the
certificate radius exceeds realistic overlaps at a few thousand samples, so
certified credit is currently zero on every dataset; `TRA certified` then
behaves as a weighted Bonferroni allocation.

## Development

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
```

For the locked environment, use `uv sync --locked --extra test` and
`uv run --locked --extra test pytest -q -W error`. `uv.lock` records resolved
dependency versions, including Python-version-specific variants.

```python
from trajectory_risk.configs import load_config
from trajectory_risk.runner import init_runner_state

cfg = load_config("configs/default.yaml")
state = init_runner_state(cfg)  # Configures JAX before creating arrays.
```

`parse_config()` supports Pyrallis command-line overrides, for example
`--config_path configs/default.yaml --history.capacity 1024`. Physical `t_max`
is required. `risk_mode` is a value-serialized enum: `certified` or
`experimental_plugin`. Unsupported controller choices fail explicitly at startup.
`build_signal`, `build_accounting_guard`, `build_dependence_estimator`,
`build_controller`, and `build_weight_updater` select components before compilation.
`build_risk_model` and `build_solver` select risk and optimization kernels;
the module CLI runs the synthetic experiment described below.

## Run a reproducible experiment (Phase 7)

```sh
.venv/bin/python -m trajectory_risk --config_path configs/synthetic_demo.yaml
.venv/bin/python -m trajectory_risk --config_path configs/synthetic_demo.yaml \
  --runner.seed 42 --runner.output_dir results/seed42
```

The small demo issues 64 origins and processes three tail rounds. The standard
algorithm configurations retain the approved `t_max: 100000`, H=10 and default
window sizes. All algorithm choices and solver controls use the same CLI:

| YAML | Dependence | Controller | Trajectory signal | Utility weights |
|---|---|---|---|---|
| `default.yaml` | Adaptive windows | ACI | Completed-window probability | Failure-aware efficiency |
| `full_history_aci.yaml` | Full history | ACI | Reveal-only | Static |
| `fixed_window_aci.yaml` | Fixed window | ACI | Pending-valid | Marginal efficiency |
| `adaptive_window_aci.yaml` | Adaptive windows | ACI | Completed-window probability | Failure-aware efficiency |
| `adaptive_window_pid.yaml` | Adaptive windows | PID | Learned probability | Failure-aware efficiency |

`examples.SyntheticIntervalAdapter` is a numerical geometry fixture: IID
Y_t ~ Uniform[-1,1], ordinary sets [-(1-a),1-a], sizes 2(1-a), and critical
levels 1-|Y_t|. It uses one draw per physical time, shared across the horizons
revealed that round. Different horizons of one origin see independent future
draws, giving joint ordinary-event probability a*b. The fixture supplies
whole-space sets for conservative actions and an empty horizon-1 set for
aggressive actions. Their actual labels remain separate from counterfactual
ordinary critical levels. There is no trained forecasting model. Supplied
adversarial outcomes are useful for testing geometry and timing, and do not
inherit the IID distribution assumption.

For an application, implement `TrajectoryAdapter.observe`, `prepare`, and
`issue` and pass it to the same runner:

```python
from trajectory_risk.runner import OnlineRunner

runner = OnlineRunner(cfg, adapter)
result = runner.run()
print(result.summary)
# Or advance consecutive rounds interactively with runner.step().
```

Each physical round requires exactly one actual membership for every due issued
origin/horizon. Critical levels may be explicitly unavailable. Feedback batches
may reorder slots; missing or duplicate actual observations reject the round.
Failed trajectories continue supplying later marginal and pairwise observations.
The numerical round resolves history, updates exact accounting and signals,
consumes the previous issue queue, updates the controller once, updates dependence
and current utility, applies the guard, freezes the forest, and solves. A separate
acceptance check and metadata validation run before `adapter.issue`. The host
commits the returned immutable state after issuance succeeds, so the new
provisional signal affects only the next budget. Any callback/numerical failure
aborts that runner, preventing automatic duplicate issuance after an uncertain
external side effect. Invalid caller times are rejected before callbacks.

`build_round_kernels(cfg, history)` exposes pure `step(state, observations,
current, t)` and `drain(state, observations, t)` functions for JIT or `lax.scan`.
They return a candidate state and diagnostics with an error code; check acceptance
before committing a state or issuing its action. Python geometry callbacks never
run inside compilation. Scalar parameters and algorithm callables are selected
at startup; all numerical states are fixed-shape PyTrees.

`runner.drain: true` processes all H rounds after issuance stops, including later
criticals from origins whose actual failure was already known. Drain rounds update
history, accounting, signals, the controller and dependence; they neither prepare
new geometry nor update utility or run allocation. `runner.drain: false` reports
pending trajectories and a failure-rate interval, leaving the completed-cohort
failure rate and calibration error undefined. `runner.jit` selects compilation;
`runner.seed` selects the synthetic stream. The seed uses physical-time fold-in,
so extending a run preserves its outcome prefix.

With `runner.output_dir`, the runner writes resolved `config.yaml`, one compact
record per physical round to `trace.jsonl`, and `summary.json`. Existing experiment
artifacts are refused. Traces include actual horizon feedback and critical masks,
per-origin atomic signal packets, queued/controller increments, q, budgets,
weights/utility readiness, action and reference allocation, forest masks,
edge certification/missing counts, adaptive expert weights and regret eligibility,
and solver feasibility, model/acceptance risk, convergence and restoration.
Edge arrays use lexicographic unordered horizon pairs. Extreme/drain alphas and
undefined numerical diagnostics serialize as JSON null, never fabricated
ordinary vectors or nonstandard NaN literals. Valid infinite critical endpoints
serialize as "+inf"/"-inf", keeping them distinct from missing data. Default
logging streams with bounded
memory; `runner.collect_trace: true` additionally retains records in the returned
`ExperimentResult.trace`.

Summaries report the resolved-label accounting identity and, after full drain,
the per-origin signal identity. The queue identity applies even without drain.
The accounting bound uses |q-q_init|/(eta_q*N)+H/N. The reported set size is the
sum of interpolated current ordinary-family size curves, averaged only over
ordinary origins with all curves available; no finite size is invented for
whole-space actions. `certified` describes the acceptance policy under the
stated marginal and stable conditional-law assumptions. Experimental allocations
are always marked non-certified; neither model KKT convergence nor empirical
calibration supplies a global optimization or coverage theorem.

## Phase-2 feedback use

`init_runner_state` initializes independent accounting and signal PyTrees.
`build_feedback_kernels(cfg)` returns concrete `observe` and `issue` callables;
compile them with `jax.jit`, or compose them into `jax.lax.scan`.

At each physical round, starting at time 0:

1. Write the newly available actual horizon memberships and optional ordinary
   critical levels to history, then resolve actual trajectory labels.
2. Call `feedback.observe(state, t)` and check its error with `require_accepted`.
   Its `pre_issuance_increment` sums terminal corrections, pending refresh
   corrections, and the previous queued issue contribution. That queue is cleared
   once. The exact accounting state reads newly resolved history labels directly.
3. Supply the aggregate to the ordinary controller once for this physical round;
   `ExactAccountingGuard.protect` overrides its proposal at q's safety thresholds.
   A returned extreme action must actually be issued by the adapter.
4. After issuing and recording the current origin with `record_issuance`, call
   `feedback.issue(state, t)`. Its atomic packet is diagnostic feedback; its
   `pre_issuance_increment` is zero. The new contribution is queued in RunnerState
   for time t+1 and cannot affect the current allocation.
5. After issuance ends, keep calling `observe` on consecutive physical rounds
   while outstanding actual labels resolve, without issuing new origins.

Each operation returns a new state. Duplicate/skipped physical rounds and unsafe
overwrites are rejected without changing state or consuming the queue.
Fixed-size `SignalPacket`s carry signed increments, origin IDs, event kinds, and
resolved-label diagnostics. Zero corrections have inactive event masks, while
resolved-label masks still record the terminal event. The q update never reads
these signal packets.

The ordinary-round signals are reveal-only (no issue contribution), pending-valid
(+beta at issue and -1 on failure), completed-window probability, and a small
logistic model. Completed-window feedback uses the configured prior (default beta)
until its full origin cohort exists. Both probability signals support
`refresh_pending=False` as an ablation.

The logistic model initializes its intercept from the prior; a finite logistic
initialization requires a prior strictly between 0 and 1. Its frozen features are
`[1, issued_alpha, issue_covariates]`. It performs one mean-batch SGD step on the
newly resolved ordinary examples each round, using Bernoulli cross-entropy and
the configured learning rate. Terminal corrections use the last booked
probabilities before training; only trajectories still pending are refreshed
afterward. Nonfinite model updates/predictions are rejected rather than booked.

On extreme rounds, every signal uses the deterministic actual provisional label
(0 conservative, 1 aggressive), overriding its ordinary-round rule. These origins
have zero stored model features, no logistic training, and no pending refresh.
Their actual terminal correction is zero; their exact q contribution still waits
for the prescribed reveal time. Ordinary-family critical levels remain separate.
Consequently, each origin's complete atomic contribution remains beta-Z even
when provisional models are inaccurate or size/critical data are unavailable.

## Dependence estimation (Phase 3)

`build_dependence_estimator(cfg.dependence, cfg.t_max)` constructs a full-history,
fixed-window, or adaptive-window estimator. Initialize with `init_state(history)`;
all unordered horizon pairs are the default candidates. An optional explicit edge
array can restrict candidates; endpoints are 1-based and must be distinct pairs.

Call `update(state, history, t)` after observing current horizons and BEFORE
recording origin t. Check its error and retain its returned state. Then query
`estimate(state, history, t)` at that same time. Duplicate/skipped updates and
stale estimate queries are rejected. Updating a window after eviction of its
oldest outgoing origin is rejected rather than silently using the wrong window.
Drain updates are supported through `t_max + H - 1`; no post-`t_max` forecasts
are registered for adaptive learning.

- Full history maintains lifetime CDF counts/sample counts independently of ring
  eviction. Its stable radius is the main paper's anytime formula with global
  complexity `C = M**2 * sum(d_e)`.
- Fixed/adaptive windows update rolling CDF counts by adding the newly revealed
  pair and subtracting the exact outgoing origin. Their finite-horizon radii use
  `log(C * t_max**2 / delta)`. Adaptive mixtures weight expert radii on the single
  interval-uniform event, without an additional `log(J)` confidence factor.
- Adaptive predictions use all configured experts, even in diagnostic burn-in;
  there is no renormalization over a subset of windows. Learning starts only with
  the complete largest window for that edge. `(H+1,E,J,M,M)` pending snapshots
  retain the predictions and mixture weights made at each origin.
- At reveal, integrated Brier losses score those snapshots. Only that origin's
  residue modulo `d_e` changes. The multiplicative step is followed by the exact
  fixed-share step with `sigma_e = 1/ceil(t_max/d_e)`. A common normalization
  afterward prevents underflow and leaves the note's predictions unchanged by
  homogeneity. It is not an EMA or a generic replacement expert algorithm.
- Missing pairs remain explicit: no older-origin backfill and no invented Brier
  loss. Full-history certification stays unavailable after any required pair is
  missing. Window certificates can recover once the exact current window is
  complete. Missing adaptive predictions/outcomes make `adaptive_regret_valid`
  false; subsequent usable losses may update the learner, but the complete-data
  regret theorem is not claimed for that stream.

Estimates separate `point_surface`, `lower_surface`, `radius`, `sample_count`,
`ready`, and `certified`, plus expert/loss and missingness diagnostics. A finite
reported radius alone is not a certificate: both readiness and certification
flags must be checked. V1 certifies only under configured conditional stability;
`unassumed` surfaces are point/tracking estimates. Window certification is limited
to pre-`t_max` physical times. Fixed-window readiness requires W available pairs;
adaptive readiness requires all dictionary windows to be complete.

`build_overlap_credit_selector(cfg.risk_mode)` provides the allocation boundary.
Certified mode uses only ready certified lower surfaces and emits an edge-specific
fallback mask for zero overlap/Boole credit. Experimental mode uses point surfaces
and marks the credit non-certified, even if an optional lower certificate exists.
The credit's certification flag does not certify an arbitrary adapter's margins.

## Controllers and utility weights (Phase 4)

ACI updates `clip(b + gamma * g_pre, 0, 1)` once per round. PID uses
`clip(b0 + kp*e + ki*I + kd*(e-e_previous), 0, 1)`, with an optional symmetric
`integral_limit`. Empty-feedback rounds still advance PID's derivative clock.
PID is a generic budget adaptation, not the original conformal-score PID method.
Both controllers reject duplicate/skipped rounds and nonfinite feedback. DtACI runs
its expert grid on the scalar budget with feedback delayed by H, scoring each origin
against the budget actually issued to it. Fixed-share gain classes/factories raise
the agreed informative `NotImplementedError`.

Weight updaters receive current predictable ordinary-family curves, their
availability mask, and `alpha_reference`. Static weights normalize application
importance (uniform by default). Efficiency targets are `c * alpha_reference * m`.
Failure-aware targets divide `(efficiency_target + eps_efficiency)` by
`(eps_failure + first_failure_mass)`, with masses from complete origin cohorts.

The numerical ordering is **raw target → L1 normalization → bounded-simplex
Euclidean projection → EMA → radial step cap**. Normalizing first (writeup eq. 17)
removes the data's units, which otherwise broke sum(w)=1 through float rounding.
The scalar-dual projection uses fixed-iteration JAX bisection. Convex segment
updates preserve positive normalized bounded weights, and the configured L2 cap
limits each step. If any current curve is unavailable, or the failure cohort is
not yet complete, use the static application target with the same smoothing/cap;
missing curve data is not interpreted as zero marginal value for that horizon.

## Combined preallocation layers

`build_preallocation_kernel(cfg)` selects concrete methods before compilation.
Use its returned callable in place of a separate `feedback.observe` call:

```python
import jax
from trajectory_risk.runner import build_preallocation_kernel, require_accepted

before = jax.jit(build_preallocation_kernel(cfg))
result = before(state, t, current_forecast_inputs)
require_accepted(result)
state = result.state
```

It performs exact accounting and signal correction/training/refresh, one aggregate
controller update, dependence update/snapshot, current utility adaptation, ordinary
proposal, and the exact safety override. Any rejected layer preserves the whole
input state and issue queue. Diagnostics from a rejected round must not be used
for allocation. After actual issuance and `record_issuance`, use `feedback.issue`
to queue the new origin's contribution for the next round.

The helper returns the dependence estimate, utility diagnostics, ordinary
proposal, and explicit safety action. Follow it with the allocation kernel below.
This helper does not call a forecasting adapter. The complete runner enforces
the issuance cutoff at `t_max`; drain-only rounds process pending observations.

## Frozen forests and risk evaluators (Phase 5)

`build_risk_model(cfg.solver, cfg.risk_mode, history)` returns a freeze kernel.
Pass the updated dependence state/estimate and last ordinary `alpha_reference`.
The default Kruskal routine selects a maximum-weight spanning forest using exact
certified grid credit at that reference, with lexicographic horizon-pair ties.
All-pair candidates give a spanning tree; sparse graphs give a forest. Configured
fixed forests, including the empty Boole forest, are supported. The returned
immutable `FrozenRiskModel` never reselects edges when evaluating trial alphas.

The two numerical representations remain explicit:

- `risk_model_for_sqp(alpha)` uses bilinear interpolation. Certified mode
  interpolates the usable lower-credit grid, including southwest prefix maxima;
  experimental mode interpolates the point grid. Interpolation itself is never
  marked as a certificate. Derivatives are deterministic one-sided derivatives
  at grid endpoints and may change at cell boundaries.
- `certified_risk(alpha)` uses the exact monotone lower-grid extension: maximum
  certified lower credit over southwest grid corners, zero below either first
  grid coordinate. Its edge credits are step functions, so its total risk can
  have downward jumps and is not generally monotone along an allocation segment.
- `acceptance_risk(alpha)` selects certified risk in certified mode and the
  interpolated point risk in experimental mode. Unready/uncertified edges provide
  zero certified overlap credit, with explicit fallback masks.

Both representations use the same frozen forest and marginal charge `sum(alpha)`.
Invalid frozen inputs have an error flag and infinite evaluated risk. The
certification flag is conditional on the external adapter's marginal conformal
contract; it does not certify arbitrary forecasting geometry.

## Dense QP and SQP/RTI (Phase 6)

`build_solver(cfg.solver)` returns `SQPSolver`. Initialize with
`init_state(history, alpha_reference)` and call
`solve(state, budget, weights, frozen_model)` to obtain `(new_state, result)`.
The state retains the primal allocation, risk/box multipliers, QP primal,
auxiliary variable, original-unit duals, penalty, and last diagnostics.

SQP uses JAX autodifferentiation for objective gradients/Hessians and model-risk
Jacobians, with `B = Hessian(-sum(w*log(alpha))) + mu*I`. The small dense QP has
one linearized risk constraint plus allocation boxes. ADMM scales variables and
constraint rows, retains warm starts in original units, caches its Cholesky
factor until the penalty changes, and checks original-unit consensus and
stationarity residuals as well as inequality feasibility. It is independently
implemented using generic patterns studied in
[GPU-SLS SQP](https://github.com/trustworthyrobotics/gpu_sls/blob/main/src/gpu_sls/gpu_sqp.py)
and [ADMM](https://github.com/trustworthyrobotics/gpu_sls/blob/main/src/gpu_sls/gpu_admm.py);
no SLS dynamics/control source is copied. SciPy is used only for offline references.

`full_sqp` runs up to `max_sqp_iterations` and stops early on model KKT/relative-step
convergence. `rti` performs the configured fixed number of iterations, default 1,
with retained primal/dual/QP state. Set a larger iteration cap explicitly when
switching the representative default YAML to full SQP. Merit/Armijo backtracking
uses the derivative model, and all iterations have fixed array shapes.

Every proposed issued point is independently checked against the acceptance
evaluator and exact allocation boxes. Restoration samples segments from the
best known accepted point or minimum allocation, checks every candidate, and may
refine a tested local feasible/infeasible bracket. It never infers global
feasibility from monotonicity. The best tested feasible utility is retained;
for utility ties within `(1 + abs(risk_dual))*feasibility_tol`, an accepted
model-converged iterate is preferred over an older nonstationary point. The
final selected point is checked again. No global optimizer guarantee is claimed.

`SolverResult` separates model risk, acceptance risk, feasibility, certification,
model KKT residual, convergence, restoration, SQP/QP iteration counts, and required
action. `objective` is the minimization cost; `utility` is its negative. A finite
feasible point can be issued even when numerical convergence is false. A failed
solve preserves the previous warm state, returns NaN alpha entries, and requests
the conservative extreme. `NO_ACCEPTED_ALLOCATION` does not prove global problem
infeasibility. Experimental results always have `certified=False`.

The KKT residual is **for the interpolated model at the returned allocation**:
gradient-scaled stationarity, primal/dual feasibility, and multiplier-scaled
complementarity. It is not a global optimization gap or stationarity certificate
for the discontinuous acceptance problem. The original smooth certified SQP
regret theorem is not automatically inherited by this model/acceptance separation.

Useful YAML controls include `qp_max_iterations`, `qp_abs_tol`, `qp_rel_tol`,
`qp_initial_penalty`, penalty limits/update interval, proximal regularization,
`line_search_steps`, `restoration_steps`, `armijo`, `backtrack_factor`, and
`merit_penalty`. `feasibility_tol` is the explicit acceptance allowance above the
budget; float64 offline-reference tests use tighter tolerances than the example.

## Allocation orchestration before actual issuance

`build_allocation_kernel(cfg, history)` consumes an accepted `PreAllocationUpdate`,
freezes the risk model, and optimizes only ordinary rounds. Formal q safety
extremes bypass the solver entirely, including the aggressive extreme; solver
failure never replaces that override with a conservative action. A failed
ordinary solve requests the conservative extreme instead of an unchecked alpha.

```python
from trajectory_risk.runner import build_allocation_kernel

allocate = jax.jit(build_allocation_kernel(cfg, state.history))
allocation = allocate(result)  # result from the preallocation kernel above
require_accepted(allocation)
state = allocation.state
```

`AllocationUpdate.accepted` means an action is ready, which may be an extreme.
Inspect `action_mode` before using `result.alpha`. The adapter receives `None`
for extreme alphas. Solver state may update on an accepted ordinary proposal,
but `alpha_reference`, origin history, and issue feedback remain unchanged until
actual issuance/recording. `OnlineRunner` performs those adapter calls and
enforces the physical issuance horizon.

## Numerical boundary and history conventions

- Origins/time start at 0. Horizons and configured edge endpoints start at 1.
  One origin is issued per physical round, with origin IDs increasing by one.
- A horizon is processed at exactly `origin_id + horizon`. Use an explicit missing
  mask when a critical level is unavailable at that time. Actual set membership
  is still required. Late counterfactual backfilling is not supported in v1.
- Failure means `critical_level < alpha`; equality is containment. Finite critical
  levels and +/- infinity are supported. NaN denotes unavailable critical data.
- `horizon_violation` describes actual issued sets. `critical_level` describes
  the ordinary conformal family, including counterfactual levels on extreme rounds.
  They are checked for consistency only on ordinary rounds.
- Extreme origins store their action mode and NaN alpha entries. The Python
  issuance adapter receives `alpha=None` for extremes. `alpha_reference` retains
  the last ordinary issued allocation; initially it is `clip(beta/H)` per horizon.
- `trajectory_label=-1` is pending, 0 is success, and 1 is failure.
  `first_failure=H` encodes both success and failure at the final horizon;
  failure histograms always mask by `trajectory_label == 1`.
- A failed origin remains in history for later marginal/dependence observations.
  Ring slots can be reused only after all actual horizons have been processed.
- Completed-window queries use fixed origin cohorts at least H steps old. Missing
  origin/critical entries retain their position and are never replaced by older ones.
  Lifetime missing-critical counts survive ring eviction; edge queries also expose
  the missing pattern in the requested cohort.
- Set-size arrays describe the ordinary family and are numeric, nonnegative,
  nonincreasing curves. The library is metric-agnostic. Current predictable curves
  are passed separately for efficiency evaluation before the next solve.
- Pure updates return immutable result records with an `UpdateError` code.
  Rejected updates leave state unchanged under eager execution and JIT. The host
  must check results with `require_accepted` where success is expected.
- Query window lengths are static when compiled, for example
  `jax.jit(recent_completed_origins, static_argnames=("window",))`.

The adapter owns marginal conformal assumptions; this package does not claim
marginal validity for arbitrary external forecasting geometry.

## Approved implementation phases

The [revised implementation plan](docs/implementation_plan.md) lists files,
classes, invariants, tests, dependencies, and advancement gates for each phase.

| Phase | Components | Required gate |
|---|---|---|
| 1 | Configuration, immutable history, adapter inputs, reference allocation and queue state | Timing, ties, cohorts, wraparound, startup validation, JIT equivalence |
| 2 | Exact accounting and four predict-and-correct signals | Per-origin telescoping; logistic terminal corrections before training and refresh |
| 3 | Three dependence estimators and edge-specific residue fixed-share learning | Numerical references, exact delayed updates, burn-in and missingness/certification masks |
| 4 | Aggregate physical-round ACI/PID and three utility updaters | One update per round; bounded normalized weights with EMA and step cap |
| 5 | Deterministic forests and separate model/acceptance risk | Frozen tree, monotone grid certificate, optimistic-model rejection |
| 6 | Dense ADMM, full SQP, configurable RTI and restoration | Reference QP/NLP checks; issued acceptance feasibility and explicit failure flags |
| 7 | Runner, synthetic numerical example, configurations and diagnostics | No lookahead, queued issue feedback, accounting identities and reproducibility |

Fixed-share gain aggregation is deferred, with an informative controller stub
available. Dependency roles: JAX and
Pyrallis/YAML at runtime; pytest/NumPy for tests; SciPy and optional JAXopt for
offline references. The existing PDFs are left unchanged.

The optimization architecture distinguishes `model_risk`, `acceptance_risk`,
and `certified`. In certified mode acceptance uses the monotone grid certificate,
with zero overlap credit for unready or uncertified edges. In experimental mode
point/interpolated acceptance is allowed, and every solver result is marked
non-certified. Interpolation is never itself presented as a certificate.
