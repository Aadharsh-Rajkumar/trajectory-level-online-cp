# Revised version-1 implementation plan

The user's decisions following the document review supersede conflicting
appendix passages. All seven phases are authorized and implemented, retaining the
approved architecture. Existing PDFs and their sources stay unchanged.

## Phase 1: configuration, interfaces, and history

- **Files/classes:** `pyproject.toml`, `uv.lock`, `.gitignore`, `README.md`,
  `configs/default.yaml`; `__init__.py`, `configs.py` (nine configuration
  dataclasses, `RiskMode`), `history.py` (`ActionMode`, `AlphaGrid`, `HistoryState`,
  query/update records), `runner.py` (adapter protocol, predictable input records,
  `RunnerState`, `RunnerUpdate`).
- **Invariants:** Strict critical-level comparisons; chronological horizon
  observations at k+h; fixed origin cohorts; distinct actual outcomes and
  ordinary-family critical levels; explicit missing masks/counts; safe ring
  reuse; NaN alpha storage for extremes; last ordinary `alpha_reference` retained
  across extremes; explicit queued issue feedback in runner state. `t_max` is
  required at experiment level. Risk mode is one enum, not competing booleans.
- **Tests:** `test_configs.py`, `test_history.py`, float64 setup in `conftest.py`:
  startup validation, YAML/CLI parsing, ties, reveal timing, wraparound,
  missingness, extreme outcomes, failure histograms, current-curve efficiency,
  eager/JIT equivalence, independent batches, and compiled stream updates.
- **Dependencies:** JAX and Pyrallis/YAML at runtime; pytest/NumPy for tests.
- **Gate:** All Phase-1 tests pass with warnings treated as errors. Numerical
  states and queries compose under JIT, vmap where appropriate, and lax.scan.

## Phase 2: accounting and four trajectory signals

Implemented, including pure feedback orchestration and an explicit issue queue.
Logistic fitting uses one mean-batch cross-entropy SGD step per round; no prefix
features or extreme-action training examples are added. Regression coverage
includes model-error rejection, CPU float32/float64, compiled streaming, and
independent batch execution. Ordinary budget controllers remain in Phase 4.

- **Files/classes:** `signals.py`: `TrajectorySignal`, `RevealOnlySignal`,
  `PendingValidSignal`, `CompletedWindowProbabilitySignal`,
  `LearnedProbabilitySignal`, fixed-size per-origin event packets and signal
  states. Begin `controllers.py` with `ExactAccountingGuard`.
- **Invariants:** Only newly resolved actual Z labels update the unprojected q
  state. Every trajectory's atomic signal contributions telescope to beta-Z.
  Completed-window estimates use full origin cohorts and prior beta during
  burn-in. Logistic features are frozen at ordinary issuance. Terminal
  corrections use the stored pre-training provisional values; remove resolved
  trajectories, train, then refresh remaining pending ordinary trajectories.
  Extremes use deterministic provisional labels, without fabricated learned
  alpha features or logistic training examples. Actual labels still resolve at
  their prescribed physical times. Issue contributions remain queued separately
  in `RunnerState`.
- **Tests:** `test_signals.py`, accounting tests in `test_controllers.py`: all
  signals telescope; refresh on/off; simultaneous training and resolution;
  no-lookahead; actual extreme labels independent of counterfactual levels.
- **Dependencies:** None added.
- **Gate:** Per-origin telescoping and exact resolved-label accounting hold for
  arbitrary provisional model errors and after ring wraparound.

## Phase 3: dependence estimators

Implemented. Lifetime counts and rolling window updates match brute-force NumPy
references; adaptive learning matches the exact residue-wise fixed-share update.
Certification and missingness gates, post-selection radii, explicit credit-mode
selection, float32/float64, and compiled streaming are covered by regression tests.

- **Files/classes:** `dependence.py`: `PairwiseDependenceEstimator`,
  `FullHistoryDependenceEstimator`, `FixedWindowDependenceEstimator`,
  `AdaptiveWindowDependenceEstimator`, `DependenceEstimate`, numerical states.
- **Invariants:** Lifetime counts for full history; fixed-origin windows with no
  backfilling; strict lower-orthant surfaces. Store expert predictions made at
  each origin; score them only after reveal using integrated Brier loss. Update
  only the appropriate residue with the exact multiplicative/fixed-share rule.
  Derive n_max,e=ceil(t_max/d_e) and sigma_e=1/n_max,e. Adaptive burn-in requires
  all dictionary windows; no dynamic expert renormalization. Stable certification
  uses the estimator's justified global confidence event. Missing required pairs
  invalidate the affected certificate and never invent outcomes or expert losses.
  Available-observation point estimates remain explicitly diagnostic/non-certified.
- **Tests:** `test_dependence.py`: brute-force references, confidence radii,
  strict ties, residue isolation, prediction snapshots, exact fixed-share updates,
  burn-in, missing patterns/counts, and eager/JIT equivalence.
- **Dependencies:** None added.
- **Gate:** Every estimator matches its reference. Certification, readiness, and
  point estimation are distinct; missing-data updates cannot claim the complete
  observation theorem. No adaptive-regret theorem is claimed for fabricated losses.

## Phase 4: controllers and utility weights

Implemented. ACI/PID consume aggregate physical-round signals, both deferred
controller interfaces fail explicitly, and all utility modes follow the approved
raw-projection/EMA/step-cap ordering. A combined preallocation helper composes
Phases 2–4 transactionally; allocation and full adapter orchestration remain later
phases. Integration tests cover accounting, queue consumption, safety overrides,
and reference preservation across extreme actions.

- **Files/classes:** `controllers.py`: `OuterBudgetController`, `ACIController`,
  `PIDBudgetController`, `ExactAccountingGuard`, and informative
  `DtACIController`/`FixedShareGainController` stubs. `utility_weights.py`:
  `UtilityWeightUpdater`, `StaticWeightUpdater`,
  `MarginalEfficiencyWeightUpdater`, `FailureAwareEfficiencyWeightUpdater`.
- **Invariants:** Exactly one controller update per physical round using terminal
  corrections + pending revisions + previous queued issue contribution. Project
  ACI once after aggregation; PID receives zero on empty-feedback rounds.
  Utility order: raw target, bounded-simplex projection, EMA, radial step cap.
  Use current predictable size curves around alpha_reference, uniform default
  application weights, and complete cohorts for failure-aware statistics.
  Failure-aware burn-in uses static weights; efficiency-only adaptation may
  activate earlier with available current curves.
- **Tests:** `test_controllers.py`, `test_utility_weights.py`: signs, permutation
  independence near budget limits, update counts, queued timing, deferred choices,
  weight positivity/normalization/bounds/step caps, and JIT equivalence.
- **Dependencies:** None added.
- **Gate:** Controller timing and all weight constraints pass. Gain aggregation
  stays deferred until a counterfactual expert loss is defined; literal DtACI
  stays deferred until scalar nestedness and delayed critical-budget feedback exist.

## Phase 5: forests and model/acceptance risk

Implemented: immutable risk inputs, deterministic Kruskal/fixed forests,
bilinear derivative models, exact southwest-prefix certificate evaluation,
and mode-specific acceptance. Tests cover grid boundaries, optimistic-model
rejection, invalid input flags, sparse/empty forests, and JIT equivalence.

- **Files/classes:** Begin `solver.py`: frozen risk inputs, forest selection,
  derivative model and acceptance evaluators.
- **Invariants:** All unordered horizon pairs by default; maximum-weight spanning
  tree evaluated using available certified credit at alpha_reference;
  lexicographic ties; configured fixed forests for ablations. Freeze the tree.
  Bilinear interpolation initially supplies SQP derivatives and is not certified.
  Certified acceptance uses the exact monotone lower-grid extension, with zero
  credit for unready/uncertified edges. Experimental acceptance may use the
  plug-in/interpolated risk and always reports certified=False. Expose model_risk,
  acceptance_risk, and certified independently. The acceptance risk can be
  discontinuous; smooth certified SQP regret assumptions are not inferred from
  this architecture.
- **Tests:** Risk/forest cases in `test_solver.py`: deterministic selection, frozen
  trees, grid boundaries, zero-credit fallback, and optimistic-model rejection.
- **Dependencies:** None added; no graph-library requirement.
- **Gate:** Model feasibility never masquerades as certified acceptance; modes
  preserve explicit diagnostics.

## Phase 6: dense ADMM, SQP/RTI, and restoration

Implemented: scaled dense ADMM with original-unit warm starts/duals, objective
autodifferentiation, Armijo/model merit search, fixed-count RTI, full SQP,
explicit acceptance restoration, and unambiguous solver statuses. The allocation
wrapper preserves formal safety overrides and last usable state on failure.
Tests compare QP/NLP output with high-accuracy offline references and cover
model optimism, disconnected feasible segments, failed restoration, warm/cold
starts, RTI tracking, float32, JIT, scan, and independent batches.

- **Files/classes:** Complete `solver.py`: QP/SQP immutable states and configs,
  dense ADMM, `SQPSolver`, `SolverState`, `SolverResult`.
- **Invariants:** JAX autodifferentiation, regularized objective Hessian, retained
  primal/dual/QP warm starts, fixed-shape iterations, merit backtracking. RTI
  defaults to one iteration with configurable fixed count. Evaluate every issued
  candidate against acceptance_risk. Restoration tests candidates explicitly;
  do not assume monotonic risk along a segment. An infeasible minimum allocation
  or failed restoration returns a failure requiring conservative extreme issuance.
  Model feasibility, acceptance feasibility, convergence, and certification stay
  distinct. Experimental rounds carry no theorem-level coverage claim.
- **Tests:** `test_solver.py`: QP and offline NLP references, derivatives, cold/warm
  starts, RTI tracking, discontinuous acceptance, infeasible budgets, and JIT equality.
- **Dependencies:** SciPy and optional JAXopt for offline reference tests; no new
  direct dependency in online numerical kernels.
- **Gate:** Every certified-feasible result passes certified acceptance numerically.
  Failures cannot issue unchecked ordinary allocations. Global optimality is not
  inferred solely from small KKT residuals.

## Phase 7: runner and reproducible experiments

Implemented: pure transactional round/drain kernels, host adapter orchestration,
mandatory observation completeness, an independent final acceptance check,
seeded IID interval geometry, all five algorithm YAMLs and a small demo, strict
streaming JSON artifacts and resolved configuration snapshots. Integration tests
cover all estimator/signal combinations, both controllers and all utility modes,
per-origin telescoping, exact accounting, future-outcome isolation, extreme and
solver-failure actions, missing critical data, experimental labeling, callback
failure handling, eager/JIT equivalence, and complete `lax.scan` execution.


- **Files/classes:** `runner.py` (`RoundKernels`, `OnlineRunner`, diagnostics/totals and CLI),
  `examples.py`, `__main__.py`, all five planned YAML configurations,
  a synthetic numerical example without a specific forecasting model, README and
  attribution notices where needed. Extend `RunnerState` with component PyTrees.
- **Invariants:** Observe actual memberships, update history and q, terminal signal
  corrections, logistic training and pending refresh, consume prior issue queue,
  one controller update, dependence and utility updates, proposal and exact guard,
  frozen risk inputs, solve/check/restore, issue and store, then queue new issue
  feedback. Update alpha_reference only on ordinary issuance. Process available
  counterfactual critical levels independently of actual extreme outcomes.
  Stop issuance at t_max and optionally drain pending labels without new issuance.
- **Tests:** `test_end_to_end.py` and completed layer suites: no-lookahead,
  accounting and signal identities, safety boundaries, missingness/fallback,
  experimental labeling, algorithm changes by YAML, and reproducibility.
- **Dependencies:** None beyond prior phases.
- **Gate:** The float64 CPU suite passes. Runtime flags expose mode, certification,
  fallback, missingness, actual action, solver convergence and acceptance.
  One YAML selects each implemented algorithm combination without experiment-code
  changes. No complete forecasting package or unsupported validity claim is added.
