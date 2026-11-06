# 11/06 suite: how to read these results

Nine datasets, AR(5) forecaster, β = 0.10, H = 10, 2000 online origins (wind: 440).
This repeats the 10/30 suite with the tangent safeguard from section 2.5 of the note.
`SUMMARY.md` and `summary_all.png` compare every dataset and method at a glance.

```
stocks/     amd, nvidia, amazon, blackberry, fanniemae   (daily log returns)
robotics/   drone_x                                     (drone x-position, dynamics shift at t=1000)
synthetic/  regime_shift                                (variance 4.0 <-> 0.3 every 1000 steps)
energy/     solar (Palo Alto hourly irradiance), wind
```

## What changed since 10/30

**Safeguard.** The hard override at q = ±1 is replaced by eqs. 24-25:

    v_t = logit(b̃_t) + K tan(π/2 · clip(q_t / (η B), -c, c))
    b_t = σ(v_t) for -L < v_t < L;  whole-space sets at v_t <= -L;  an empty set at v_t >= L

As failures accumulate, q_t falls and the budget shrinks continuously instead of switching
off at a threshold. Settings, which the note leaves open:

| Symbol | Value | Why |
|---|---|---|
| K | 1 | Gain in logit units: at q/(ηB) = -0.5, the odds of the budget fall by a factor of e |
| B | 50 | Constant envelope, counted in excess failures. One shock fails about H trajectories, so this allows about five |
| ε_b, L | 0.01, 6 | Proposals are clipped to [0.01, 0.99], and L must exceed log(99) = 4.6 |
| c | 0.94 (derived) | Saturation at \|q/η\| = cB = 47 excess outcomes |

K and B were picked after a short check on three datasets (AMD, Nvidia, regime_shift), so
they are not tuned blind. B = 25 calibrates a little tighter but uses whole-space sets more.

**Two additions the note does not spell out**

1. **Budget floor.** Eq. 25 issues an ordinary allocation whenever v > -L, but no allocation
   exists below b = H·α_min. Those rounds now get the tightest ordinary allocation instead of
   whole-space sets. Only ordinary rounds change, so eqs. 26-27 still hold. This is what cuts
   DtACI's whole-space rounds; most of them were infeasible budgets, not safeguard actions.
2. **DtACI scores against the issued budget.** Its counterfactual family now uses b_t rather
   than the proposal b̃_t, so Z_k(b_k) still equals the actual label under supervision.

**Wind.** The calibration window went from 100 to 150 (t_max 480 to 440). With window 100,
H·α_min was 0.119, above β, so no allocation could meet the target and wind fell back to
whole-space sets on 36-54% of rounds under ACI and PID at 10/30. Wind numbers are therefore
not directly comparable with 10/30; use its own hard-threshold rerun in `5_safeguard.png`.

## What the results show

Tangent vs hard thresholds on the same data, across all nine datasets (`SUMMARY.md` has each one):

| Controller | Trajectory failure (median, range) | Rounds on whole-space sets (median, range) |
|---|---|---|
| DtACI | 0.150 (0.108-0.168) → **0.115** (0.102-0.150) | 25% (0-35%) → **1%** (0-12%) |
| ACI | 0.098 (0.077-0.103) → 0.098 (0.089-0.104) | 5% (0-9%) → **0%** |
| PID | 0.083 (0.050-0.097) → 0.093 (0.075-0.103) | 1% (0-1%) → **0%** |

- **DtACI's blow-up is contained.** Under hard thresholds it sat on the q = -1 rail on six datasets,
  and 0.15 was simply that rail's bound: 0.10 + 1/(0.01 × 2000). `5_safeguard.png` shows q_t/η now
  levelling off inside ±47 instead of falling to -100.
- **It is contained, not cured.** DtACI still sits above β on all nine datasets (solar only just, at 0.102). Its proposal
  b̃_t still swings between about 0 and 1 in a sawtooth; the safeguard is correcting that, and q_t
  stays negative as a result.
- **PID and ACI barely needed it.** They rarely approached a rail. The safeguard removes their
  whole-space rounds and moves PID's conservative runs closer to β (solar 0.050 → 0.093).
- **Every run is inside the guarantee.** Realized calibration error is within the eq. 27 bound in
  all cases (0.033 at T = 2000, 0.148 for wind at T = 440).

**Limits to keep in mind**

- **Intervals got wider.** The narrow DtACI intervals at 10/30 were paid for with failures and
  whole-space rounds. On rounds that were ordinary under both safeguards, AMD's mean width is 0.290
  with tangent and 0.210 with thresholds.
- **Static β/H is still hard to beat on stocks.** It has lower failure and narrower intervals than
  DtACI on AMD, BlackBerry, Fannie Mae and Nvidia, than ACI on AMD, BlackBerry and Fannie Mae, and
  than PID on Amazon, AMD and Fannie Mae. Feedback pays off where static over-covers: solar, wind,
  drone. Static has no calibration guarantee on any path.
- **Wind is the weakest result.** DtACI fails 0.150 and the uncertified plug-in 0.168. With only 440
  origins, the bound is too loose to force much.
- **Fannie Mae still uses whole-space sets on 12% of rounds** under DtACI, because q_t reaches the envelope.
- **Certified overlap credit is still zero on every dataset**, so TRA cert behaves as a weighted
  Bonferroni allocation. This run does not change that.

**Why DtACI keeps overshooting.** A side check, not part of these results, reran DtACI with its
step sizes capped at 0.016 instead of 0.128: AMD went to failure 0.097 and 0% whole-space (from 0.111
and 4%), with mean proposal 0.15 (from 0.34); regime_shift went to 0.108 and 0% (from 0.118 and 1%),
with mean proposal 0.21 (from 0.38). That supports fast steps plus H-delayed feedback as the cause,
on two datasets so far.

**Run notes.** Nvidia's first run died from a segfault in JAX's XLA compiler, with no Python
traceback, and completed on rerun. The two crashes seen at 10/30 (`INVALID_INPUT` on drone_x PID and
blackberry Boole + DtACI) did not recur, and their cause was never identified.

## Methods (one runner, config variants)

| Method | Allocation across horizons | Budget controller |
|---|---|---|
| TRA cert + DtACI | dependence-aware, certified overlap credit only | DtACI |
| TRA cert + ACI | same | fixed-gain ACI (γ=0.01) |
| TRA cert + PID | same | PID (kp=0.01, ki=0.001, kd=0.002, from `configs/adaptive_window_pid.yaml`) |
| TRA plug-in + DtACI | dependence-aware, point-estimate credit (no guarantee) | DtACI |
| Boole + DtACI | uniform Bonferroni split | DtACI |
| Boole + ACI (previous) | uniform Bonferroni split | ACI (the earlier system) |
| Static β/H | α_h = 0.01 every step, no feedback or safeguard | none |

## Files in each dataset folder, in reading order

1. **`1_overview.png`**: the whole run.
   (a) rolling miscoverage for all methods, with the blow-up windows shaded;
   (b) interval width on ordinary rounds; (c) issued budget b_t (solid) and controller proposal b̃_t (faint);
   (d) share of rounds on whole-space intervals; (e) accounting state q_t against the saturation
   envelope; (f) the raw data (stock prices for stocks).
2. **`2_per_horizon_budget.png`**: how the trajectory budget is split across the 10 steps over time
   (stacked α_1..α_10, top edge = Σα, black = issued budget), per controller, plus mean issued α_h
   vs realized per-step miss rate.
3. **`3_blowup_<i>_t<a>-<b>.png`**: zoom on the two worst miscoverage windows of TRA cert + DtACI:
   prices with violation times marked, the actual tubes every 5 origins, which method failed at
   which origin, which horizon was missed, and the budget, q_t and α_h inside the window.
4. **`4_overlap.png`**: Boole charge vs overlap credit, credit claimed vs realized Boole slack,
   empirical failure dependence across horizons, and the horizon pairs used by the credit tree.
5. **`5_safeguard.png`**: the direct test of section 2.5. For DtACI, ACI and PID on the same data:
   what the safeguard does to the budget, q_t/η against the envelope, and rolling miscoverage,
   tangent vs hard thresholds.
6. `summary.json` (all numbers), `data.npz` (raw arrays), `log.txt`.
