"""1-D trajectory-level online CP experiments on the datasets from DtACI_ALPaCA and friends.

Usage:  .venv_local/bin/python -u experiments/run_1d.py [--forecaster ar|alpaca]
            [--t_max 2000] [--datasets amd ...] [--out DIR]
Per dataset, writes into DIR (or results/<timestamp>_1d_<forecaster>/<dataset>/):
  1_overview.png            miscoverage, width, controller budget, fallbacks, accounting q_t, data
  2_per_horizon_budget.png  how the trajectory budget is split across the H steps over time
  3_blowup_<i>_t<a>-<b>.png zoom on the worst miscoverage windows (prices, tubes, failures)
  4_overlap.png             Boole charge vs overlap credit diagnostics
  5_safeguard.png           tangent safeguard (note section 2.5) vs the hard q thresholds
  data.npz, summary.json

Intervals are mean +/- scale * Q, with Q the conformal quantile of the last
WINDOW revealed standardized residuals at the issued alpha. AR has unit scale;
ALPaCA supplies its predictive standard deviation.
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from trajectory_risk.configs import RiskMode, load_config
from trajectory_risk.alpaca import ALPaCAForecaster
from trajectory_risk.conformal_adapter import (ARForecaster, ConformalIntervalAdapter, _half_width,
                                               compute_forecasts)
from trajectory_risk.controllers import build_accounting_guard
from trajectory_risk.dependence import candidate_edges
from trajectory_risk.runner import OnlineRunner

ROOT = Path(__file__).resolve().parents[1]
# Earlier DtACI_ALPaCA project: stock CSVs, regime-shift and drone generators.
OLD = Path(os.environ.get("DTACI_ALPACA_DIR", "/Users/aadharshrajkumar/Documents/DtACI_ALPaCA"))
N_TRAIN, WINDOW, H = 500, 300, 10
ALPHA_MIN = 0.004  # must exceed 1/(WINDOW+1) for finite conformal quantiles
STOCKS = {"amd": "AMD", "blackberry": "Blackberry", "fanniemae": "FannieMae"}


# ── datasets (one scalar series each) ────────────────────────────────────────
def _read_csv_column(path, column):
    with open(path) as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))
    return rows, np.array([float(r[column]) for r in rows])


def close_prices(ticker):
    _, close = _read_csv_column(OLD/f"DtACI/MajorDailyReturns/{ticker}DailyPrices.csv", "Close")
    return close[::-1]                                 # file is newest-first


def daily_returns(n, ticker):
    return np.diff(np.log(close_prices(ticker)))[-n:]


def nvidia_returns(n):
    _, y = _read_csv_column(OLD/"ALPaCA/datasets/Nvidia-fc.csv", "log_value")
    return y[-n:]


def npy_series(n, name):
    """Series pre-extracted to data/ (pickles/CSVs from conformal-time-series and ALPaCA)."""
    return np.load(ROOT/"data"/f"{name}.npy")[-n:]


def regime_shift(n, t0):
    """Coordinate 1 of the old 2-D regime-shift 'stocks' data (var 4.0 <-> 0.3, gap 1000)."""
    sys.path.insert(0, str(OLD))
    from integrated_alpaca_dtaci import generate_multidim_regime_shift
    S1, S2 = np.array([[4.0, 0.5], [0.5, 0.3]]), np.array([[0.3, -0.2], [-0.2, 4.0]])
    prefix = np.random.default_rng(0).normal(0, 2.0, t0)   # D1 + calibration in regime 1
    return np.concatenate([prefix, generate_multidim_regime_shift(n-t0, 1000, S1, S2)[:, 0]])


def drone_x(n, t0):
    """x-position of the old drone-tracking robotics data, dynamics shift at online t=1000."""
    sys.path.insert(0, str(OLD/"scripts"))
    from generate_drone_tracking import generate_drone_trajectory
    pos, _ = generate_drone_trajectory(T=n, shift_t=t0+1000)
    return pos[:, 0]


def level_series(name, series):
    """What a reader wants to see under the forecast target: the price for stocks."""
    if name in STOCKS:
        return close_prices(STOCKS[name])[-series.size:], "close price ($)"
    if name in ("nvidia", "amazon"):
        return 100*np.exp(np.cumsum(series)), "price rebased to 100 (from log returns)"
    return series, "series value (forecast target)"


# ── methods = config variants of the same JAX runner ────────────────────────
TRA = "TRA cert"
METHODS = {  # label -> color
    f"{TRA} + DtACI": "#1f6feb",
    f"{TRA} + ACI": "#0891b2",
    f"{TRA} + PID": "#db2777",
    "TRA plug-in + DtACI": "#d97706",
    "Boole + DtACI": "#6b7280",
    "Boole + ACI (previous)": "#9333ea",
}
STATIC_COLOR = "#16a34a"
CONTROLLERS = [f"{TRA} + DtACI", f"{TRA} + ACI", f"{TRA} + PID"]
COLORS = list(METHODS.values())  # compare_forecasters.py
THRESHOLD_COLOR = "#9ca3af"


def make_configs(t_max, safeguard="tangent", gain=1.0, envelope=50.0):
    """gain, envelope = K, B of note eq. (24). B counts excess failures: one shock fails
    about H overlapping trajectories, so B=50 leaves room for roughly five of them."""
    base = load_config(str(ROOT/"configs/default.yaml"))
    base = replace(base, t_max=t_max,
                   history=replace(base.history, alpha_min=ALPHA_MIN, horizon=H),
                   controller=replace(base.controller, kind="dtaci"),
                   safety=replace(base.safety, kind=safeguard, tangent_gain=gain, tangent_envelope=envelope,
                                  tangent_budget_floor=H*ALPHA_MIN),
                   runner=replace(base.runner, drain=True, jit=True))
    aci = replace(base.controller, kind="aci")
    # Gains from configs/adaptive_window_pid.yaml.
    pid = replace(base.controller, kind="pid", kp=0.01, ki=0.001, kd=0.002, integral_limit=20.0)
    boole = replace(base, solver=replace(base.solver, forest_kind="fixed", fixed_forest=()),
                    weights=replace(base.weights, kind="static"))
    cfgs = [base, replace(base, controller=aci), replace(base, controller=pid),
            replace(base, risk_mode=RiskMode.EXPERIMENTAL_PLUGIN), boole, replace(boole, controller=aci)]
    return dict(zip(METHODS, cfgs))


def static_bonferroni(adapter, beta):
    """No-feedback baseline alpha_h = beta/H, scored from the same conformal p-values."""
    a = beta/H
    fails = np.array([bool(np.any(r["level"] < a)) for r in adapter.records.values()])
    widths = np.array([[2*r["scale"][h]*_half_width(r["sdesc"][h], a) for h in range(H)]
                       for r in adapter.records.values()])
    return fails, widths


def run_method(cfg, series, forecasts):
    adapter = ConformalIntervalAdapter(cfg, series, n_train=N_TRAIN, window=WINDOW,
                                       forecasts=forecasts)
    runner = OnlineRunner(cfg, adapter)
    t0 = time.time()
    log = {k: [] for k in ("budget", "proposal", "q", "risk", "forest")}
    while runner.next_time < cfg.t_max + H:
        d = runner.step()
        if runner.next_time <= cfg.t_max:
            log["budget"].append(float(d.budget))
            log["proposal"].append(float(d.proposal))
            log["q"].append(float(d.q))
            log["risk"].append(float(d.solver.acceptance_risk))
            log["forest"].append(np.asarray(d.forest_mask))
    recs = [adapter.records[k] for k in range(cfg.t_max)]
    modes = np.array([r["mode"] for r in recs])
    alphas = np.array([r["alpha"] for r in recs])
    viol = np.array([r["viol"] for r in recs])
    risk = np.where(modes == 0, log["risk"], np.nan)
    return dict(adapter=adapter, modes=modes, alphas=alphas, viol=viol,
                fails=viol.any(1), viol_count=viol.sum(1),
                widths=np.array([2*r["half_width"] for r in recs]),
                budget=np.array(log["budget"]), proposal=np.array(log["proposal"]),
                q=np.array(log["q"]), risk=risk,
                credit=np.where(modes == 0, np.nansum(alphas, 1), np.nan) - risk,
                forest=np.array(log["forest"]), seconds=time.time()-t0)


def q_rails(cfg):
    """q values at which the safeguard forces its extreme actions."""
    if cfg.safety.kind == "tangent":
        guard = build_accounting_guard(cfg.safety, cfg.beta)
        rail = guard.saturation*cfg.safety.tangent_envelope*cfg.safety.eta_q
        return rail, -rail
    return cfg.safety.q_upper, cfg.safety.q_lower


def calibration_bound(cfg, t):
    """Pathwise bound on |mean Z - beta|: note eq. (27) for tangent, eq. (22) for thresholds."""
    s, beta = cfg.safety, cfg.beta
    if s.kind == "tangent":
        return (q_rails(cfg)[0]/s.eta_q + 2*H*max(beta, 1-beta))/t
    bq = max(s.q_upper + s.eta_q*beta*H, -s.q_lower + s.eta_q*(1-beta)*H)
    return (bq/s.eta_q + H)/t


def safeguard_label(cfg):
    s = cfg.safety
    if s.kind == "tangent":
        return f"tangent safeguard K={s.tangent_gain:g}, B={s.tangent_envelope:g}"
    return f"threshold safeguard q in [{s.q_lower:g}, {s.q_upper:g}]"


def rolling(x, w=200):
    """Trailing mean over w origins, ignoring NaN (e.g. whole-space fallback rounds)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    c = np.cumsum(np.insert(np.where(ok, x, 0), 0, 0))
    n = np.cumsum(np.insert(ok, 0, 0))
    out = np.full(x.size, np.nan)
    cnt = n[w:]-n[:-w]
    out[w-1:] = np.where(cnt > 0, (c[w:]-c[:-w])/np.maximum(cnt, 1), np.nan)
    return out


def finite_mean(w):
    w = w[np.isfinite(w).all(axis=1)]
    return float(w.mean()) if w.size else float("nan")


def blowup_windows(fails, count=2, half=100):
    """Origin windows centred on the highest centred-100 miscoverage peaks, non-overlapping."""
    f = np.asarray(fails, float)
    score = np.convolve(f, np.ones(half), "same")/half
    picked = []
    for _ in range(count):
        if not np.any(np.isfinite(score)) or np.nanmax(score) <= 0:
            break
        k = int(np.nanargmax(score))
        a, b = max(0, k-half), min(f.size, k+half)
        picked.append((a, b))
        score[max(0, a-half):min(f.size, b+half)] = -np.inf
    return sorted(picked)


FORECASTER = "AR(5) forecaster"
FOCUS = f"{TRA} + DtACI"  # method shown in the zooms; first TRA method that ran


def _shade(a, windows, label=True):
    for i, (lo, hi) in enumerate(windows):
        a.axvspan(lo, hi, color="#fca5a5", alpha=0.25, lw=0,
                  label="blow-up windows (see 3_blowup_*)" if label and i == 0 else None)


# ── 1: overview ──────────────────────────────────────────────────────────────
def plot_overview(name, out, results, static, beta, cfg, level, level_label, base, windows):
    fig, ax = plt.subplots(3, 2, figsize=(17, 14))
    a = ax[0, 0]
    for m, r in results.items():
        a.plot(rolling(r["fails"]), c=METHODS[m], lw=1.3, label=m)
    a.plot(rolling(static[0]), c=STATIC_COLOR, ls="--", label="Static β/H (no feedback)")
    a.axhline(beta, c="k", ls=":", lw=1, label=f"target β={beta}")
    a.axhline(10/200, c="#9ca3af", ls="-.", lw=0.8, label="one shock = H/200 = 0.05")
    _shade(a, windows)
    a.set(title="(a) Rolling trajectory miscoverage (200-origin window)", xlabel="origin t",
          ylim=(0, 0.45))
    a.legend(fontsize=7, ncol=2)

    a = ax[0, 1]
    for m, r in results.items():
        w = np.where(np.isfinite(r["widths"]), r["widths"], np.nan)
        a.plot(rolling(np.nanmean(w, 1)), c=METHODS[m], lw=1.2, label=m)
    a.plot(rolling(static[1].mean(1)), c=STATIC_COLOR, ls="--", label="Static β/H")
    a.set(title="(b) Rolling mean interval width, ordinary rounds only\n"
                "(fallback rounds issue infinite intervals; see (d))", xlabel="origin t")

    a = ax[1, 0]
    for m, r in results.items():
        a.plot(r["proposal"], c=METHODS[m], lw=0.5, alpha=0.3)
        a.plot(np.where(r["modes"] == 0, r["budget"], np.nan), c=METHODS[m], lw=0.8, label=m)
    a.axhline(beta, c="k", ls=":", lw=1)
    a.set(title="(c) Budget issued to the allocator b_t (solid) and controller proposal b̃_t (faint)",
          xlabel="origin t", ylim=(0, 1))
    a.legend(fontsize=7, ncol=2)

    a = ax[1, 1]
    for m, r in results.items():
        a.plot(rolling((r["modes"] != 0).astype(float), 100), c=METHODS[m], lw=1.2, label=m)
    _shade(a, windows, label=False)
    a.set(title="(d) Fraction of fallback rounds (whole-space intervals), rolling 100",
          xlabel="origin t", ylim=(0, 1))

    a = ax[2, 0]
    for m, r in results.items():
        a.plot(r["q"], c=METHODS[m], lw=1, label=m)
    upper, lower = q_rails(cfg)
    a.axhline(upper, c="r", ls="--", lw=1, label=f"{upper:.2f}: forces an empty set")
    a.axhline(lower, c="b", ls="--", lw=1, label=f"{lower:.2f}: forces whole-space sets")
    a.set(title=f"(e) Exact accounting state q_t = q_0 + η Σ(β − Z_k), η={cfg.safety.eta_q}",
          xlabel="origin t")
    a.legend(fontsize=7)

    a = ax[2, 1]
    t = np.arange(len(next(iter(results.values()))["fails"]) + H)
    a.plot(t, level[base:base+t.size], c="k", lw=0.8)
    _shade(a, windows)
    a.set(title=f"(f) Data: {level_label}", xlabel="physical time (origin index)")
    a.legend(fontsize=7)
    fig.suptitle(f"{name}: overview (β={beta}, H={H}, {FORECASTER}, {safeguard_label(cfg)})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out/"1_overview.png", dpi=110)
    plt.close(fig)


# ── 2: per-horizon budget ────────────────────────────────────────────────────
def plot_per_horizon(name, out, results, beta, windows):
    shown = [m for m in METHODS if m.startswith("TRA") and m in results]
    fig, ax = plt.subplots(len(shown), 2, figsize=(17, 3.6*len(shown)),
                           gridspec_kw=dict(width_ratios=[3, 1]))
    cols = plt.cm.viridis(np.linspace(0, 0.95, H))
    for row, m in zip(ax, shown):
        r = results[m]
        ordinary = r["modes"] == 0
        al = np.where(ordinary[:, None], r["alphas"], np.nan)
        sm = np.stack([rolling(al[:, h], 25) for h in range(H)], 1)
        sm = np.nan_to_num(sm)
        a = row[0]
        a.stackplot(np.arange(sm.shape[0]), sm.T, colors=cols, labels=[f"α_{h+1}" for h in range(H)])
        a.plot(rolling(np.where(ordinary, r["budget"], np.nan), 25), c="k", lw=1, label="issued budget b_t")
        fb = np.where(~ordinary)[0]
        if fb.size:
            a.scatter(fb, np.zeros_like(fb), marker="|", c="#9ca3af", s=40, label="fallback round")
        _shade(a, windows, label=False)
        a.set(title=f"{m}: per-step risk α_h stacked (rolling 25, ordinary rounds); top edge = Σα",
              xlabel="origin t", ylim=(0, None))
        a.legend(fontsize=6, ncol=6, loc="upper left")
        a = row[1]
        x = np.arange(1, H+1)
        a.bar(x-0.2, np.nanmean(al, 0), 0.4, color=METHODS[m], label="mean issued α_h")
        a.bar(x+0.2, r["viol"][ordinary].mean(0) if ordinary.any() else np.zeros(H), 0.4,
              color="#dc2626", alpha=0.7, label="realized miss rate V_h")
        a.axhline(beta/H, c=STATIC_COLOR, ls="--", lw=1, label="β/H (Bonferroni)")
        a.set(title="per-step: issued α_h vs realized miss", xlabel="horizon h", xticks=x)
        a.legend(fontsize=7)
    fig.suptitle(f"{name}: how the trajectory budget is split across the {H} steps ({FORECASTER})",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out/"2_per_horizon_budget.png", dpi=110)
    plt.close(fig)


# ── 3: blow-up zooms ─────────────────────────────────────────────────────────
def plot_blowup(name, out, i, window, results, ref, level, level_label, beta):
    lo, hi = window
    focus = results[FOCUS]
    ref_recs = ref.records
    base = ref.base
    fig = plt.figure(figsize=(17, 15))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1.2, 1, 1])

    a = fig.add_subplot(gs[0, :])
    t = np.arange(lo, hi+H)
    a.plot(t, level[base+lo:base+hi+H], c="k", lw=1)
    vt = sorted({k+h+1 for k in range(lo, hi) for h in range(H) if focus["viol"][k, h]})
    if vt:
        a.scatter(vt, level[base+np.array(vt)], c="#dc2626", s=18, zorder=3,
                  label=f"times where a {FOCUS} tube was violated")
    a.set(title=f"Data around the blow-up: {level_label}", xlim=(lo, hi+H))
    a.legend(fontsize=8)

    a = fig.add_subplot(gs[1, :])
    y = ref.y[base+lo:base+hi+H]
    a.plot(t, y, "k.-", lw=0.6, ms=3, label="y (forecast target)")
    finite = []
    for k in range(lo, hi, 5):
        rec = ref_recs[k]
        ts = np.arange(k+1, k+H+1)
        if rec["mode"] != 0:
            a.axvline(k, c="#9ca3af", lw=0.6, alpha=0.6)
            continue
        hw = rec["half_width"]
        finite.append(hw.max())
        a.fill_between(ts, rec["yhat"]-hw, rec["yhat"]+hw, alpha=0.22,
                       color="#dc2626" if rec["viol"].any() else "#1f6feb")
    if finite:
        span = np.nanmax(np.abs(y - np.nanmean(y))) + max(finite)
        a.set_ylim(np.nanmean(y)-1.2*span, np.nanmean(y)+1.2*span)
    a.set(title=f"{FOCUS}: H={H} tubes every 5 origins (red = trajectory failed, blue = covered, "
                "grey line = fallback / whole-space)", xlim=(lo, hi+H), xlabel="physical time")
    a.legend(fontsize=8)

    a = fig.add_subplot(gs[2, 0])
    ms = list(results)
    grid = np.zeros((len(ms), hi-lo))
    for j, m in enumerate(ms):
        r = results[m]
        grid[j] = np.where(r["modes"][lo:hi] != 0, 1, np.where(r["fails"][lo:hi], 2, 0))
    a.imshow(grid, aspect="auto", interpolation="nearest", cmap=ListedColormap(["#f3f4f6", "#9ca3af", "#dc2626"]),
             vmin=0, vmax=2, extent=(lo, hi, len(ms)-0.5, -0.5))
    a.set(yticks=range(len(ms)), yticklabels=ms, xlabel="origin k",
          title="Per-origin outcome: red = trajectory failed, grey = fallback, light = covered")

    a = fig.add_subplot(gs[2, 1])
    v = np.where((focus["modes"][lo:hi] != 0)[:, None], np.nan, focus["viol"][lo:hi].astype(float))
    a.imshow(v.T, aspect="auto", interpolation="nearest", cmap=ListedColormap(["#e5e7eb", "#dc2626"]),
             extent=(lo, hi, H+0.5, 0.5))
    a.set(xlabel="origin k", ylabel="horizon h",
          title=f"{FOCUS}: which steps were missed (V_k,h; blank = fallback)")

    a = fig.add_subplot(gs[3, 0])
    for m in [c for c in CONTROLLERS if c in results]:
        r = results[m]
        a.plot(np.arange(lo, hi), np.where(r["modes"] == 0, r["budget"], np.nan)[lo:hi], c=METHODS[m],
               label=f"{m}: issued b_t")
        a.plot(np.arange(lo, hi), r["proposal"][lo:hi], c=METHODS[m], lw=0.7, alpha=0.4, label=f"{m}: proposal b̃_t")
        a.plot(np.arange(lo, hi), r["q"][lo:hi], c=METHODS[m], ls=":", lw=1, label=f"{m}: q_t")
    a.axhline(beta, c="k", ls=":", lw=1)
    a.set(title="Issued budget, controller proposal and accounting state q_t in the window", xlabel="origin k")
    a.legend(fontsize=7, ncol=2)

    a = fig.add_subplot(gs[3, 1])
    al = np.where((focus["modes"][lo:hi] == 0)[:, None], focus["alphas"][lo:hi], 0)
    a.stackplot(np.arange(lo, hi), al.T, colors=plt.cm.viridis(np.linspace(0, 0.95, H)),
                labels=[f"α_{h+1}" for h in range(H)])
    a.set(title=f"{FOCUS}: per-step α_h issued at each origin (0 = fallback)", xlabel="origin k")
    a.legend(fontsize=6, ncol=5)
    rate = focus["fails"][lo:hi].mean()
    fig.suptitle(f"{name}: blow-up window {i} (origins {lo}-{hi}; {FOCUS} miscoverage "
                 f"{rate:.2f} vs β={beta})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out/f"3_blowup_{i}_t{lo}-{hi}.png", dpi=110)
    plt.close(fig)


# ── 4: overlap ───────────────────────────────────────────────────────────────
def plot_overlap(name, out, results, ref):
    """How much does the alpha-overlap (Kounias-Hunter) correction buy?"""
    tra = {m: results[m] for m in (f"{TRA} + DtACI", "TRA plug-in + DtACI") if m in results}
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))

    a = ax[0, 0]
    for m, r in tra.items():
        c = METHODS[m]
        a.plot(rolling(np.where(r["modes"] == 0, np.nansum(r["alphas"], 1), np.nan)), c=c,
               label=f"{m}: Σα (marginal charge)")
        a.plot(rolling(r["risk"]), c=c, ls="--", label=f"{m}: Σα − overlap credit")
        a.plot(rolling(np.where(r["modes"] == 0, r["budget"], np.nan)), c=c, ls=":", lw=1,
               label=f"{m}: budget b (ordinary rounds)")
    a.set(title="Budget vs marginal charge (gap between solid and dashed = overlap credit)",
          xlabel="origin t")
    a.legend(fontsize=7)

    a = ax[0, 1]
    for m, r in tra.items():
        c = METHODS[m]
        a.plot(rolling(r["credit"]), c=c, label=f"{m}: credit claimed Σ_tree p̂_st")
        a.plot(rolling(r["viol_count"] - r["fails"]), c=c, ls="--",
               label=f"{m}: realized Boole slack Σ_h V_h − Z")
    a.set(title="Overlap credit claimed vs realized Boole slack (per trajectory)",
          xlabel="origin t")
    a.legend(fontsize=7)

    # Method-independent: how dependent are horizon failures in this series?
    a = ax[1, 0]
    lv = np.array([r["level"] for r in ref.records.values()])
    lv = lv[np.isfinite(lv).all(1)]
    lev = 0.05
    fail = lv < lev
    cond = np.full((H, H), np.nan)
    for s in range(H):
        if fail[:, s].sum():
            cond[s] = fail[fail[:, s]].mean(0)
    im = a.imshow(cond, vmin=0, vmax=1, cmap="viridis", origin="lower",
                  extent=(0.5, H+0.5, 0.5, H+0.5))
    fig.colorbar(im, ax=a)
    a.set(title=f"P(horizon t fails | horizon s fails) at α={lev} (independent ≈ {lev})",
          xlabel="t", ylabel="s")

    a = ax[1, 1]
    edges = np.asarray(candidate_edges(H))
    for (m, r), off in zip(tra.items(), (-0.2, 0.2)):
        ordinary = r["modes"] == 0
        freq = r["forest"][ordinary].mean(0) if ordinary.any() else np.zeros(len(edges))
        top = np.argsort(-freq, kind="stable")[:12]
        a.bar(np.arange(len(top))+off, freq[top], width=0.4, color=METHODS[m], label=m)
        for i, e in enumerate(top):
            a.text(i+off, freq[e]+0.01, f"{edges[e][0]}-{edges[e][1]}", ha="center",
                   fontsize=6, rotation=90)
    a.set(title="Horizon pairs in the credit tree (selection frequency, top 12)",
          ylim=(0, 1.2), xticks=[])
    a.legend(fontsize=7)
    fig.suptitle(f"{name}: overlap (Σα − Σ p_st) diagnostics ({FORECASTER})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out/"4_overlap.png", dpi=110)
    plt.close(fig)


# ── 5: tangent safeguard vs hard thresholds ──────────────────────────────────
def plot_safeguard(name, out, results, baseline, cfg, floor):
    """Same data, forecaster and controller; only the safeguard differs (note section 2.5)."""
    shown = [m for m in CONTROLLERS if m in results]
    fig, ax = plt.subplots(len(shown), 3, figsize=(18, 3.8*len(shown)), squeeze=False)
    beta, eta = cfg.beta, cfg.safety.eta_q
    upper, _ = q_rails(cfg)
    for row, m in zip(ax, shown):
        r, c = results[m], METHODS[m]
        a = row[0]
        a.plot(r["proposal"], c=c, lw=0.6, alpha=0.35, label="controller proposal b̃_t")
        a.plot(np.where(r["modes"] == 0, r["budget"], np.nan), c=c, lw=0.9, label="issued b_t = σ(v_t)")
        a.axhline(beta, c="k", ls=":", lw=1, label=f"β={beta}")
        a.axhline(floor, c="#6b7280", ls="--", lw=0.8, label=f"budget floor H·α_min={floor:.3g}")
        a.set(title=f"{m}: what the safeguard does to the budget", xlabel="origin t", ylim=(0, 1))
        a.legend(fontsize=7)
        a = row[1]
        a.plot(r["q"]/eta, c=c, lw=1, label="tangent")
        if m in baseline:
            a.plot(baseline[m]["q"]/eta, c=THRESHOLD_COLOR, lw=1, label="hard thresholds")
            a.axhline(cfg.safety.q_lower/eta, c=THRESHOLD_COLOR, ls=":", lw=0.8)
        a.axhline(upper/eta, c=c, ls="--", lw=0.8, label=f"±cB = ±{upper/eta:.0f}")
        a.axhline(-upper/eta, c=c, ls="--", lw=0.8)
        a.set(title="Excess outcomes q_t/η = Σ(β − Z_k)", xlabel="origin t")
        a.legend(fontsize=7)
        a = row[2]
        a.plot(rolling(r["fails"]), c=c, lw=1.2,
               label=f"tangent: fail {r['fails'].mean():.3f}, whole-space {(r['modes'] != 0).mean():.0%}")
        if m in baseline:
            b = baseline[m]
            a.plot(rolling(b["fails"]), c=THRESHOLD_COLOR, lw=1.2,
                   label=f"hard thresholds: fail {b['fails'].mean():.3f}, whole-space {(b['modes'] != 0).mean():.0%}")
        a.axhline(beta, c="k", ls=":", lw=1)
        a.set(title="Rolling trajectory miscoverage (200 origins)", xlabel="origin t", ylim=(0, 0.45))
        a.legend(fontsize=7)
    fig.suptitle(f"{name}: {safeguard_label(cfg)} vs hard thresholds ({FORECASTER})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out/"5_safeguard.png", dpi=110)
    plt.close(fig)


def outcome_stats(r):
    return dict(failure_rate=float(r["fails"].mean()), fallback_rate=float((r["modes"] != 0).mean()),
                mean_width_ordinary=finite_mean(r["widths"]), q_min=float(r["q"].min()), q_max=float(r["q"].max()),
                peak_rolling_failure=float(np.nanmax(rolling(r["fails"]))), seconds=round(r["seconds"], 1))


def main():
    global FORECASTER, N_TRAIN, WINDOW, ALPHA_MIN, FOCUS
    p = argparse.ArgumentParser()
    p.add_argument("--t_max", type=int, default=2000)
    p.add_argument("--forecaster", choices=["ar", "alpaca"], default="ar")
    p.add_argument("--datasets", nargs="*", default=["amd", "nvidia", "regime_shift", "drone_x"])
    p.add_argument("--n_train", type=int, default=N_TRAIN)
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--scale", action="store_true",
                   help="divide the series by its training-segment std (for large-unit data like solar)")
    p.add_argument("--out", help="output dir for a single dataset; lets datasets run in parallel")
    p.add_argument("--safeguard", choices=["tangent", "threshold"], default="tangent",
                   help="note section 2.5 tangent supervision, or the hard q_lower/q_upper override")
    p.add_argument("--tangent_K", type=float, default=1.0, help="gain K of note eq. (24)")
    p.add_argument("--tangent_B", type=float, default=50.0, help="constant envelope B_t of note eq. (24)")
    p.add_argument("--no_baseline", action="store_true",
                   help="skip rerunning the ACI/PID/DtACI controllers with hard thresholds for 5_safeguard.png")
    args = p.parse_args()
    N_TRAIN, WINDOW = args.n_train, args.window
    ALPHA_MIN = max(ALPHA_MIN, 1.2/(WINDOW+1))
    FORECASTER = {"ar": "AR(5) forecaster", "alpaca": "ALPaCA forecaster"}[args.forecaster]
    t0 = N_TRAIN + WINDOW + H
    n = t0 + args.t_max + H
    loaders = {"amd": lambda: daily_returns(n, "AMD"), "nvidia": lambda: nvidia_returns(n),
               "regime_shift": lambda: regime_shift(n, t0), "drone_x": lambda: drone_x(n, t0),
               "blackberry": lambda: daily_returns(n, "Blackberry"),
               "fanniemae": lambda: daily_returns(n, "FannieMae"),
               "amazon": lambda: npy_series(n, "amazon"),
               "solar": lambda: npy_series(n, "solar_palo_alto"),
               "wind": lambda: npy_series(n, "wind")}
    root = ROOT/"results"/(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+f"_1d_{args.forecaster}")
    cfgs = make_configs(args.t_max, args.safeguard, args.tangent_K, args.tangent_B)
    cfg0 = next(iter(cfgs.values()))
    beta = cfg0.beta
    for d in args.datasets:
        out = Path(args.out) if args.out else root/d
        out.mkdir(parents=True, exist_ok=True)
        series = loaders[d]()
        if series.size < n:
            raise SystemExit(f"{d}: {series.size} points < n_train+window+t_max+2H = {n}")
        level, level_label = level_series(d, series)
        if args.scale:
            s = series[:N_TRAIN].std()
            series, level_label = series/s, level_label + f" (target scaled by 1/{s:.3g})"
        model = ARForecaster() if args.forecaster == "ar" else ALPaCAForecaster(horizon=H)
        forecasts = compute_forecasts(model, series, N_TRAIN, WINDOW, H, args.t_max)
        results, crashed = {}, {}
        for m, c in cfgs.items():
            try:
                results[m] = run_method(c, series, forecasts)
                print(f"  {d}: {m} done in {results[m]['seconds']:.0f}s", flush=True)
            except Exception as e:  # keep the other methods' results; reported in summary.json
                crashed[m] = f"{type(e).__name__}: {e}"
                print(f"  {d}: {m} CRASHED: {crashed[m]}", flush=True)
        baseline = {}
        if args.safeguard == "tangent" and not args.no_baseline:
            for m in CONTROLLERS:
                c = cfgs[m]
                try:
                    baseline[m] = run_method(replace(c, safety=replace(c.safety, kind="threshold")), series, forecasts)
                    print(f"  {d}: {m} with hard thresholds done in {baseline[m]['seconds']:.0f}s", flush=True)
                except Exception as e:
                    crashed[f"{m} (hard thresholds)"] = f"{type(e).__name__}: {e}"
                    print(f"  {d}: {m} with hard thresholds CRASHED: {e}", flush=True)
        if not results:
            raise SystemExit(f"{d}: every method crashed: {crashed}")
        FOCUS = next(m for m in list(METHODS) + list(results) if m in results)
        ref = results[FOCUS]["adapter"]
        static = static_bonferroni(ref, beta)
        windows = blowup_windows(results[FOCUS]["fails"])
        table = {m: dict(failure_rate=float(r["fails"].mean()),
                         failure_rate_last_1000=float(r["fails"][-1000:].mean()),
                         failure_rate_ordinary=float(r["fails"][r["modes"] == 0].mean()),
                         mean_width_ordinary=finite_mean(r["widths"]),
                         mean_alpha_sum=float(np.nanmean(r["alphas"][r["modes"] == 0].sum(1))),
                         mean_budget=float(np.mean(r["proposal"])),
                         mean_issued_budget=float(np.mean(r["budget"][r["modes"] == 0])) if (r["modes"] == 0).any() else None,
                         peak_rolling_failure=float(np.nanmax(rolling(r["fails"]))),
                         calibration_error=abs(float(r["fails"].mean())-beta),
                         calibration_bound=calibration_bound(cfgs[m], args.t_max),
                         mean_credit=float(np.nanmean(r["credit"])),
                         realized_slack=float(np.mean(r["viol_count"]-r["fails"])),
                         fallback_rate=float((r["modes"] != 0).mean()),
                         q_min=float(r["q"].min()), q_max=float(r["q"].max()),
                         seconds=round(r["seconds"], 1))
                 for m, r in results.items()}
        table["Static β/H (no feedback)"] = dict(
            failure_rate=float(static[0].mean()),
            failure_rate_last_1000=float(static[0][-1000:].mean()),
            mean_width_ordinary=finite_mean(static[1]), mean_alpha_sum=beta)
        plot_overview(d, out, results, static, beta, cfg0, level, level_label, ref.base, windows)
        plot_per_horizon(d, out, results, beta, windows)
        for i, w in enumerate(windows, 1):
            plot_blowup(d, out, i, w, results, ref, level, level_label, beta)
        plot_overlap(d, out, results, ref)
        if baseline:
            plot_safeguard(d, out, results, baseline, cfg0, H*ALPHA_MIN)
        np.savez(out/"data.npz", methods=np.array(list(results)),
                 static_fails=static[0], static_widths=static[1],
                 **{f"{i}_{k}": r[k] for i, r in enumerate(results.values())
                    for k in ("fails", "widths", "alphas", "modes", "budget", "proposal", "q", "risk",
                              "credit", "viol", "viol_count", "forest")},
                 baseline_methods=np.array(list(baseline)),
                 **{f"thr{i}_{k}": r[k] for i, r in enumerate(baseline.values())
                    for k in ("fails", "widths", "modes", "budget", "proposal", "q")})
        (out/"summary.json").write_text(json.dumps(
            dict(dataset=d, t_max=args.t_max, n_train=N_TRAIN, window=WINDOW, scaled=args.scale,
                 blowup_windows=windows, zoom_method=FOCUS, crashed=crashed, safeguard=safeguard_label(cfg0),
                 methods=table, hard_threshold_baseline={m: outcome_stats(r) for m, r in baseline.items()}),
            indent=2))
        print(f"\n== {d}", flush=True)
        for m, v in table.items():
            print(f"  {m:26s} fail={v['failure_rate']:.3f} last1000={v['failure_rate_last_1000']:.3f} "
                  f"width={v['mean_width_ordinary']:.4g}"
                  + (f" fallback={v['fallback_rate']:.2f} b̃={v['mean_budget']:.3f} "
                     f"credit={v['mean_credit']:.4f} q=[{v['q_min']:.2f},{v['q_max']:.2f}]"
                     if "fallback_rate" in v else ""), flush=True)
        for m, r in baseline.items():
            print(f"  {m:26s} hard thresholds: fail={r['fails'].mean():.3f} "
                  f"fallback={(r['modes'] != 0).mean():.2f}", flush=True)
        print(f"results -> {out}")


if __name__ == "__main__":
    main()
