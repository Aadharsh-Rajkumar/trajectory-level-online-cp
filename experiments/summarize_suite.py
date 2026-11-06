"""Cross-dataset summary of a suite written by run_suite.sh.

Usage: PYTHONPATH=experiments:src .venv_local/bin/python experiments/summarize_suite.py [results/11_06-tangent_safeguard]
Writes SUMMARY.md and summary_all.png into the suite folder.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_1d import CONTROLLERS, METHODS, STATIC_COLOR

STATIC = "Static β/H (no feedback)"


def main(root="results/11_06-tangent_safeguard"):
    root = Path(root)
    runs = sorted(root.glob("*/*/summary.json"))
    data = {f"{p.parent.parent.name}/{p.parent.name}": json.loads(p.read_text()) for p in runs}
    methods = list(METHODS) + [STATIC]
    lines = [f"# Suite summary ({root})", "",
             "Trajectory miscoverage, target β = 0.10. Cells: overall / last 1000 origins / "
             "fallback-round share. Width is the mean interval width on ordinary rounds.", ""]
    lines += ["| dataset | " + " | ".join(methods) + " |", "|---" * (len(methods)+1) + "|"]
    for name, s in data.items():
        cells = []
        for m in methods:
            v = s["methods"].get(m)
            if v is None:
                cells.append("crashed" if m in s.get("crashed", {}) else "—")
                continue
            fb = f" / fb {v['fallback_rate']:.2f}" if "fallback_rate" in v else ""
            cells.append(f"{v['failure_rate']:.3f} / {v['failure_rate_last_1000']:.3f}{fb}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "| dataset (mean width) | " + " | ".join(methods) + " |", "|---" * (len(methods)+1) + "|"]
    for name, s in data.items():
        lines.append(f"| {name} | " + " | ".join(
            f"{s['methods'][m]['mean_width_ordinary']:.4g}" if m in s["methods"] else "—" for m in methods) + " |")
    if any(s.get("hard_threshold_baseline") for s in data.values()):
        controllers = [m for m in methods if m in CONTROLLERS]
        lines += ["", "Safeguard comparison: tangent (note section 2.5) vs hard q thresholds, same data and "
                  "controller. Cells: failure rate / whole-space share / worst rolling-200 failure, tangent → thresholds.", "",
                  "| dataset | " + " | ".join(controllers) + " |", "|---" * (len(controllers)+1) + "|"]
        for name, s in data.items():
            cells = []
            for m in controllers:
                v, b = s["methods"].get(m), s.get("hard_threshold_baseline", {}).get(m)
                cells.append(f"{v['failure_rate']:.3f} / {v['fallback_rate']:.2f} / {v['peak_rolling_failure']:.2f} → "
                             f"{b['failure_rate']:.3f} / {b['fallback_rate']:.2f} / {b['peak_rolling_failure']:.2f}"
                             if v and b else "—")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += ["", "Realized calibration error |mean Z − β| vs the pathwise bound (note eq. 27).", "",
                  "| dataset | " + " | ".join(controllers) + " |", "|---" * (len(controllers)+1) + "|"]
        for name, s in data.items():
            lines.append(f"| {name} | " + " | ".join(
                f"{s['methods'][m]['calibration_error']:.3f} ≤ {s['methods'][m]['calibration_bound']:.3f}"
                if m in s["methods"] else "—" for m in controllers) + " |")
    incomplete = [f"{p.parent.parent.name}/{p.parent.name}" for p in sorted(root.glob("*/*/log.txt"))
                  if not (p.parent/"summary.json").exists()]
    if incomplete:
        lines += ["", f"**Incomplete runs (log but no summary.json, missing from every table): {', '.join(incomplete)}**"]
    crashed = {n: s["crashed"] for n, s in data.items() if s.get("crashed")}
    if crashed:
        lines += ["", "Crashed runs:", *[f"- {n}: {c}" for n, c in crashed.items()]]
    (root/"SUMMARY.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(2, 1, figsize=(16, 9))
    x = np.arange(len(data))
    w = 0.8/len(methods)
    for j, m in enumerate(methods):
        c = METHODS.get(m, STATIC_COLOR)
        for a, key in zip(ax, ("failure_rate", "failure_rate_last_1000")):
            vals = [s["methods"].get(m, {}).get(key, np.nan) for s in data.values()]
            a.bar(x + (j - len(methods)/2)*w + w/2, vals, w, color=c, label=m)
    for a, title in zip(ax, ("overall", "last 1000 origins")):
        a.axhline(0.1, c="k", ls=":", lw=1)
        a.set(xticks=x, xticklabels=list(data), ylabel="trajectory miscoverage",
              title=f"Trajectory miscoverage by dataset and method ({title}); dotted = β")
    ax[0].legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(root/"summary_all.png", dpi=110)
    print((root/"SUMMARY.md").read_text())


if __name__ == "__main__":
    main(*sys.argv[1:2])
