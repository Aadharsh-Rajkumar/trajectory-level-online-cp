"""AR vs ALPaCA forecaster under identical trajectory-risk methods.

Usage: .venv_local/bin/python experiments/compare_forecasters.py RESULTS_AR RESULTS_ALPACA
Reads the <dataset>/data.npz files written by run_1d.py and saves
<dataset>/5_ar_vs_alpaca.png plus comparison.json into RESULTS_ALPACA.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_1d import COLORS, finite_mean, rolling


def load(path):
    z = np.load(path, allow_pickle=False)
    methods = list(z["methods"])
    out = {m: {k: z[f"{i}_{k}"] for k in ("fails", "widths", "modes", "credit", "viol_count")}
           for i, m in enumerate(methods)}
    out["Static β/H (no feedback)"] = dict(fails=z["static_fails"], widths=z["static_widths"],
                                           modes=np.zeros(len(z["static_fails"]), int),
                                           credit=np.zeros(len(z["static_fails"])),
                                           viol_count=np.zeros(len(z["static_fails"])))
    return out


def main(ar_dir, alpaca_dir):
    ar_dir, alpaca_dir = Path(ar_dir), Path(alpaca_dir)
    table = {}
    for path in sorted(ar_dir.glob("*/data.npz")):
        other = alpaca_dir/path.parent.name/"data.npz"
        if not other.exists():
            continue
        name = path.parent.name
        runs = {"AR": load(path), "ALPaCA": load(other)}
        fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
        table[name] = {}
        for (m, _), c in zip(runs["AR"].items(), COLORS):
            for f, ls in (("AR", "-"), ("ALPaCA", "--")):
                r = runs[f][m]
                w = np.where(np.isfinite(r["widths"]), r["widths"], np.nan)
                ax[0].plot(rolling(r["fails"]), c=c, ls=ls, lw=1.2, label=f"{m} [{f}]")
                ax[1].plot(rolling(np.nanmean(w, 1)), c=c, ls=ls, lw=1.2)
                if m.startswith("TRA"):
                    ax[2].plot(rolling(r["credit"]), c=c, ls=ls, lw=1.2, label=f"{m} [{f}] credit")
                table[name].setdefault(m, {})[f] = dict(
                    failure_rate=float(r["fails"].mean()), mean_width=finite_mean(r["widths"]),
                    fallbacks=int((r["modes"] != 0).sum()))
        ax[0].axhline(0.1, c="k", ls=":", lw=1)
        ax[0].set(title="Rolling trajectory miscoverage (solid AR, dashed ALPaCA)", ylim=(0, 0.45),
                  xlabel="origin t")
        ax[0].legend(fontsize=6, ncol=2)
        ax[1].set(title="Rolling mean width (fallback rounds excluded)", xlabel="origin t")
        ax[2].set(title="Overlap credit claimed by TRA (Σ_tree p̂_st)", xlabel="origin t")
        ax[2].legend(fontsize=7)
        fig.suptitle(f"{name}: AR(5) vs ALPaCA forecaster, same conformal + allocation pipeline")
        fig.tight_layout()
        fig.savefig(alpaca_dir/name/"5_ar_vs_alpaca.png", dpi=120)
        plt.close(fig)
    (alpaca_dir/"comparison.json").write_text(json.dumps(table, indent=2))
    for d, rows in table.items():
        print(f"\n== {d}")
        for m, v in rows.items():
            print(f"  {m:28s} " + "  ".join(
                f"{f}: fail={x['failure_rate']:.3f} width={x['mean_width']:.4g} fb={x['fallbacks']}"
                for f, x in v.items()))


if __name__ == "__main__":
    main(*sys.argv[1:3])
