"""Optional matplotlib plots for the defense sweep. Degrades gracefully: if
matplotlib is not installed, the sweep still produces CSV + Markdown.

Plots produced (per k), strategy curves over noise budget:
    noise budget vs single-out risk proxy
    noise budget vs cra_ratio_median
    noise budget vs avg info loss
    noise budget vs cra_seconds
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def _budget_num(label: str) -> float:
    label = str(label)
    return float(label[:-1]) if label.endswith("%") else float(label)


def make_plots(rows: list[dict], out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    metrics = [
        ("single_out_risk_proxy", "single-out risk proxy (%)"),
        ("cra_ratio_median", "median CRA ratio (LR/CRA)"),
        ("avg_info_loss_record_weighted", "avg info loss (record-weighted)"),
        ("cra_seconds", "CRA runtime (s)"),
    ]
    # Build no_noise budget-0 baseline points per (k, seed-averaged).
    written: list[Path] = []
    ks = sorted({r["k"] for r in rows})
    for metric, ylabel in metrics:
        fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 4), squeeze=False)
        for ax, k in zip(axes[0], ks):
            by_strat: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
            nonoise: list[float] = []
            for r in rows:
                if r["k"] != k or r.get(metric) is None:
                    continue
                if r["strategy"] == "no_noise":
                    nonoise.append(r[metric])
                else:
                    by_strat[r["strategy"]][_budget_num(r["noise_budget_label"])].append(r[metric])
            base = sum(nonoise) / len(nonoise) if nonoise else None
            for strat, bmap in sorted(by_strat.items()):
                xs = sorted(bmap)
                # prepend budget 0 = no_noise baseline
                px = ([0.0] + xs) if base is not None else xs
                py = ([base] + [sum(bmap[x]) / len(bmap[x]) for x in xs]) if base is not None \
                    else [sum(bmap[x]) / len(bmap[x]) for x in xs]
                ax.plot(px, py, marker="o", label=strat)
            if base is not None:
                ax.axhline(base, ls="--", color="gray", alpha=0.6, label="no_noise")
            ax.set_title(f"k={k}")
            ax.set_xlabel("noise budget (% of dataset)")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"plot_{metric}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
    return written
