"""Generate Figure 1 for the Spiral-Time state-equated comparison (V5).

Panel A is the conceptual comparison requested for the manuscript: two different
ordered histories are brought to exactly the same conventional current state.
The level-zero/current-state branch hides the archive, whereas Spiral-Time
exposes selected ordered history. Panels B and C show the empirical response
comparison and gain sensitivity using the stored article results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric(summary: dict, key: str, metric: str):
    value = summary["resolutions"][key][metric]
    mean = float(value["mean"])
    low = float(value["ci95_low"])
    high = float(value["ci95_high"])
    return mean, mean - low, high - mean


def _sensitivity_file(directory: Path, gain: str) -> Path:
    candidates = [
        directory / f"g_{gain}" / "state_resolution_results.json",
        directory / f"g_{gain}" / "article_results" / "state_resolution_results.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No sensitivity result file found for gain {gain}: {candidates}")


def _panel_a(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("A  State-equated comparison", loc="left", fontsize=12, fontweight="bold")

    # Ordered histories.
    ax.text(0.03, 0.88, "History a: alternating", fontsize=10, fontweight="bold")
    ax.text(0.03, 0.81, "AB -> BC -> AB -> BC -> ...", fontsize=9.5)
    ax.text(0.03, 0.68, "History b: blocked", fontsize=10, fontweight="bold")
    ax.text(0.03, 0.61, "AB -> AB -> ... -> BC -> BC -> ...", fontsize=9.5)

    # Converge on the identical conventional state.
    ax.add_patch(FancyArrowPatch((0.28, 0.55), (0.28, 0.46), arrowstyle="-|>", mutation_scale=14, lw=1.1))
    state_box = FancyBboxPatch((0.08, 0.34), 0.40, 0.11,
                               boxstyle="round,pad=0.018,rounding_size=0.02",
                               fill=False, linewidth=1.0)
    ax.add_patch(state_box)
    ax.text(0.28, 0.405, r"$X_t^{(a)}=X_t^{(b)}$ exactly", ha="center", va="center",
            fontsize=11, fontweight="bold")

    # Two ways to expose the same present.
    left_box = FancyBboxPatch((0.015, 0.07), 0.46, 0.20,
                              boxstyle="round,pad=0.012,rounding_size=0.015",
                              fill=False, linewidth=0.8)
    right_box = FancyBboxPatch((0.525, 0.07), 0.46, 0.20,
                               boxstyle="round,pad=0.012,rounding_size=0.015",
                               fill=False, linewidth=0.8)
    ax.add_patch(left_box)
    ax.add_patch(right_box)
    ax.text(0.035, 0.225, "Current-state / Markov baseline", fontsize=8.8, fontweight="bold")
    ax.text(0.035, 0.165, r"archive hidden: $Z_t^{(0)}=X_t$", fontsize=8.4)
    ax.text(0.035, 0.105, r"same $q$: $\Delta_q^{(0)}=0$", fontsize=8.4)

    ax.text(0.545, 0.225, "Spiral-Time", fontsize=8.8, fontweight="bold")
    ax.text(0.545, 0.165, r"archive exposed: $Z_t^{(r,a)}\ne Z_t^{(r,b)}$", fontsize=8.4)
    ax.text(0.545, 0.105, r"same $q$: $\Delta_q^{(r)}>0$", fontsize=8.4)


def draw(main_path: Path, sensitivity_dir: Path, output: Path) -> None:
    main = _load(main_path)
    summary = main["summary"]

    keys = ["0", "1", "4", "-1"]
    labels = ["current\nstate", "last 1", "last 4", "full\narchive"]
    x = np.arange(len(keys), dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.15), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.30, 1.0, 0.95]})

    # A: conceptual construction and comparison.
    _panel_a(axes[0])

    # B: response separation under identical continuation.
    ax = axes[1]
    phase = [_metric(summary, key, "mean_phase_rms") for key in keys]
    means = np.array([v[0] for v in phase])
    errors = np.array([[v[1] for v in phase], [v[2] for v in phase]])
    bars = ax.bar(x, means, yerr=errors, capsize=3, width=0.62)
    ax.set_xticks(x, labels)
    ax.set_ylabel("mean phase-trajectory RMS")
    ax.set_title("B  Response under identical continuation", loc="left", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    # Keep the zero control visible without inventing a nonzero value.
    if means[0] == 0:
        ax.plot([x[0] - 0.22, x[0] + 0.22], [0, 0], linewidth=2.2)

    # C: sensitivity to history-expression gain at full archive visibility.
    ax = axes[2]
    gains = ["0.07", "0.14", "0.28"]
    gx = np.arange(len(gains), dtype=float)
    vals = []
    for gain in gains:
        obj = _load(_sensitivity_file(sensitivity_dir, gain))
        vals.append(_metric(obj["summary"], "-1", "mean_phase_rms"))
    gy = np.array([v[0] for v in vals])
    ge = np.array([[v[1] for v in vals], [v[2] for v in vals]])
    ax.bar(gx, gy, yerr=ge, capsize=3, width=0.58)
    ax.set_xticks(gx, gains)
    ax.set_xlabel(r"history-expression gain $\gamma_H$")
    ax.set_ylabel("mean phase-trajectory RMS")
    ax.set_title("C  Full-history feedback sensitivity", loc="left", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--sensitivity-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(args.main, args.sensitivity_dir, args.output)


if __name__ == "__main__":
    main()
