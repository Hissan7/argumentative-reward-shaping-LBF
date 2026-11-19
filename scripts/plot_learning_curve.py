"""plot the learning curves (Q-learning vs random, plus the heuristic if it has runs) from results/raw."""

from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path("results/raw")
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#555555" # colours


def latest_runs(exp: str) -> list[Path]:
    runs = sorted(p for p in (RESULTS / exp).iterdir() if p.is_dir())
    latest = max(p.name.split("_")[0] for p in runs)
    chosen = [p for p in runs if p.name.split("_")[0] == latest]
    print(f"{exp}: {len(chosen)} runs from launch {latest}")
    return chosen


def load(runs: list[Path], file: str, col: str, window: int = 1) -> tuple[np.ndarray, np.ndarray]:
    frames = [pd.read_csv(r / file) for r in runs]
    values = np.stack([f[col].rolling(window, min_periods=window).mean() for f in frames])
    return frames[0]["episode"].to_numpy(), values

# t-based 95% confidence interval
def mean_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = values.shape[0]
    half = stats.t.ppf(0.975, n - 1) * values.std(axis=0, ddof=1) / np.sqrt(n)
    return values.mean(axis=0), half


def band(ax, x, values, color, label, **line_kw) -> None:
    mean, half = mean_ci(values)
    ax.plot(x, mean, color=color, lw=2, label=label, **line_kw)
    ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.18, lw=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-exp", default="baseline_5x5")
    parser.add_argument("--random-exp", default="random_5x5")
    parser.add_argument("--heuristic-exp", default="heuristic_5x5")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("results/figures/baseline_5x5_learning_curve"))
    args = parser.parse_args()

    q_runs, r_runs = latest_runs(args.q_exp), latest_runs(args.random_exp)
    n_q, n_r = len(q_runs), len(r_runs)
    h_runs = latest_runs(args.heuristic_exp) if (RESULTS / args.heuristic_exp).exists() else []

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax, col, ylabel in [(axes[0], "length", "Episode length (steps)"),(axes[1], "env_return", "Env-only return"),]:
        band(ax,*load(q_runs, "metrics.csv", col, args.window),BLUE,"Q-learning (training, ε-greedy)",)
        band(ax,*load(q_runs, "eval_metrics.csv", col),BLUE,"Q-learning (greedy eval, ε = 0)",ls="--",)
        band(ax, *load(r_runs, "metrics.csv", col, args.window), ORANGE, "Random agent")
        if h_runs:
            band(ax, *load(h_runs, "metrics.csv", col, args.window), GREY, "Heuristic (nearest food)", ls=":")
        ax.set_xlabel("Training episode")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, None)
    axes[0].set_ylim(0, None)
    axes[1].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, loc="center right")
    n_h = f" / {len(h_runs)}" if h_runs else ""
    fig.suptitle(f"1-player 5×5 LBF, 1 food: tabular Q-learning vs random{' vs heuristic' if h_runs else ''}\n"
        f"(n = {n_q} / {n_r}{n_h} seeds; shaded = 95% CI, t-dist.; "
        f"training curves = {args.window}-episode rolling mean)",
        fontsize=9.5,)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=200)
    print(f"Saved {args.out}.png and .pdf")


if __name__ == "__main__":
    main()
