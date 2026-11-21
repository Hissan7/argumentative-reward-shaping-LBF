"""Regret of a Q-learning experiment against value iteration"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from plot_learning_curve import latest_runs, mean_ci
from argrl.agents.value_iteration import enumerate_states, optimal_actions, optimal_steps, value_iteration
from argrl.envs.lbf_env import LBFEnv

TABLES = Path("results/tables")


def state_optimality(q_file: Path, states, Q_star) -> tuple[float, int]:
    data = np.load(q_file)
    learned = {tuple(int(x) for x in s): q for s, q in zip(data["states"], data["q"])}
    n_opt = 0
    for s in states:
        q = learned.get(s, np.zeros(6))  # unvisited. all-zero row as in the agent
        n_opt += set(np.flatnonzero(q == q.max()).tolist()) <= optimal_actions(Q_star, s)
    return n_opt / len(states), sum(s not in learned for s in states)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="baseline_5x5")
    args = parser.parse_args()

    runs = latest_runs(args.exp)
    cfg = yaml.safe_load((runs[0] / "config.yaml").read_text())
    gamma = cfg["agent"]["gamma"]
    states = enumerate_states(cfg["env"]["field_size"], cfg["env"]["max_player_level"])
    V, Q = value_iteration(states, gamma=gamma, field_size=cfg["env"]["field_size"])

    ev = cfg["eval"]
    # a fresh env per seed gives each seed's history-free start state 
    starts = [LBFEnv(**cfg["env"]).reset(seed=s)[0] for s in range(ev["seed_start"], ev["seed_start"] + ev["episodes"])]
    opt_len = float(np.mean([optimal_steps(V, s, gamma) for s in starts]))

    rows = []
    for run in runs:
        final_len = pd.read_csv(run / "eval_metrics.csv")["length"].iloc[-1]
        row = {"run": run.name, "final_greedy_length": final_len, "eval_regret_steps": final_len - opt_len}
        if (run / "q_table.npz").exists():
            row["frac_states_optimal"], row["n_states_unvisited"] = state_optimality(run / "q_table.npz", states, Q)
        rows.append(row)
    table = pd.DataFrame(rows)

    print(f"optimal mean length on the {ev['episodes']} eval start states: {opt_len:.2f}")
    for col in table.columns[1:]:
        mean, half = mean_ci(table[col].to_numpy(dtype=float)[:, None])
        print(f"{col}: {mean[0]:.3f} ± {half[0]:.3f} (95% CI, n = {len(table)})")

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / f"regret_{runs[0].parent.name}_{runs[0].name.split('_')[0]}.csv"
    table.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
