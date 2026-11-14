"""Train one agent on single-player LBF"""

from __future__ import annotations
import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argrl.agents.q_learning import QLearningAgent
from argrl.agents.random_agent import RandomAgent
from argrl.envs.lbf_env import LBFEnv

AGENTS = {"q_learning": QLearningAgent, "random": RandomAgent}
RESULTS = Path("results/raw")

# play an episode
def run_episode(env: LBFEnv, agent, seed: int | None = None, learn: bool = True) -> dict:
    s, info = env.reset(seed=seed)
    env_return, length = 0.0, 0
    while True:
        a = agent.select_action(s, greedy=not learn)
        s_next, r, terminated, truncated, info = env.step(a)
        if learn:
            agent.update(s, a, r, s_next, terminated)  # truncation still bootstraps
        env_return = env_return + info["env_reward"]
        length = length + 1
        s = s_next
        if terminated or truncated:
            break
    if learn:
        agent.end_episode()
    return {"env_return": env_return,"length": length,"terminated": terminated,"truncated": truncated,"food_collected_frac": info["food_collected_frac"],}


def git_state() -> tuple[str, bool]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)
    return commit.stdout.strip(), bool(status.stdout.strip())

# train one agent for one run seed
def train_seed(cfg: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    env_seed, agent_seed = (int(x) for x in np.random.SeedSequence(seed).generate_state(2)) # make seperate seeds so  env and agent streams never coincide.
    env, eval_env = LBFEnv(**cfg["env"]), LBFEnv(**cfg["env"])
    params = {k: v for k, v in cfg["agent"].items() if k != "type"}
    agent = AGENTS[cfg["agent"]["type"]](n_actions=env.n_actions, seed=agent_seed, **params)
    ev = cfg["eval"]
    eval_seeds = range(ev["seed_start"], ev["seed_start"] + ev["episodes"])

    rows, eval_rows = [], []
    for ep in range(cfg["n_episodes"] + 1):
        if ep % ev["every"] == 0 or ep == cfg["n_episodes"]:
            runs = pd.DataFrame(run_episode(eval_env, agent, s, learn=False) for s in eval_seeds)
            means = runs[["env_return", "length", "food_collected_frac"]].mean()
            eval_rows.append({"episode": ep, **means.to_dict()})
        if ep == cfg["n_episodes"]:
            break
        epsilon = agent.epsilon  # the value used during this episode
        row = run_episode(env, agent, seed=env_seed if ep == 0 else None)
        rows.append({"episode": ep, **row, "epsilon": epsilon})

    extra = {"env_seed": env_seed, "agent_seed": agent_seed}
    if hasattr(agent, "q"):
        extra["n_states_in_q_table"] = len(agent.q)  # states seen in training or evaluation
    return pd.DataFrame(rows), pd.DataFrame(eval_rows), extra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    commit, dirty = git_state()
    if dirty:
        print("there are uncommited changes.")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    for seed in cfg["seeds"]:
        run_dir = RESULTS / cfg["name"] / f"{timestamp}_{seed}"
        run_dir.mkdir(parents=True, exist_ok=False)  # making sure results aren't overwritten
        t0 = time.perf_counter()
        metrics, eval_metrics, extra = train_seed(cfg, seed)
        meta = {
            "git_commit": commit,
            "git_dirty": dirty,
            "seed": seed,
            **extra,
            "timestamp": timestamp,
            "config_file": str(args.config),
            "python": platform.python_version(),
            "packages": {p: version(p) for p in ("lbforaging", "numpy", "pandas")},
            "wall_clock_s": round(time.perf_counter() - t0, 2),
        }
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        eval_metrics.to_csv(run_dir / "eval_metrics.csv", index=False)
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        last = eval_metrics.iloc[-1]
        print(
            f"{cfg['name']} seed {seed}: {meta['wall_clock_s']}s | final greedy eval: "
            f"return {last.env_return:.2f}, length {last.length:.1f} -> {run_dir}"
        )


if __name__ == "__main__":
    main()
