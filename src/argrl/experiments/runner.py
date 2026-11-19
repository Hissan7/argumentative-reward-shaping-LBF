"""Train and evaluate one agent over all seeds of a config, and write the run folders."""

from __future__ import annotations
import json
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from argrl.envs.lbf_env import LBFEnv
from argrl.experiments.config import ExperimentConfig, dump_config
from argrl.experiments.registry import AGENTS

RESULTS = Path("results/raw")


def run_episode(env: LBFEnv, agent, seed: int | None = None, learn: bool = True) -> dict:
    s, info = env.reset(seed=seed)
    env_return, length = 0.0, 0
    while True:
        a = agent.select_action(s, greedy=not learn)
        s_next, r, terminated, truncated, info = env.step(a)
        if learn:
            agent.update(s, a, r, s_next, terminated)  # truncation still bootstraps (D4)
        env_return = env_return + info["env_reward"]
        length = length + 1
        s = s_next
        if terminated or truncated:
            break
    if learn:
        agent.end_episode()
    return {
        "env_return": env_return,
        "length": length,
        "terminated": terminated,
        "truncated": truncated,
        "food_collected_frac": info["food_collected_frac"],
    }


def git_state() -> tuple[str, bool]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)
    return commit.stdout.strip(), bool(status.stdout.strip())


# returns training metrics, eval metrics, extra meta fields and the trained agent
def train_seed(cfg: ExperimentConfig, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict, object]:
    # separate env and agent seeds so their random streams never coincide (D10)
    env_seed, agent_seed = (int(x) for x in np.random.SeedSequence(seed).generate_state(2))
    env, eval_env = LBFEnv(**asdict(cfg.env)), LBFEnv(**asdict(cfg.env))
    agent = AGENTS[cfg.agent.type](n_actions=env.n_actions, seed=agent_seed, **cfg.agent.params)
    ev = cfg.eval
    eval_seeds = range(ev.seed_start, ev.seed_start + ev.episodes)

    rows, eval_rows = [], []
    for ep in range(cfg.n_episodes + 1):
        if ep % ev.every == 0 or ep == cfg.n_episodes:
            runs = pd.DataFrame(run_episode(eval_env, agent, s, learn=False) for s in eval_seeds)
            means = runs[["env_return", "length", "food_collected_frac"]].mean()
            eval_rows.append({"episode": ep, **means.to_dict()})
        if ep == cfg.n_episodes:
            break
        epsilon = agent.epsilon  # the value used during this episode
        row = run_episode(env, agent, seed=env_seed if ep == 0 else None)
        rows.append({"episode": ep, **row, "epsilon": epsilon})

    extra = {"env_seed": env_seed, "agent_seed": agent_seed}
    if hasattr(agent, "q"):
        extra["n_states_in_q_table"] = len(agent.q)  # states seen in training or evaluation
    return pd.DataFrame(rows), pd.DataFrame(eval_rows), extra, agent


def save_q_table(agent, path: Path) -> None:
    np.savez(path, states=np.array(list(agent.q.keys())), q=np.stack(list(agent.q.values())))


def run_experiment(cfg: ExperimentConfig, config_file: str, results: Path = RESULTS) -> list[Path]:
    commit, dirty = git_state()
    if dirty:
        print("WARNING: uncommitted changes; these runs are marked git_dirty = true.")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_dirs = []
    for seed in cfg.seeds:
        run_dir = results / cfg.name / f"{timestamp}_{seed}"
        run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite results
        t0 = time.perf_counter()
        metrics, eval_metrics, extra, agent = train_seed(cfg, seed)
        meta = {
            "git_commit": commit,
            "git_dirty": dirty,
            "seed": seed,
            **extra,
            "timestamp": timestamp,
            "config_file": config_file,
            "python": platform.python_version(),
            "packages": {p: version(p) for p in ("lbforaging", "numpy", "pandas")},
            "wall_clock_s": round(time.perf_counter() - t0, 2),
        }
        (run_dir / "config.yaml").write_text(dump_config(cfg))
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        eval_metrics.to_csv(run_dir / "eval_metrics.csv", index=False)
        if hasattr(agent, "q"):
            save_q_table(agent, run_dir / "q_table.npz")
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        last = eval_metrics.iloc[-1]
        print(
            f"{cfg.name} seed {seed}: {meta['wall_clock_s']}s | final greedy eval: "
            f"return {last.env_return:.2f}, length {last.length:.1f} -> {run_dir}"
        )
        run_dirs.append(run_dir)
    return run_dirs
