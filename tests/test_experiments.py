import dataclasses
from pathlib import Path
import numpy as np
import pytest
import yaml
from argrl.experiments.config import dump_config, from_dict, load_config
from argrl.experiments.registry import AGENTS
from argrl.experiments.runner import run_experiment

CONFIGS = Path("experiments/configs")


def raw(name: str = "baseline_5x5") -> dict:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


@pytest.mark.parametrize("name", ["baseline_5x5", "random_5x5", "heuristic_5x5"])
def test_config_roundtrip(name):
    cfg = load_config(CONFIGS / f"{name}.yaml")
    assert from_dict(yaml.safe_load(dump_config(cfg))) == cfg


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(n_episode=10),  # typo at the top level
        lambda d: d["env"].update(feild_size=5),
        lambda d: d["eval"].update(evrey=50),
        lambda d: d["agent"].update(aplha=0.1),
        lambda d: d["agent"].update(type="dqn"),
        lambda d: d.pop("seeds"),  # missing key
    ],
)
def test_unknown_or_missing_key_rejected(mutate):
    d = raw()
    mutate(d)
    with pytest.raises((TypeError, ValueError)):
        from_dict(d)


def test_registry():
    for cls in AGENTS.values():
        agent = cls(n_actions=6, seed=0)
        assert 0 <= agent.select_action((2, 2, 1, 0, 0, 1)) < 6


def test_short_run_deterministic(tmp_path):
    cfg = load_config(CONFIGS / "baseline_5x5.yaml")
    cfg = dataclasses.replace(cfg, n_episodes=20, seeds=[0], eval=dataclasses.replace(cfg.eval, every=10))
    a, b = (run_experiment(cfg, "test", results=tmp_path / x)[0] for x in "ab")
    for f in ("metrics.csv", "eval_metrics.csv", "config.yaml"):
        assert (a / f).read_bytes() == (b / f).read_bytes()
    qa, qb = np.load(a / "q_table.npz"), np.load(b / "q_table.npz")
    assert (qa["states"] == qb["states"]).all() and (qa["q"] == qb["q"]).all()
