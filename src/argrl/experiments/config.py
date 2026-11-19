"""Experiment configs: YAML <-> dataclasses. Unknown or missing keys raise an error."""

from __future__ import annotations
import inspect
from dataclasses import asdict, dataclass, field
from pathlib import Path
import yaml
from argrl.experiments.registry import AGENTS


@dataclass
class EnvConfig:
    field_size: int
    max_num_food: int
    min_player_level: int
    max_player_level: int
    sight: int
    max_episode_steps: int


@dataclass
class AgentConfig:
    type: str
    params: dict = field(default_factory=dict)


@dataclass
class EvalConfig:
    every: int
    episodes: int
    seed_start: int


@dataclass
class ExperimentConfig:
    name: str
    env: EnvConfig
    agent: AgentConfig
    n_episodes: int
    seeds: list[int]
    eval: EvalConfig


def from_dict(d: dict) -> ExperimentConfig:
    d = dict(d)
    agent = dict(d.pop("agent"))
    agent_type = agent.pop("type")
    if agent_type not in AGENTS:
        raise ValueError(f"unknown agent type {agent_type!r}; known: {sorted(AGENTS)}")
    allowed = set(inspect.signature(AGENTS[agent_type]).parameters) - {"n_actions", "seed"}
    if unknown := set(agent) - allowed:
        raise TypeError(f"unknown {agent_type} parameters: {sorted(unknown)}")
    return ExperimentConfig(
        env=EnvConfig(**d.pop("env")),
        agent=AgentConfig(agent_type, agent),
        eval=EvalConfig(**d.pop("eval")),
        **d,
    )


def to_dict(cfg: ExperimentConfig) -> dict:
    d = asdict(cfg)
    d["agent"] = {"type": cfg.agent.type, **cfg.agent.params}  # same flat layout as the YAML
    return d


def load_config(path: Path) -> ExperimentConfig:
    return from_dict(yaml.safe_load(Path(path).read_text()))


def dump_config(cfg: ExperimentConfig) -> str:
    return yaml.safe_dump(to_dict(cfg), sort_keys=False)
