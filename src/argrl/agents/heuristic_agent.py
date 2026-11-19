"""Hand-coded heuristic: walk to the nearest cell next to the food, then LOAD."""

from __future__ import annotations
import numpy as np
from argrl.envs.lbf_env import State

NONE, NORTH, SOUTH, WEST, EAST, LOAD = range(6)
MOVES = {NORTH: (-1, 0), SOUTH: (1, 0), WEST: (0, -1), EAST: (0, 1)}


# TODO: assumes the 1-food encoding; with more food or players, "nearest loadable" needs the levels
def good_actions(state: State) -> list[int]:
    fr, fc, _, ar, ac, _ = state
    targets = [(fr + dr, fc + dc) for dr, dc in MOVES.values()]

    def dist(r: int, c: int) -> int:
        return min(abs(r - tr) + abs(c - tc) for tr, tc in targets)

    d = dist(ar, ac)
    if d == 0:
        return [LOAD]
    return [
        a
        for a, (dr, dc) in MOVES.items()
        if (ar + dr, ac + dc) != (fr, fc)
        and dist(ar + dr, ac + dc) == d - 1
    ]


# one action from good_actions: the first (rows before columns) or a random one
def walk_to_food(state: State, rng: np.random.Generator | None = None) -> int:
    actions = good_actions(state)
    return actions[0] if rng is None else int(rng.choice(actions))


class HeuristicAgent:
    epsilon = 0.0

    def __init__(self, n_actions: int = 6, seed: int | None = None, tie_break: str = "rng") -> None:
        if tie_break not in ("rng", "fixed"):
            raise ValueError(f"tie_break must be 'rng' or 'fixed', got {tie_break!r}")
        self.n_actions = n_actions
        self.tie_break = tie_break
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: State, greedy: bool = False) -> int:
        return walk_to_food(state, self.rng if self.tie_break == "rng" else None)

    def update(self, *args: object, **kwargs: object) -> float:
        return 0.0

    def end_episode(self) -> None:
        pass
