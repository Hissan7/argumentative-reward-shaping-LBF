"""This is the uniform random agent"""

from __future__ import annotations
from collections.abc import Hashable
import numpy as np

# picks actions uniformly at random
class RandomAgent:
    epsilon = 1.0  # always explores

    def __init__(self, n_actions: int, seed: int | None = None) -> None:
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: Hashable, greedy: bool = False) -> int:
        return int(self.rng.integers(self.n_actions))

    def update(self, *args: object, **kwargs: object) -> float:
        return 0.0

    def end_episode(self) -> None:
        pass
