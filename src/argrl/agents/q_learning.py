"""Tabular epsilon-greedy Q-learning. This is the the no-shaping baseline that ArgRS will extend"""

from __future__ import annotations
from collections import defaultdict
from collections.abc import Hashable
import numpy as np


class QLearningAgent:

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 1000,
        q_init: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.epsilon = epsilon_start
        self.rng = np.random.default_rng(seed)
        self.q: defaultdict[Hashable, np.ndarray] = defaultdict(
            lambda: np.full(n_actions, q_init, dtype=float)
        )
        self._episodes_done = 0

    # Select epsilon-greedy or purely greedy action. 
    def select_action(self, state: Hashable, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.q[state]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    # One Q-learning update
    def update(
        self,
        s: Hashable,
        a: int,
        r: float,
        s_next: Hashable,
        terminated: bool,
        shaping: float = 0.0,
    ) -> float:
        bootstrap = 0.0 if terminated else self.gamma * self.q[s_next].max()
        td_error = r + shaping + bootstrap - self.q[s][a]
        self.q[s][a] = self.q[s][a] + self.alpha * td_error
        return float(td_error)

    def end_episode(self) -> None:
        self._episodes_done = self._episodes_done + 1
        frac = min(self._episodes_done / self.epsilon_decay_episodes, 1.0)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)
