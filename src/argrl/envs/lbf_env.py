"""Single-agent wrapper around lbforaging

References:
- Level-Based Foraging environment: Christianos, Schäfer & Albrecht (2020), NeurIPS; Papoudakis et al. (2021), NeurIPS Datasets & Benchmarks.
- Time limit treated as truncation, not a terminal state: Pardo et al. (2018), Time Limits in Reinforcement Learning, ICML.
"""

from __future__ import annotations
import numpy as np
from lbforaging.foraging import ForagingEnv

State = tuple[int, ...]

# turn one agent's observation vector into a tuple of ints 
def encode_state(obs: np.ndarray) -> State:
    return tuple(int(x) for x in obs)

# One-agent LBF
class LBFEnv:
    
    n_actions = 6 # theres 6 actions : none, north, south, west, east, load

    def __init__(
        self,
        field_size: int = 5,
        max_num_food: int = 1,
        min_player_level: int = 1,
        max_player_level: int = 2,
        max_episode_steps: int = 50,
        sight: int | None = None,
    ) -> None:
        self.max_episode_steps = max_episode_steps
        self._env = ForagingEnv(
            players=1,
            min_player_level=min_player_level,
            max_player_level=max_player_level,
            min_food_level=1,
            max_food_level=None,  
            field_size=(field_size, field_size),
            max_num_food=max_num_food,
            sight=field_size if sight is None else sight,
            max_episode_steps=max_episode_steps,
            force_coop=False,
        )
        self._n_food_spawned = 0
    
    # start episode
    def reset(self, seed: int | None = None) -> tuple[State, dict]:
        # lbforaging's spawn treats the last episode's end position as occupied, so clear it (D21)
        for p in self._env.players:
            p.position = None
        obs, _ = self._env.reset(seed=seed)
        self._n_food_spawned = int(np.count_nonzero(self._env.field))
        return encode_state(obs[0]), self._info(0.0)

    # returns : state, reward, terminated, truncated, info
    def step(self, action: int) -> tuple[State, float, bool, bool, dict]:
        obs, rewards, _done, _truncated, _ = self._env.step([action])
        # lbforaging 2.0.0 reports the time limit as done=True, so we ignore its flags and work them out here
        terminated = bool(self._env.field.sum() == 0)
        truncated = not terminated and self._env.current_step >= self.max_episode_steps
        reward = float(rewards[0])
        return encode_state(obs[0]), reward, terminated, truncated, self._info(reward)

    def _info(self, reward: float) -> dict:
        food_left = np.count_nonzero(self._env.field)
        return {"env_reward": reward, "food_collected_frac": 1 - food_left / self._n_food_spawned,"step": self._env.current_step,}
