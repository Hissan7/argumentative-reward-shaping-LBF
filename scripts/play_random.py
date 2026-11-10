"""Sanity check: run Level-Based Foraging with random agents and render it.

Usage:
    uv run python scripts/play_random.py
    uv run python scripts/play_random.py --episodes 5 --no-render

Notes on lbforaging 2.0.0, checked against the installed source:
- step() returns (obs, rewards, done, truncated, info). `rewards` is a list with one entry per agent.
- The time limit is reported as done=True and truncated is always False. The RL code must
  handle this (see CLAUDE.md, section 6).
- Rendering is not automatic on step(): env.render() has to be called every step.
- Registered IDs only go from 2 to 9 players. For a single player, build ForagingEnv directly
  (see make_single_player_env below).
- BUG in 2.0.0: with render_mode="human", reset() tries to draw the players before it has
  placed them, which raises TypeError. So we never pass render_mode. We call env.render()
  ourselves after reset() and after every step() instead.
"""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
import lbforaging  # noqa: F401  (importing registers the Foraging-* env IDs)
from lbforaging.foraging import ForagingEnv


def make_single_player_env() -> ForagingEnv:
    """A 5x5 grid with one agent and one food item, for the first tabular Q-learning baseline."""
    return ForagingEnv(
        players=1,
        min_player_level=1,
        max_player_level=2,
        min_food_level=1,
        max_food_level=None,  # None: food level comes from the player levels
        field_size=(5, 5),
        max_num_food=1,
        sight=5,
        max_episode_steps=50,
        force_coop=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="Foraging-8x8-2p-1f-v3")
    parser.add_argument("--single", action="store_true", help="use the 1-player 5x5 env")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()

    render = not args.no_render
    if args.single:
        env = make_single_player_env()
    else:
        env = gym.make(args.env_id).unwrapped  # no render_mode: see the bug note at the top

    n_agents = len(env.action_space)
    print(f"Env: {'single 5x5' if args.single else args.env_id} | agents: {n_agents}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done, steps, totals = False, 0, [0.0] * n_agents
        if render:
            env.render()  # first frame, drawn after the players have been placed
            time.sleep(1.0 / args.fps)
        while not done:
            actions = env.action_space.sample()  # one random action per agent
            obs, rewards, done, truncated, info = env.step(actions)
            done = done or truncated
            totals = [t + r for t, r in zip(totals, rewards)]
            steps += 1
            if render:
                env.render()
                time.sleep(1.0 / args.fps)
        print(f"Episode {ep + 1}: {steps} steps, rewards per agent = {[round(t, 3) for t in totals]}")

    try:
        env.close()
    except AttributeError:
        # pyglet 1.5 on macOS raises this when the window closes. It is harmless:
        # all episodes have already finished.
        pass


if __name__ == "__main__":
    main()