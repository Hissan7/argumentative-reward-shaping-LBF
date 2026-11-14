"""Run Level-Based Foraging with random agents and render it."""

from __future__ import annotations
import argparse
import time
import gymnasium as gym
import lbforaging  
from lbforaging.foraging import ForagingEnv

# a 5x5 grid with 1 agent and 1 food item for first tabular Q-learning baseline
def make_single_player_env() -> ForagingEnv:
    return ForagingEnv(
        players=1,
        min_player_level=1,
        max_player_level=2,
        min_food_level=1,
        max_food_level=None,  # it is None because food level comes from the player levels
        field_size=(5, 5),
        max_num_food=1,
        sight=5,
        max_episode_steps=50,
        force_coop=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="Foraging-8x8-2p-1f-v3")
    parser.add_argument(
        "--single", action="store_true", help="use the 1-player 5x5 env"
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()

    render = not args.no_render
    if args.single:
        env = make_single_player_env()
    else:
        env = gym.make(args.env_id).unwrapped  

    n_agents = len(env.action_space)
    print(f"Env: {'single 5x5' if args.single else args.env_id} | agents: {n_agents}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        done, steps, totals = False, 0, [0.0] * n_agents
        if render:
            env.render()  # first frame drawn after the players have been placed
            time.sleep(1.0 / args.fps)
        while not done:
            actions = env.action_space.sample()  # one random action per agent
            _obs, rewards, done, truncated, _info = env.step(actions)
            done = done or truncated
            totals = [t + r for t, r in zip(totals, rewards)]
            steps = steps + 1
            if render:
                env.render()
                time.sleep(1.0 / args.fps)
        print(f"Episode {ep + 1}: {steps} steps, rewards per agent = {[round(t, 3) for t in totals]}")

    try:
        env.close()
    except AttributeError:
        pass


if __name__ == "__main__":
    main()
