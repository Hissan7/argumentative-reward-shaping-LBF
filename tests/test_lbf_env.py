import numpy as np
from argrl.agents.heuristic_agent import LOAD, NONE, walk_to_food
from argrl.envs.lbf_env import LBFEnv


def test_truncation_flags():
    env = LBFEnv(max_episode_steps=50)
    env.reset(seed=0)
    for t in range(1, 51):
        _, _, terminated, truncated, _ = env.step(NONE)
        assert terminated is False
        assert truncated is (t == 50)


def test_termination_flags():
    env = LBFEnv()
    s, info = env.reset(seed=0)
    assert info["food_collected_frac"] == 0
    for _ in range(50):
        s, r, terminated, truncated, info = env.step(walk_to_food(s))
        if terminated:
            break
    assert terminated and not truncated
    assert r == 1.0 == info["env_reward"]  
    assert info["food_collected_frac"] == 1
    assert s[:3] == (-1, -1, 0)  # collected food is encoded as (-1, -1, 0)


def test_load_on_last_step_is_termination_not_truncation():
    env = LBFEnv(max_episode_steps=50)
    s, _ = env.reset(seed=0)
    while (a := walk_to_food(s)) != LOAD:
        s, *_ = env.step(a)
    assert env._env.current_step < 49
    while env._env.current_step < 49:
        s, *_ = env.step(NONE)
    # at step 50, the time limit and the food both end the episode.
    _, _, terminated, truncated, _ = env.step(LOAD)
    assert terminated and not truncated


def test_state_is_hashable_and_deterministic():
    def rollout(seed: int) -> list[tuple[int, ...]]:
        env, rng = LBFEnv(), np.random.default_rng(123)
        states = [env.reset(seed=seed)[0]]
        for _ in range(3):  # later resets continue the seeded stream
            for _ in range(20):
                s, _, terminated, truncated, _ = env.step(int(rng.integers(6)))
                states.append(s)
                if terminated or truncated:
                    break
            states.append(env.reset()[0])
        return states

    a, b = rollout(7), rollout(7)
    assert a == b
    assert all(isinstance(x, int) for x in a[0]) and len(a[0]) == 6
    assert len({hash(s) for s in a}) > 1


# regression test for D21: the start state depends on the seed only, not on the previous episode
def test_same_seed_same_start_regardless_of_history():
    eval_seeds = range(1_000_000, 1_000_020)
    fresh = [LBFEnv().reset(seed=s)[0] for s in eval_seeds]
    env, rng = LBFEnv(), np.random.default_rng(0)
    for prior in range(200):
        env.reset(seed=prior)
        for _ in range(int(rng.integers(1, 50))):  # end the previous episode somewhere different each time
            _, _, terminated, truncated, _ = env.step(int(rng.integers(6)))
            if terminated or truncated:
                break
        i = prior % len(fresh)
        assert env.reset(seed=eval_seeds[i])[0] == fresh[i]
