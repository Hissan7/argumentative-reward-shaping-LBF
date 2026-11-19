import numpy as np
import pytest
from argrl.agents.value_iteration import (
    LOAD,
    enumerate_states,
    optimal_actions,
    optimal_steps,
    transition,
    value_iteration,
)
from argrl.agents.heuristic_agent import walk_to_food
from argrl.envs.lbf_env import LBFEnv
from argrl.envs.state_space import count_states

GAMMA = 0.99


@pytest.fixture(scope="module")
def solved():
    states = enumerate_states()
    V, Q = value_iteration(states, gamma=GAMMA)
    return states, V, Q


def test_state_count():
    assert len(enumerate_states()) == 648


def test_model_matches_env():
    env, rng = LBFEnv(), np.random.default_rng(0)
    seen = set()
    for ep in range(3000):
        s, _ = env.reset(seed=ep)
        while True:
            a = int(rng.integers(6))
            s_next, r, terminated, truncated, _ = env.step(a)
            assert transition(s, a) == (s_next, r, terminated)
            seen.add((s, a))
            s = s_next
            if terminated or truncated:
                break
    assert len(seen) == 648 * 6  # every (state, action) pair was checked


def test_bellman_residual(solved):
    states, V, _ = solved
    for s in states:
        backups = []
        for a in range(6):
            s_next, r, terminated = transition(s, a)
            backups.append(r + (0.0 if terminated else GAMMA * V[s_next]))
        assert abs(V[s] - max(backups)) < 1e-9


@pytest.mark.parametrize(
    ("state", "k"),
    [
        ((2, 2, 1, 2, 1, 1), 1),  # next to the food: LOAD now
        ((2, 2, 1, 2, 0, 1), 2),
        ((2, 2, 1, 0, 0, 2), 4),  # corner: 3 moves to (1, 2) or (2, 1), then LOAD
        ((1, 1, 2, 4, 4, 2), 6),
    ],
)
def test_value_matches_distance(solved, state, k):
    _, V, Q = solved
    assert V[state] == pytest.approx(GAMMA ** (k - 1))
    assert optimal_steps(V, state, GAMMA) == k
    assert (LOAD in optimal_actions(Q, state)) == (k == 1)


# a fresh env per seed: lbforaging's reset depends on where the last episode ended (D21)
def test_eval_seed_optimum_matches_heuristic(solved):
    _, V, _ = solved
    for seed in range(1_000_000, 1_000_020):
        env = LBFEnv()
        s, _ = env.reset(seed=seed)
        k, steps = optimal_steps(V, s, GAMMA), 0
        while True:
            s, _, terminated, truncated, _ = env.step(walk_to_food(s))
            steps += 1
            if terminated or truncated:
                break
        assert terminated and steps == k


def test_state_space_formula_matches_enumeration():
    assert count_states(5, 1, 1, 2) == len(enumerate_states()) == 648
