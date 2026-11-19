from argrl.agents.heuristic_agent import LOAD, MOVES, HeuristicAgent, good_actions
from argrl.agents.value_iteration import enumerate_states, optimal_actions, value_iteration
from argrl.envs.lbf_env import LBFEnv


def test_loads_when_adjacent():
    for ar, ac in [(1, 2), (3, 2), (2, 1), (2, 3)]:
        assert good_actions((2, 2, 1, ar, ac, 1)) == [LOAD]


def test_never_steps_into_food():
    for s in enumerate_states():
        for a in good_actions(s):
            if a in MOVES:
                assert (s[3] + MOVES[a][0], s[4] + MOVES[a][1]) != (s[0], s[1])


def test_heuristic_is_optimal_everywhere():
    states = enumerate_states()
    _, Q = value_iteration(states)
    for s in states:
        assert set(good_actions(s)) <= optimal_actions(Q, s), s


def test_heuristic_reproducible():
    def rollout(seed: int) -> list[int]:
        env, agent = LBFEnv(), HeuristicAgent(seed=seed)
        actions = []
        for ep in range(5):
            s, _ = env.reset(seed=ep)
            while True:
                a = agent.select_action(s)
                actions.append(a)
                s, _, terminated, truncated, _ = env.step(a)
                if terminated or truncated:
                    break
        return actions

    assert rollout(3) == rollout(3)
