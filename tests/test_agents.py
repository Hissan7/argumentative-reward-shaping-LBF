import numpy as np
import pytest
from argrl.agents.q_learning import QLearningAgent
from argrl.agents.random_agent import RandomAgent


def toy_agent() -> QLearningAgent:
    agent = QLearningAgent(n_actions=6, alpha=0.5, gamma=0.9)
    agent.q["s_next"] = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    return agent


@pytest.mark.parametrize(("terminated", "expected"), [(False, 1.4), (True, 0.5)])
def test_q_update_toy(terminated, expected):
    agent = toy_agent()
    agent.update("s", 0, r=1.0, s_next="s_next", terminated=terminated)
    assert agent.q["s"][0] == pytest.approx(expected)


def test_shaping_hook_adds_exactly_to_target():
    plain, shaped = toy_agent(), toy_agent()
    td_plain = plain.update("s", 0, 1.0, "s_next", terminated=False)
    td_shaped = shaped.update("s", 0, 1.0, "s_next", terminated=False, shaping=0.3)
    assert td_shaped - td_plain == pytest.approx(0.3)


def test_epsilon_decays_linearly_then_stays():
    agent = QLearningAgent(6, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_episodes=10)
    for _ in range(5):
        agent.end_episode()
    assert agent.epsilon == pytest.approx(0.55)
    for _ in range(20):
        agent.end_episode()
    assert agent.epsilon == pytest.approx(0.1)


def test_greedy_ties_broken_at_random():
    agent = QLearningAgent(6, seed=0)
    assert len({agent.select_action("unseen", greedy=True) for _ in range(100)}) == 6


def test_random_agent_reproducible():
    a, b = RandomAgent(6, seed=3), RandomAgent(6, seed=3)
    assert [a.select_action(None) for _ in range(50)] == [
        b.select_action(None) for _ in range(50)
    ]
