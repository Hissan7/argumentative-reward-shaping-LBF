"""Agent type name (the `agent.type` in a config) -> agent class."""

from argrl.agents.heuristic_agent import HeuristicAgent
from argrl.agents.q_learning import QLearningAgent
from argrl.agents.random_agent import RandomAgent

AGENTS = {"random": RandomAgent, "q_learning": QLearningAgent, "heuristic": HeuristicAgent}
