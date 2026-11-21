"""Exact optimal values and policy for 1-player 1-food LBF by value iteration.

References:
- Value iteration: Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., Sec. 4.4.
"""

from __future__ import annotations
import numpy as np
from argrl.envs.lbf_env import State

NONE, NORTH, SOUTH, WEST, EAST, LOAD = range(6)
MOVES = {NORTH: (-1, 0), SOUTH: (1, 0), WEST: (0, -1), EAST: (0, 1)}


def enumerate_states(field_size: int = 5, max_player_level: int = 2) -> list[State]:
    inner = range(1, field_size - 1)  # food never spawns on the border
    return [
        (fr, fc, fl, ar, ac, al)
        for fr in inner
        for fc in inner
        for al in range(1, max_player_level + 1)
        for fl in range(1, al + 1)
        for ar in range(field_size)
        for ac in range(field_size)
        if (ar, ac) != (fr, fc)
    ]


# deterministic model: (s_next, reward, terminated)
def transition(s: State, a: int, field_size: int = 5) -> tuple[State, float, bool]:
    fr, fc, fl, ar, ac, al = s
    if a == LOAD:
        if abs(fr - ar) + abs(fc - ac) == 1 and al >= fl:
            return (-1, -1, 0, ar, ac, al), 1.0, True
        return s, 0.0, False
    if a in MOVES:
        r, c = ar + MOVES[a][0], ac + MOVES[a][1]
        if 0 <= r < field_size and 0 <= c < field_size and (r, c) != (fr, fc):
            return (fr, fc, fl, r, c, al), 0.0, False
    return s, 0.0, False  # NONE, or a blocked move 


def value_iteration(states: list[State],n_actions: int = 6,gamma: float = 0.99,tol: float = 1e-10,max_iter: int = 10_000,field_size: int = 5,) -> tuple[dict[State, float], dict[State, np.ndarray]]:
    model = {s: [transition(s, a, field_size) for a in range(n_actions)] for s in states}
    V = dict.fromkeys(states, 0.0)
    for _ in range(max_iter):
        Q = {s: np.array([r + (0.0 if term else gamma * V[s2]) for s2, r, term in model[s]]) for s in states}
        V_new = {s: float(q.max()) for s, q in Q.items()}
        residual = max(abs(V_new[s] - V[s]) for s in states)
        V = V_new
        if residual < tol:
            return V, Q
    raise RuntimeError(f"value iteration did not converge in {max_iter} iterations")


def optimal_actions(Q_star: dict[State, np.ndarray], s: State, atol: float = 1e-9) -> set[int]:
    q = Q_star[s]
    return set(np.flatnonzero(q >= q.max() - atol).tolist())


def optimal_steps(V_star: dict[State, float], s: State, gamma: float = 0.99) -> int:
    return 1 + round(np.log(V_star[s]) / np.log(gamma))
