"""Number of non-terminal tabular states for LBF with full sight (D12).

A state is the remaining food (positions, levels) plus the players (positions, levels).
Any subset of a valid food layout is itself valid, so partly-collected states are the
layouts with fewer items. Food levels go from 1 to the sum of the player levels
(lbforaging 2.0.0 with max_food_level=None and up to 3 players).
"""

from __future__ import annotations
from itertools import combinations, product
from math import perm


# lbforaging spawn_food: no other food in the 3x3 neighbourhood or 2 cells away in the same row/column
def _conflict(a: tuple[int, int], b: tuple[int, int]) -> bool:
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dr, dc) <= 1 or (dr == 0 and dc <= 2) or (dc == 0 and dr <= 2)


def food_layouts(field_size: int, n_food: int) -> int:
    inner = [(r, c) for r in range(1, field_size - 1) for c in range(1, field_size - 1)]
    return sum(
        all(not _conflict(a, b) for a, b in combinations(cells, 2))
        for cells in combinations(inner, n_food)
    )


def count_states(field_size: int, max_num_food: int, n_players: int, max_player_level: int) -> int:
    cells = field_size**2
    total = 0
    for j in range(1, max_num_food + 1):
        layouts = food_layouts(field_size, j)
        level_combos = sum(sum(lv) ** j for lv in product(range(1, max_player_level + 1), repeat=n_players))
        total += layouts * level_combos * perm(cells - j, n_players)  # players on distinct non-food cells
    return total
