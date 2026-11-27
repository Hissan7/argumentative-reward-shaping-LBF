"""Bipolar argumentation framework: arguments with base scores, attacks and supports.

Attacks as in Dung (1995); supports and base scores as in Rago, Čyras & Toni (2016).
Knows nothing about LBF (CLAUDE.md §5).
"""

from __future__ import annotations
import heapq
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Argument:
    name: str
    base_score: float = 1.0  # used by DF-QuAD only
    description: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.base_score <= 1.0:  # also rejects NaN
            raise ValueError(f"base score of {self.name!r} must be in [0, 1], got {self.base_score}")


class ArgumentationFramework:
    """Immutable, validated framework. Relations are (source, target) pairs of argument names."""

    __slots__ = ("_arguments", "_attacks", "_supports", "_attackers", "_supporters", "_order")

    def __init__(
        self,
        arguments: Iterable[Argument],
        attacks: Iterable[tuple[str, str]] = (),
        supports: Iterable[tuple[str, str]] = (),
    ) -> None:
        args: dict[str, Argument] = {}
        for a in arguments:
            if a.name in args:
                raise ValueError(f"duplicate argument name {a.name!r}")
            args[a.name] = a
        attacks, supports = frozenset(attacks), frozenset(supports)
        for kind, rel in (("attack", attacks), ("support", supports)):
            for src, dst in rel:
                if src not in args or dst not in args:
                    raise ValueError(f"{kind} ({src!r}, {dst!r}) names an unknown argument")
        if both := attacks & supports:
            raise ValueError(f"pairs both attack and support: {sorted(both)}")
        if selfs := sorted(s for s, t in supports if s == t):
            raise ValueError(f"self-support is not allowed: {selfs}")  # self-attack is allowed (Dung 1995)

        self._arguments = MappingProxyType(args)
        self._attacks, self._supports = attacks, supports
        self._attackers = {n: frozenset(s for s, t in attacks if t == n) for n in args}
        self._supporters = {n: frozenset(s for s, t in supports if t == n) for n in args}
        self._order = self._leaves_first()

    def _leaves_first(self) -> tuple[str, ...] | None:
        # Kahn's algorithm with ties broken by name, so the order is deterministic; None if cyclic
        parents = {n: self._attackers[n] | self._supporters[n] for n in self._arguments}
        children: dict[str, list[str]] = {n: [] for n in self._arguments}
        for n, ps in parents.items():
            for p in ps:
                children[p].append(n)
        waiting = {n: len(ps) for n, ps in parents.items()}
        ready = [n for n, k in waiting.items() if k == 0]
        heapq.heapify(ready)
        order = []
        while ready:
            n = heapq.heappop(ready)
            order.append(n)
            for c in children[n]:
                waiting[c] -= 1
                if waiting[c] == 0:
                    heapq.heappush(ready, c)
        return tuple(order) if len(order) == len(self._arguments) else None

    @property
    def arguments(self) -> Mapping[str, Argument]:
        return self._arguments

    @property
    def attacks(self) -> frozenset[tuple[str, str]]:
        return self._attacks

    @property
    def supports(self) -> frozenset[tuple[str, str]]:
        return self._supports

    def attackers(self, name: str) -> frozenset[str]:
        return self._attackers[name]

    def supporters(self, name: str) -> frozenset[str]:
        return self._supporters[name]

    def is_acyclic(self) -> bool:
        """True if attacks ∪ supports has no cycle (a self-attack counts as one)."""
        return self._order is not None

    def topological_order(self) -> list[str]:
        """Every argument after all its attackers and supporters; ties broken by name."""
        if self._order is None:
            raise ValueError("the framework has a cycle over attacks ∪ supports")
        return list(self._order)

    def _key(self) -> tuple:
        return frozenset(self._arguments.values()), self._attacks, self._supports

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ArgumentationFramework) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return f"ArgumentationFramework({len(self._arguments)} arguments, {len(self._attacks)} attacks, {len(self._supports)} supports)"
