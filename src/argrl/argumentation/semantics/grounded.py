"""Grounded semantics (Dung 1995) and its in/out/undec labelling (Caminada 2006, Def. 6)."""

from __future__ import annotations
from typing import Literal
from argrl.argumentation.framework import ArgumentationFramework

Label = Literal["in", "out", "undec"]


def characteristic(af: ArgumentationFramework, S: frozenset[str]) -> frozenset[str]:
    """Dung's F(S): the arguments whose every attacker is attacked by S (Dung 1995, Def. 16)."""
    return frozenset(a for a in af.arguments if all(af.attackers(b) & S for b in af.attackers(a)))


def grounded_extension(af: ArgumentationFramework, *, ignore_supports: bool = False) -> frozenset[str]:
    """Least fixed point of F, by iterating from the empty set (Dung 1995, Def. 20 and Lemma 28)."""
    if af.supports and not ignore_supports:
        raise ValueError("grounded semantics has no supports; pass ignore_supports=True to drop them (D25)")
    S: frozenset[str] = frozenset()
    while (nxt := characteristic(af, S)) != S:
        S = nxt
    return S


def grounded_labelling(af: ArgumentationFramework, *, ignore_supports: bool = False) -> dict[str, Label]:
    """in = grounded extension; out = attacked by an in argument; undec = the rest."""
    g = grounded_extension(af, ignore_supports=ignore_supports)
    return {a: "in" if a in g else "out" if af.attackers(a) & g else "undec" for a in sorted(af.arguments)}
