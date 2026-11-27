from itertools import combinations
import numpy as np
import pytest
from argrl.argumentation.framework import Argument, ArgumentationFramework
from argrl.argumentation.semantics.grounded import characteristic, grounded_extension, grounded_labelling


def af(names, attacks=(), supports=()):
    return ArgumentationFramework([Argument(n) for n in names], attacks, supports)


def random_af(rng, acyclic: bool) -> ArgumentationFramework:
    names = [f"a{i}" for i in range(rng.integers(1, 7))]
    pairs = [(x, y) for i, x in enumerate(names) for j, y in enumerate(names) if not acyclic or i < j]
    return af(names, [p for p in pairs if rng.random() < 0.3])


def conflict_free(f, S) -> bool:
    return not any((a, b) in f.attacks for a in S for b in S)


def complete_extensions(f) -> list[frozenset[str]]:
    # brute force: a conflict-free S is complete iff S = F(S) (Dung 1995, Lemma 24)
    subsets = (frozenset(c) for k in range(len(f.arguments) + 1) for c in combinations(f.arguments, k))
    return [S for S in subsets if conflict_free(f, S) and characteristic(f, S) == S]


@pytest.mark.parametrize(
    "names, attacks, expected",
    [
        ("", [], set()),
        ("a", [], {"a"}),
        ("ab", [("a", "b")], {"a"}),
        ("abc", [("a", "b"), ("b", "c")], {"a", "c"}),
        ("ab", [("a", "b"), ("b", "a")], set()),  # Nixon diamond (Dung 1995)
        ("a", [("a", "a")], set()),
        ("ab", [("a", "a"), ("a", "b")], set()),
        ("abc", [("a", "b"), ("b", "c"), ("c", "a")], set()),
        ("abcd", [("a", "b"), ("b", "a"), ("b", "c"), ("c", "d")], set()),
        ("abcde", [("a", "b"), ("c", "b"), ("b", "d"), ("d", "e")], {"a", "c", "d"}),
    ],
)
def test_textbook_cases(names, attacks, expected):
    assert grounded_extension(af(names, attacks)) == expected


def test_self_attacker_leaves_its_target_undec():
    assert grounded_labelling(af("ab", [("a", "a"), ("a", "b")])) == {"a": "undec", "b": "undec"}


def test_dung_example_21():
    # Dung 1995, Example 3 (p. 326) and Example 21 (p. 329)
    f = af(["i1", "i2", "a"], [("i1", "a"), ("a", "i1"), ("i2", "a")])
    steps = [frozenset()]
    for _ in range(3):
        steps.append(characteristic(f, steps[-1]))
    assert steps[1:] == [{"i2"}, {"i1", "i2"}, {"i1", "i2"}]
    assert grounded_extension(f) == {"i1", "i2"}


def test_labelling():
    rng = np.random.default_rng(0)
    for _ in range(200):
        f = random_af(rng, acyclic=False)
        lab = grounded_labelling(f)
        assert {a for a, v in lab.items() if v == "in"} == grounded_extension(f)
        for a, v in lab.items():
            att = [lab[b] for b in f.attackers(a)]
            if v == "in":
                assert all(x == "out" for x in att)
            elif v == "out":
                assert "in" in att
            else:
                assert "in" not in att and "undec" in att


def test_grounded_is_least_complete():
    rng = np.random.default_rng(1)
    for _ in range(200):
        f = random_af(rng, acyclic=False)
        g = grounded_extension(f)
        complete = complete_extensions(f)
        assert g in complete  # Theorem 25
        assert all(g <= S for S in complete)


def test_acyclic_unique_complete_and_stable():
    rng = np.random.default_rng(2)
    for _ in range(200):
        f = random_af(rng, acyclic=True)
        g = grounded_extension(f)
        assert complete_extensions(f) == [g]  # Theorem 30
        assert all(f.attackers(a) & g for a in f.arguments if a not in g)  # stable


def test_supports_raise_unless_ignored():
    f = af("abc", attacks=[("a", "b")], supports=[("c", "b")])
    with pytest.raises(ValueError, match="supports"):
        grounded_extension(f)
    with pytest.raises(ValueError, match="supports"):
        grounded_labelling(f)
    assert grounded_extension(f, ignore_supports=True) == {"a", "c"}


def test_order_independent():
    rng = np.random.default_rng(3)
    for _ in range(50):
        f = random_af(rng, acyclic=False)
        names, attacks = list(f.arguments), list(f.attacks)
        rng.shuffle(names)
        rng.shuffle(attacks)
        assert grounded_extension(af(names, attacks)) == grounded_extension(f)
