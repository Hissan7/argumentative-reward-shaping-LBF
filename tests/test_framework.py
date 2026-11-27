import ast
from pathlib import Path
import pytest
from argrl.argumentation.framework import Argument, ArgumentationFramework

ARGUMENTATION = Path("src/argrl/argumentation")
FORBIDDEN = ("argrl.envs", "argrl.agents", "lbforaging")


def af(names, attacks=(), supports=()):
    return ArgumentationFramework([Argument(n) for n in names], attacks, supports)


def test_rejects_unknown_argument():
    with pytest.raises(ValueError, match="unknown"):
        af("ab", attacks=[("a", "c")])
    with pytest.raises(ValueError, match="unknown"):
        af("ab", supports=[("c", "a")])


def test_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate"):
        af("aba")


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan")])
def test_rejects_bad_base_score(score):
    with pytest.raises(ValueError, match="base score"):
        Argument("a", score)


def test_rejects_attack_and_support_same_pair():
    with pytest.raises(ValueError, match="both"):
        af("ab", attacks=[("a", "b")], supports=[("a", "b")])


def test_rejects_self_support():
    with pytest.raises(ValueError, match="self-support"):
        af("a", supports=[("a", "a")])


def test_allows_self_attack():
    f = af("a", attacks=[("a", "a")])
    assert f.attackers("a") == {"a"}


def test_attackers_and_supporters():
    f = af("abcd", attacks=[("a", "c"), ("b", "c")], supports=[("d", "c"), ("a", "b")])
    assert f.attackers("c") == {"a", "b"}
    assert f.supporters("c") == {"d"}
    assert f.supporters("b") == {"a"}
    assert f.attackers("a") == f.supporters("a") == frozenset()


@pytest.mark.parametrize(
    "attacks, supports, acyclic",
    [
        ([("a", "b"), ("b", "c")], [], True),
        ([("a", "b"), ("b", "a")], [], False),
        ([("a", "b"), ("b", "c"), ("c", "a")], [], False),
        ([("a", "a")], [], False),
        ([], [("a", "b"), ("b", "a")], False),
        ([("a", "b")], [("b", "a")], False),
    ],
)
def test_acyclicity(attacks, supports, acyclic):
    f = af("abc", attacks, supports)
    assert f.is_acyclic() is acyclic
    if not acyclic:
        with pytest.raises(ValueError, match="cycle"):
            f.topological_order()


def test_topological_order_leaves_first_and_deterministic():
    names = "edcba"
    attacks = [("a", "c"), ("b", "c"), ("c", "e")]
    supports = [("d", "e")]
    order = af(names, attacks, supports).topological_order()
    assert order == ["a", "b", "c", "d", "e"]
    shuffled = af(reversed(names), list(reversed(attacks)), supports).topological_order()
    assert shuffled == order


def test_immutable_and_hashable():
    f = af("ab", attacks=[("a", "b")])
    g = af("ba", attacks=[("a", "b")])
    assert f == g and hash(f) == hash(g) and len({f, g}) == 1
    with pytest.raises(AttributeError):
        f.attacks = frozenset()
    with pytest.raises(TypeError):
        f.arguments["c"] = Argument("c")


def imported_modules(path: Path) -> list[str]:
    package = list(path.relative_to("src").with_suffix("").parts[:-1])
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            found.append(".".join(base + ([node.module] if node.module else [])))
    return found


def test_argumentation_has_no_env_imports():
    files = list(ARGUMENTATION.rglob("*.py"))
    assert files
    for path in files:
        for module in imported_modules(path):
            assert not module.startswith(FORBIDDEN), f"{path} imports {module} (CLAUDE.md §5)"
