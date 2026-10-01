from liuer.itertools import (
    all_combinations,
    all_combinations_with_required,
    chunked,
    combinations_with_required,
    flatten,
    unique_everseen,
    windowed,
)


def test_chunked_keeps_short_final_chunk():
    assert list(chunked(range(5), 2)) == [(0, 1), (2, 3), (4,)]


def test_flatten_one_level():
    assert list(flatten([[1, 2], [3], []])) == [1, 2, 3]


def test_unique_everseen_preserves_order():
    assert list(unique_everseen(["a", "b", "a", "c"])) == ["a", "b", "c"]


def test_windowed():
    assert list(windowed([1, 2, 3, 4], 3)) == [(1, 2, 3), (2, 3, 4)]


def test_all_combinations():
    assert list(all_combinations(["a", "b"])) == [("a",), ("b",), ("a", "b")]


def test_combinations_with_required():
    assert list(combinations_with_required(["a", "b", "c"], ["a"], 2)) == [
        ("a", "b"),
        ("a", "c"),
    ]


def test_all_combinations_with_required():
    assert list(all_combinations_with_required(["a", "b", "c"], ["a"])) == [
        ("a", "b"),
        ("a", "c"),
        ("a", "b", "c"),
    ]
