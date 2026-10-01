"""Iterator helpers inspired by :mod:`itertools`.

The module re-exports the standard library itertools functions and adds small
recipes that are useful enough to keep close to the original API.
"""

from collections import deque
from itertools import *  # noqa: F403
from itertools import combinations, islice, tee, zip_longest


def take(n, iterable):
    """Return the first *n* items from *iterable* as a list."""
    return list(islice(iterable, n))


def chunked(iterable, n, *, strict=False):
    """Yield tuples of length *n* from *iterable*.

    The final tuple may be shorter unless ``strict=True``.
    """
    if n <= 0:
        raise ValueError("n must be greater than 0")
    iterator = iter(iterable)
    while chunk := tuple(islice(iterator, n)):
        if strict and len(chunk) != n:
            raise ValueError("chunked() got an incomplete final chunk")
        yield chunk


def flatten(iterable):
    """Flatten one nesting level."""
    for group in iterable:
        yield from group


def consume(iterator, n=None):
    """Advance *iterator* quickly by *n* steps, or consume it entirely."""
    if n is None:
        deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)


def ilen(iterable):
    """Return the number of items produced by *iterable*."""
    return sum(1 for _ in iterable)


def quantify(iterable, pred=bool):
    """Count how many items satisfy *pred*."""
    return sum(1 for item in iterable if pred(item))


def first(iterable, default=None):
    """Return the first item, or *default* when the iterable is empty."""
    return next(iter(iterable), default)


def last(iterable, default=None):
    """Return the last item, or *default* when the iterable is empty."""
    items = deque(iterable, maxlen=1)
    return items[0] if items else default


def windowed(iterable, n, *, step=1):
    """Yield overlapping windows of length *n*."""
    if n <= 0:
        raise ValueError("n must be greater than 0")
    if step <= 0:
        raise ValueError("step must be greater than 0")

    iterator = iter(iterable)
    window = deque(islice(iterator, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    index = 0
    for item in iterator:
        window.append(item)
        index += 1
        if index % step == 0:
            yield tuple(window)


def unique_everseen(iterable, key=None):
    """Yield unique items while preserving first-seen order."""
    seen = set()
    for item in iterable:
        marker = item if key is None else key(item)
        if marker not in seen:
            seen.add(marker)
            yield item


def partition(pred, iterable):
    """Split items into ``(false_items, true_items)`` iterators."""
    left, right = tee((pred(item), item) for item in iterable)
    return (
        (item for flag, item in left if not flag),
        (item for flag, item in right if flag),
    )


def grouper(iterable, n, *, fillvalue=None, incomplete="fill"):
    """Collect data into fixed-length chunks.

    ``incomplete`` can be ``"fill"``, ``"ignore"``, or ``"strict"``.
    """
    if incomplete == "fill":
        args = [iter(iterable)] * n
        yield from zip_longest(*args, fillvalue=fillvalue)
    elif incomplete == "ignore":
        yield from zip(*[iter(iterable)] * n)
    elif incomplete == "strict":
        yield from chunked(iterable, n, strict=True)
    else:
        raise ValueError("incomplete must be 'fill', 'ignore', or 'strict'")


def all_combinations(iterable, include_empty=False):
    """Yield combinations of every possible length."""
    items = tuple(iterable)
    start_length = 0 if include_empty else 1
    for r in range(start_length, len(items) + 1):
        yield from combinations(items, r)


def combinations_with_required(iterable, required_items, r):
    """Yield length-*r* combinations that include all required items."""
    items = tuple(iterable)
    required_items = tuple(required_items)
    missing_items = [item for item in required_items if item not in items]
    if missing_items:
        missing = ", ".join(str(item) for item in missing_items)
        raise ValueError(f"Missing required items: {missing}")

    optional_length = r - len(required_items)
    if optional_length < 0:
        return

    optional_items = [item for item in items if item not in required_items]
    for combo in combinations(optional_items, optional_length):
        yield required_items + combo


def all_combinations_with_required(iterable, required_items, include_empty_optional=False):
    """Yield all combinations that include the required items."""
    items = tuple(iterable)
    required_items = tuple(required_items)
    missing_items = [item for item in required_items if item not in items]
    if missing_items:
        missing = ", ".join(str(item) for item in missing_items)
        raise ValueError(f"Missing required items: {missing}")

    optional_items = [item for item in items if item not in required_items]
    start_length = 0 if include_empty_optional else 1
    for optional_length in range(start_length, len(optional_items) + 1):
        for combo in combinations(optional_items, optional_length):
            yield required_items + combo
