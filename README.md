# liuer

Liuer mimics well-known Python packages and fills practical gaps with small, compatible extensions.

The name comes from "六耳": it listens to familiar APIs, imitates their shape, and adds the missing helper functions that are useful in everyday Python work.

## Installation

```bash
pip install liuer
```

For local development:

```bash
git clone https://github.com/yihtsy/liuer.git
cd liuer
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Modules

### `liuer.scipy.stats`

Extensions inspired by `scipy.stats`.

```python
from liuer.scipy.stats import compare_pos_neg_groups

table = compare_pos_neg_groups(
    data=df,
    lab_column="label",
    pos_label=1,
    neg_label=0,
)
```

```python
from liuer.scipy.stats import group_stats, stat_test

summary = group_stats(df, group_col="Group", num_feats=["age"], cat_feats=["sex"])
tests = stat_test(df, label_col="label", pos_label=1, targ_feats=["score"])
```

### `liuer.itertools`

Iterator helpers inspired by the Python standard library `itertools` and the recipe ecosystem.

```python
from liuer.itertools import all_combinations, chunked, flatten, windowed

list(chunked(range(5), 2))
# [(0, 1), (2, 3), (4,)]

list(flatten([[1, 2], [3]]))
# [1, 2, 3]

list(windowed([1, 2, 3, 4], 3))
# [(1, 2, 3), (2, 3, 4)]

list(all_combinations(["a", "b"]))
# [('a',), ('b',), ('a', 'b')]
```

### `liuer.matplotlib.pyplot`

Visualization helpers built on top of `matplotlib.pyplot`.

```python
from liuer.matplotlib import pyplot as lplt

fig, ax = lplt.radar_chart(
    labels=["A", "B", "C"],
    values=[0.8, 0.4, 0.7],
)
```

```python
grid = lplt.plot_jitter_boxplot_with_significance(
    df,
    target_features=["score"],
    p_values_dict={"score": 0.03},
    ordered_labels=["Control", "Case"],
    show=False,
)
```

For convenience, the same plotting helpers are also available from `liuer.plot`.

## Development Workflow

```bash
python -m pytest
python -m ruff check .
python -m build
python -m twine check dist/*
```

Every release should update `CHANGELOG.md`, keep the GitHub README current, and publish only from a clean git working tree.

PyPI publishing is intended to run through GitHub Actions Trusted Publishing. After configuring `Yihtsy/liuer` as a trusted publisher for the `liuer` project on PyPI, push a release tag such as:

```bash
git tag v0.0.2
git push origin v0.0.2
```

## License

MIT
