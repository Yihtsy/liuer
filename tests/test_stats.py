import pandas as pd

from liuer.scipy.stats import group_stats, stat_test


def test_group_stats_numeric_and_categorical():
    df = pd.DataFrame(
        {
            "Group": ["A", "A", "A", "B", "B", "B"],
            "score": [1, 2, 3, 4, 5, 6],
            "kind": ["x", "x", "y", "x", "y", "y"],
        }
    )

    result = group_stats(df, num_feats=["score"], cat_feats=["kind"])

    assert result.loc["score.median", "A"] == 2
    assert result.loc["kind=x.count", "A"] == 2


def test_stat_test_mann_whitney():
    df = pd.DataFrame({"label": [1, 1, 0, 0], "score": [3, 4, 1, 2]})

    result = stat_test(df, "label", 1, targ_feats=["score"])

    assert result.loc[0, "feature"] == "score"
    assert result.loc[0, "method"] == "Mann-Whitney U"
