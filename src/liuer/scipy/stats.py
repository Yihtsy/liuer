import math
from collections.abc import Iterable

import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from ..sklearn.metrics import safe_auc_score


def _format_number(value, digits, strip_leading_zero):
    if value is None or pd.isna(value):
        return None
    formatted = f"{value:.{digits}f}"
    if strip_leading_zero and formatted.startswith("0."):
        return formatted[1:]
    if strip_leading_zero and formatted.startswith("-0."):
        return "-" + formatted[2:]
    return formatted


def _shapiro_p_value(values):
    if len(values) < 3:
        return math.nan
    return stats.shapiro(values).pvalue


def group_stats(df, group_col="Group", group_labels=None, num_feats=None, cat_feats=None):
    """Compute grouped descriptive statistics.

    Numeric features include mean, standard deviation, median, IQR, Shapiro-Wilk
    p-value, and a boolean normality flag. Categorical features include counts
    and percentages by category.
    """
    if group_labels is None:
        group_labels = df[group_col].dropna().unique().tolist()

    all_stats = []
    for label in group_labels:
        group = df[df[group_col] == label]
        summary = {}

        if num_feats:
            for feat in num_feats:
                values = group[feat].dropna()
                shapiro_p = _shapiro_p_value(values)
                summary[f"{feat}.mean"] = values.mean()
                summary[f"{feat}.std"] = values.std()
                summary[f"{feat}.median"] = values.median()
                summary[f"{feat}.iqr"] = values.quantile(0.75) - values.quantile(0.25)
                summary[f"{feat}.shapiro_p_value"] = shapiro_p
                summary[f"{feat}.normality"] = (
                    bool(shapiro_p > 0.05) if not pd.isna(shapiro_p) else None
                )

        if cat_feats:
            for feat in cat_feats:
                values = group[feat].dropna()
                value_counts = values.value_counts()
                denominator = len(group)
                for category, count in value_counts.items():
                    summary[f"{feat}={category}.count"] = int(count)
                    summary[f"{feat}={category}.percent"] = (
                        float(count) / denominator * 100 if denominator else math.nan
                    )

        all_stats.append(pd.DataFrame({label: summary}))

    return pd.concat(all_stats, axis=1) if all_stats else pd.DataFrame()


def stat_test(
    df,
    label_col,
    pos_label,
    binary=True,
    targ_feats=None,
    exclude_cols=None,
    safe_auc=True,
    method="MannWhitney",
):
    """Run feature-wise statistical tests for binary classification data."""
    if not binary:
        raise NotImplementedError("Only binary classification is supported.")

    if targ_feats is None:
        exclude = {label_col}
        if exclude_cols is not None:
            if isinstance(exclude_cols, str):
                exclude_cols = [exclude_cols]
            elif not isinstance(exclude_cols, Iterable):
                raise TypeError("exclude_cols must be a column name or an iterable of names.")
            exclude.update(exclude_cols)
        targ_feats = [col for col in df.columns if col not in exclude]

    if isinstance(targ_feats, str) or not isinstance(targ_feats, Iterable):
        raise TypeError("targ_feats must be an iterable of feature names.")

    results = []
    for feat in targ_feats:
        pos_group = df[df[label_col] == pos_label][feat].dropna()
        neg_group = df[df[label_col] != pos_label][feat].dropna()
        if len(pos_group) == 0 or len(neg_group) == 0:
            continue

        valid_rows = df[feat].notna()
        labels = df.loc[valid_rows, label_col].values
        feat_values = df.loc[valid_rows, feat].values
        if len(set(labels)) < 2:
            auc = None
        elif safe_auc:
            auc = safe_auc_score(labels == pos_label, feat_values)
        else:
            auc = roc_auc_score(labels == pos_label, feat_values)

        if method == "MannWhitney":
            stat, p_value = stats.mannwhitneyu(pos_group, neg_group, alternative="two-sided")
            results.append(
                {
                    "feature": feat,
                    "method": "Mann-Whitney U",
                    "statistic": stat,
                    "p-value": round(p_value, 3),
                    "AUC": round(auc, 3) if auc is not None else None,
                    "pos_label median": round(pos_group.median(), 3),
                    "neg_label median": round(neg_group.median(), 3),
                    "n_pos": len(pos_group),
                    "n_neg": len(neg_group),
                }
            )
        else:
            raise ValueError(f"Unsupported method: {method}")

    return pd.DataFrame(results)


def compare_pos_neg_groups(
    data,
    lab_column,
    pos_label,
    neg_label,
    *,
    digits=2,
    p_digits=3,
    strip_leading_zero=True,
    dropna=True,
):
    """Compare numeric features between positive and negative groups.

    The function returns descriptive statistics, Welch's t-test, Mann-Whitney U,
    AUC, safe AUC, and Shapiro-Wilk normality p-values for every numeric feature.
    """
    pos_group = data[data[lab_column] == pos_label].drop(columns=[lab_column])
    neg_group = data[data[lab_column] == neg_label].drop(columns=[lab_column])
    numeric_columns = pos_group.select_dtypes(include="number").columns.intersection(
        neg_group.select_dtypes(include="number").columns
    )

    rows = []
    for column in numeric_columns:
        pos_values = pos_group[column]
        neg_values = neg_group[column]
        if dropna:
            pos_values = pos_values.dropna()
            neg_values = neg_values.dropna()

        labels = [1] * len(pos_values) + [0] * len(neg_values)
        values = list(pos_values) + list(neg_values)

        if len(pos_values) == 0 or len(neg_values) == 0:
            t_stat = t_p_value = u_stat = u_p_value = auc_value = safe_auc_value = math.nan
        else:
            t_result = stats.ttest_ind(pos_values, neg_values, equal_var=False, nan_policy="omit")
            u_result = stats.mannwhitneyu(pos_values, neg_values, alternative="two-sided")
            t_stat, t_p_value = t_result.statistic, t_result.pvalue
            u_stat, u_p_value = u_result.statistic, u_result.pvalue
            auc_value = roc_auc_score(labels, values)
            safe_auc_value = safe_auc_score(labels, values)

        pos_shapiro_p = _shapiro_p_value(pos_values)
        neg_shapiro_p = _shapiro_p_value(neg_values)

        rows.append(
            {
                "feature": column,
                "pos_mean": _format_number(pos_values.mean(), digits, False),
                "neg_mean": _format_number(neg_values.mean(), digits, False),
                "pos_std": _format_number(pos_values.std(), digits, False),
                "neg_std": _format_number(neg_values.std(), digits, False),
                "pos_median": _format_number(pos_values.median(), digits, False),
                "neg_median": _format_number(neg_values.median(), digits, False),
                "pos_25_percentile": _format_number(pos_values.quantile(0.25), digits, False),
                "neg_25_percentile": _format_number(neg_values.quantile(0.25), digits, False),
                "pos_75_percentile": _format_number(pos_values.quantile(0.75), digits, False),
                "neg_75_percentile": _format_number(neg_values.quantile(0.75), digits, False),
                "pos_IQR": _format_number(
                    pos_values.quantile(0.75) - pos_values.quantile(0.25), digits, False
                ),
                "neg_IQR": _format_number(
                    neg_values.quantile(0.75) - neg_values.quantile(0.25), digits, False
                ),
                "t_stat": _format_number(t_stat, digits, False),
                "t_p_value": _format_number(t_p_value, p_digits, strip_leading_zero),
                "u_stat": _format_number(u_stat, digits, False),
                "u_p_value": _format_number(u_p_value, p_digits, strip_leading_zero),
                "auc": _format_number(auc_value, p_digits, strip_leading_zero),
                "safe_auc": _format_number(safe_auc_value, p_digits, strip_leading_zero),
                "pos_shapiro_p": _format_number(pos_shapiro_p, p_digits, strip_leading_zero),
                "pos_normality": bool(pos_shapiro_p > 0.05) if not pd.isna(pos_shapiro_p) else None,
                "neg_shapiro_p": _format_number(neg_shapiro_p, p_digits, strip_leading_zero),
                "neg_normality": bool(neg_shapiro_p > 0.05) if not pd.isna(neg_shapiro_p) else None,
            }
        )

    return pd.DataFrame(rows)
