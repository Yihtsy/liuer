import pandas as pd


def group_sample(
    df,
    n=None,
    frac=None,
    replace=False,
    random_state=None,
    group_col='Group',
    label=None,
    ignore_index=False,
):
    """
    Randomly sample from a specified group in the DataFrame, while keeping other groups unchanged.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        n (int, optional): Number of samples to draw.
        frac (float, optional): Fraction of samples to draw.
        replace (bool): Whether to sample with replacement. Default is False.
        random_state (int, optional): Random seed for reproducibility.
        group_col (str): Column name that specifies the group. Default is 'Group'.
        label (str or int): The target group label to sample from.
        ignore_index (bool): If True, reset index after sampling.

    Returns:
        pd.DataFrame: New DataFrame with sampled target group and untouched other groups.
    """
    if label is None:
        raise ValueError("You must specify a target label for sampling.")

    if (n is None) == (frac is None):
        raise ValueError("Exactly one of 'n' or 'frac' must be specified.")

    # Split the DataFrame into the target group and others
    target_df = df[df[group_col] == label]
    other_df = df[df[group_col] != label]

    if frac is not None:
        n_samples = int(round(len(target_df) * frac))
    else:
        n_samples = n

    if not replace and n_samples > len(target_df):
        raise ValueError(
            f"Cannot sample {n_samples} samples without replacement from only {len(target_df)} available samples."
        )

    # Perform sampling on the target group
    sampled_target_df = target_df.sample(
        n=n_samples,
        replace=replace,
        random_state=random_state
    )

    # Concatenate the sampled target group with the other groups
    result_df = pd.concat([other_df, sampled_target_df], ignore_index=ignore_index)

    return result_df
