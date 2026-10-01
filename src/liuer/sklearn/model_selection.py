import numpy as np
from sklearn.model_selection import BaseCrossValidator


class StratifiedLeavePairOut(BaseCrossValidator):
    def __init__(self, positive_label, negative_label):
        self.positive_label = positive_label
        self.negative_label = negative_label

    def split(self, X, y=None, groups=None):
        positive_indices = y[y == self.positive_label].index
        negative_indices = y[y == self.negative_label].index

        for pos_idx in positive_indices:
            for neg_idx in negative_indices:
                test_indices = [pos_idx, neg_idx]
                train_indices = [i for i in y.index if i not in test_indices]
                yield np.array(train_indices), np.array(test_indices)

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(y[y == self.positive_label]) * len(y[y == self.negative_label])
