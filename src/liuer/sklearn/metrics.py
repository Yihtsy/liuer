from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def safe_auc_score(labels, values):
    """
    Ignore the positive and negative label and always obtain a value between 0.5 and 1, in case that the original roc_auc_score is smaller than 0.5.
    """
    auc_value = roc_auc_score(labels, values)
    if auc_value < 0.5:
        auc_value = 1 - auc_value
    return auc_value

def safe_roc_curve(labels, values):
    """
    Always keep the curve at the upperleft.
    Note: Assuming there are only two classes!
    """
    if roc_auc_score(labels, values) > 0.5:
        fpr, tpr, thresholds = roc_curve(labels, values)
    else:
        fpr, tpr, thresholds = roc_curve(labels, -values)
    return fpr, tpr, thresholds

def specificity_score(y_true, y_pred, pos_label, neg_label):
    """
    Calculate the specificity score.

    Parameters:
    y_true : array-like, shape (n_samples,)
        True labels.
    y_pred : array-like, shape (n_samples,)
        Predicted labels.
    pos_label : int or str, optional (default=1)
        The label of the positive class.

    Returns:
    score : float
        The specificity score (TNR).
    """
    cm = confusion_matrix(y_true, y_pred, labels=[pos_label, neg_label])
    
    tn = cm[1, 1]  # True Negative
    fp = cm[0, 1]  # False Positive

    if (tn + fp) == 0:
        return 0.0
    specificity = tn / (tn + fp)
    return specificity

class Report:
    def __init__(self, acc, prec, rec, f1, spec, auc):
        self.acc_ = acc
        self.prec_ = prec
        self.rec_ = rec
        self.f1_ = f1
        self.spec_ = spec
        self.auc_ = auc
        
def eval_model(y_true, y_pred, prob, pos_label, neg_label):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=pos_label)
    rec = recall_score(y_true, y_pred, pos_label=pos_label)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label)
    spec = specificity_score(y_true, y_pred, pos_label=pos_label, neg_label=neg_label)
    auc = safe_auc_score(y_true == pos_label, prob)
    report = Report(acc, prec, rec, f1, spec, auc)
    return report
