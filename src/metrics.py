import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """rows = true label, cols = predicted label."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_precision_recall_f1(cm: np.ndarray) -> dict[int, dict[str, float]]:
    """From a confusion matrix (see confusion_matrix()), one precision/recall/f1
    triple per class index."""
    num_classes = cm.shape[0]
    out: dict[int, dict[str, float]] = {}
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[c] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
    return out


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())
