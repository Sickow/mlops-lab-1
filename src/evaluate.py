from typing import Dict
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

def compute_metrics(y_true, y_pred, y_proba=None, pos_label=1) -> Dict[str, float]:
    out = {
        "f1": f1_score(y_true, y_pred, pos_label=pos_label),
        "precision": precision_score(y_true, y_pred, pos_label=pos_label),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if y_proba is not None:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        except Exception:
            pass
    return out
