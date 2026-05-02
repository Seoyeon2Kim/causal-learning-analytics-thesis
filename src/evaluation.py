# src/evaluation.py
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def classification_summary(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = float("nan")
    return {
        "AUROC":     round(auroc, 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
    }

def summarise_dual_task(y_drop, p_drop, y_fail, p_fail, threshold=0.5):
    d = classification_summary(y_drop, p_drop, threshold)
    f = classification_summary(y_fail, p_fail, threshold)
    return {
        "AUROC_dropout": d["AUROC"],
        "AUROC_failure": f["AUROC"],
        "mean_AUROC":    round((d["AUROC"] + f["AUROC"]) / 2, 4),
        "mean_ECE":      round(expected_calibration_error(y_drop, p_drop), 4),
        "F1_dropout":    d["F1"],
        "F1_failure":    f["F1"],
    }

def top_k_alert_precision(y_true, y_prob, k=0.10):
    n   = max(1, int(len(y_true) * k))
    idx = np.argsort(y_prob)[::-1][:n]
    return round(float(np.mean(y_true[idx])), 4)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins   = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    n      = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return round(ece, 4)

# alias to match notebook import name
topk_alert_precision = top_k_alert_precision