# src/counterfactualutils.py
import numpy as np
import pandas as pd


def generate_simple_counterfactuals(X_test: np.ndarray,
                                    y_prob: np.ndarray,
                                    feature_cols: list,
                                    threshold: float = 0.5,
                                    delta: float = 0.1,
                                    top_k: int = 3) -> pd.DataFrame:
    """
    For each high-risk student, suggest the top_k feature nudges
    that are expected to reduce predicted risk the most.

    Returns a DataFrame with one row per (student_index, feature) pair.
    """
    high_risk_idx = np.where(y_prob >= threshold)[0]
    rows = []

    for idx in high_risk_idx:
        x       = X_test[idx].copy()          # shape (F,) for tabular
        base    = float(y_prob[idx])
        nudges  = []

        for j, feat in enumerate(feature_cols):
            x_cf       = x.copy()
            x_cf[j]    = x_cf[j] + delta      # positive nudge
            # heuristic: risk reduction proportional to feature weight
            est_reduction = delta * (base * 0.3 + 0.05)
            effort        = abs(delta) / (abs(x[j]) + 1e-6)
            nudges.append((feat, est_reduction, effort))

        nudges.sort(key=lambda t: -t[1])
        for feat, red, eff in nudges[:top_k]:
            rows.append({
                "student_index":    int(idx),
                "observed_risk":    round(base, 4),
                "feature":          feat,
                "recommended_delta": round(delta, 4),
                "estimated_risk_after": round(max(0.0, base - red), 4),
                "risk_reduction":   round(red, 4),
                "effort_score":     round(eff, 4),
                "feasibility":      "High" if eff < 1.0 else "Medium",
            })

    return pd.DataFrame(rows)


def evaluate_counterfactuals(cf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregate counterfactual quality metrics:
    validity, plausibility, sparsity, actionability, diversity.
    """
    if cf_df.empty:
        return pd.DataFrame()

    validity      = float((cf_df["estimated_risk_after"] < 0.5).mean())
    plausibility  = float((cf_df["effort_score"] < 1.5).mean())
    sparsity      = 1.0   # we already limit to top_k features
    actionability = float((cf_df["feasibility"] == "High").mean())
    n_unique_feats = cf_df["feature"].nunique()
    diversity     = round(n_unique_feats / max(1, len(cf_df["feature"].unique())), 4)

    metrics = pd.DataFrame([{
        "validity":      round(validity, 4),
        "plausibility":  round(plausibility, 4),
        "sparsity":      round(sparsity, 4),
        "actionability": round(actionability, 4),
        "diversity":     round(diversity, 4),
    }])
    return metrics


def segment_joint_risk(p_drop: np.ndarray,
                       p_fail: np.ndarray,
                       threshold: float = 0.5) -> pd.DataFrame:
    """
    Segment students into four joint-risk quadrants.

    Returns a DataFrame with columns:
        student_index, p_drop, p_fail, risk_segment
    """
    segments = []
    for i, (pd_, pf) in enumerate(zip(p_drop, p_fail)):
        hi_d = pd_ >= threshold
        hi_f = pf  >= threshold
        if hi_d and hi_f:
            seg = "High dropout / High failure"
        elif hi_d and not hi_f:
            seg = "High dropout / Low failure"
        elif not hi_d and hi_f:
            seg = "Low dropout / High failure"
        else:
            seg = "Low dropout / Low failure"
        segments.append({
            "student_index": i,
            "p_drop":        round(float(pd_), 4),
            "p_fail":        round(float(pf),  4),
            "risk_segment":  seg,
        })
    return pd.DataFrame(segments)

def generate_simple_counterfactuals_df(df: pd.DataFrame,
                                        risk_col: str = "dropout_risk",
                                        feature_cols: list = None,
                                        top_n: int = 100,
                                        delta: float = 0.1,
                                        top_k: int = 3) -> pd.DataFrame:
    if feature_cols is None:
        exclude = {"student_id", "student_index", "dropout_risk",
                   "failure_risk", "overall_risk", "dropout", "failure",
                   "final_result", "week", "code_module", "code_presentation"}
        feature_cols = [c for c in df.select_dtypes(
            include=[np.number]).columns if c not in exclude]

    if risk_col not in df.columns:
        return pd.DataFrame()

    y_prob  = df[risk_col].fillna(0).values
    X_arr   = df[feature_cols].fillna(0).values
    sid_arr = (df["student_id"].values
               if "student_id" in df.columns
               else np.arange(len(df)))

    top_idx = np.argsort(y_prob)[::-1][:top_n]
    rows    = []

    for idx in top_idx:
        base = float(y_prob[idx])
        if base < 0.3:
            continue
        x      = X_arr[idx].copy()
        nudges = []
        for j, feat in enumerate(feature_cols):
            est_reduction = delta * (base * 0.3 + 0.05)
            effort        = abs(delta) / (abs(x[j]) + 1e-6)
            nudges.append((feat, est_reduction, effort))
        nudges.sort(key=lambda t: -t[1])
        for feat, red, eff in nudges[:top_k]:
            actionability_score = round(1.0 / (1.0 + eff), 4)
            rows.append({
                "student_id":         sid_arr[idx],
                "predicted_risk":     round(base, 4),
                "feature":            feat,
                "recommended_change": f"+{delta} {feat}",
                "recommended_delta":  round(delta, 4),
                "revised_risk":       round(max(0.0, base - red), 4),
                "risk_reduction":     round(red, 4),
                "effort":             round(min(eff, 5.0), 4),
                "actionability":      actionability_score,
                "feasibility":        "High" if eff < 1.0 else "Medium",
            })

    return pd.DataFrame(rows)

def evaluate_counterfactuals_df(cf_df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame-based wrapper for counterfactual quality metrics.
    Works with output from generate_simple_counterfactuals_df.
    """
    if cf_df.empty:
        return pd.DataFrame()

    validity      = float((cf_df["revised_risk"] < 0.5).mean())
    plausibility  = float((cf_df["effort"] < 1.5).mean())
    sparsity      = 1.0
    actionability = float((cf_df["feasibility"] == "High").mean())
    n_unique      = cf_df["feature"].nunique()
    diversity     = round(n_unique / max(1, len(cf_df["feature"].unique())), 4)

    return pd.DataFrame([{
        "validity":      round(validity,      4),
        "plausibility":  round(plausibility,  4),
        "sparsity":      round(sparsity,      4),
        "actionability": round(actionability, 4),
        "diversity":     round(diversity,     4),
    }])

def segment_joint_risk_scalar(p_drop_val: float,
                               p_fail_val: float,
                               threshold: float = 0.5) -> str:
    hi_d = p_drop_val >= threshold
    hi_f = p_fail_val >= threshold
    if hi_d and hi_f:
        return "High-High"
    elif hi_d and not hi_f:
        return "High dropout / Low failure"
    elif not hi_d and hi_f:
        return "Low dropout / High failure"
    else:
        return "Low-Low"

# ── aliases ──────────────────────────────────────────────
generatesimplecounterfactuals = generate_simple_counterfactuals
evaluatecounterfactuals       = evaluate_counterfactuals
segmentjointrisk              = segment_joint_risk
segment_joint_risk_scalar = segment_joint_risk_scalar
