# src/causalutils.py
import numpy as np
import pandas as pd

def correlation_dag(df: pd.DataFrame, feature_cols: list,
                    target_cols: list, threshold: float = 0.15):
    """Return edges (from, to, weight) based on correlation thresholding."""
    all_cols = feature_cols + target_cols
    corr     = df[all_cols].corr()
    edges    = []
    for f in feature_cols:
        for t in target_cols:
            w = corr.loc[f, t]
            if abs(w) >= threshold:
                edges.append({"from": f, "to": t,
                               "weight": round(w, 4),
                               "abs_weight": round(abs(w), 4)})
    return pd.DataFrame(edges).sort_values("abs_weight", ascending=False)

def direct_indirect_effects(df: pd.DataFrame,
                             feature_cols: list,
                             mediator_cols: list,
                             target: str):
    """Simple OLS-based mediation decomposition."""
    from sklearn.linear_model import LinearRegression
    rows = []
    for f in feature_cols:
        if f in mediator_cols:
            continue
        X_dir = df[[f]].fillna(0).values
        y     = df[target].fillna(0).values
        direct = LinearRegression().fit(X_dir, y).coef_[0]

        indirect = 0.0

def bootstrap_edge_stability(df: pd.DataFrame,
                              feature_cols: list,
                              target_cols: list,
                              n_bootstrap: int = 50,
                              threshold: float = 0.15,
                              sample_frac: float = 0.8) -> pd.DataFrame:
    """
    Estimate edge stability via bootstrap resampling.
    Returns each edge with a selection_rate across bootstrap runs.
    """
    from collections import defaultdict
    counts = defaultdict(int)

    for _ in range(n_bootstrap):
        sample = df.sample(frac=sample_frac, replace=True)
        edges  = correlation_dag(sample, feature_cols,
                                 target_cols, threshold)
        for _, row in edges.iterrows():
            counts[(row["from"], row["to"])] += 1

    rows = []
    for (src, tgt), cnt in counts.items():
        rows.append({
            "from":            src,
            "to":              tgt,
            "selection_rate":  round(cnt / n_bootstrap, 4),
            "stable":          cnt / n_bootstrap >= 0.7,
        })

    return (pd.DataFrame(rows)
              .sort_values("selection_rate", ascending=False)
              .reset_index(drop=True))

# ── aliases ──────────────────────────────────────────────
correlationdag          = correlation_dag
directindirecteffects   = direct_indirect_effects
bootstrapedgestability  = bootstrap_edge_stability