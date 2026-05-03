# src/causal_utils.py
import numpy as np
import pandas as pd


def correlation_dag(df: pd.DataFrame,
                    feature_cols: list,
                    target_cols: list = None,
                    threshold: float = 0.15) -> pd.DataFrame:

    if target_cols is None:
        target_cols = [c for c in ["dropout", "failure"] if c in df.columns]

    all_cols = [c for c in feature_cols + target_cols if c in df.columns]
    corr     = df[all_cols].corr()
    edges    = []

    for f in feature_cols:
        if f not in corr.index:
            continue
        for t in target_cols:
            if t not in corr.columns:
                continue
            w = corr.loc[f, t]
            if abs(w) >= threshold:
                edges.append({
                    "source":     f,
                    "target":     t,
                    "from":       f,
                    "to":         t,
                    "weight":     round(w, 4),
                    "abs_weight": round(abs(w), 4),
                })

    if not edges:
        return pd.DataFrame(columns=["source","target","from","to",
                                     "weight","abs_weight"])

    return (pd.DataFrame(edges)
              .sort_values("abs_weight", ascending=False)
              .reset_index(drop=True))


def direct_indirect_effects(df: pd.DataFrame,
                             feature_cols: list,
                             target: str,
                             mediator_cols: list = None) -> pd.DataFrame:
    from sklearn.linear_model import LinearRegression

    if mediator_cols is None:
        mediator_cols = ["n_submissions", "mean_score", "late_rate"]

    rows = []
    for f in feature_cols:
        if f == target:
            continue
        y   = df[target].fillna(0).values
        X_f = df[[f]].fillna(0).values

        direct = float(LinearRegression().fit(X_f, y).coef_[0])

        indirect = 0.0
        valid_med = [m for m in mediator_cols
                     if m in df.columns and m != f and m != target]
        for med in valid_med:
            X_m = df[[med]].fillna(0).values
            a   = float(LinearRegression().fit(X_f, X_m.ravel()).coef_[0])
            b   = float(LinearRegression().fit(X_m, y).coef_[0])
            indirect += a * b

        total = direct + indirect

        if abs(direct) > abs(indirect):
            role = "Cause"
        elif abs(indirect) > abs(direct) * 1.5:
            role = "Proxy"
        else:
            role = "Mixed"

        rows.append({
            "behavior":        f,
            "direct_effect":   round(direct,   4),
            "indirect_effect": round(indirect, 4),
            "total_effect":    round(total,    4),
            "primary_role":    role,
        })

    return (pd.DataFrame(rows)
              .sort_values("total_effect", key=abs, ascending=False)
              .reset_index(drop=True))


def bootstrap_edge_stability(df: pd.DataFrame,
                              feature_cols: list,
                              target_cols: list = None,
                              n_boot: int = 30,
                              threshold: float = 0.15,
                              sample_frac: float = 0.8) -> pd.DataFrame:

    if target_cols is None:
        target_cols = [c for c in ["dropout", "failure"] if c in df.columns]

    from collections import defaultdict
    counts = defaultdict(int)

    for _ in range(n_boot):
        sample = df.sample(frac=sample_frac, replace=True)
        edges  = correlation_dag(sample, feature_cols,
                                 target_cols, threshold)
        for _, row in edges.iterrows():
            counts[(row["source"], row["target"])] += 1

    rows = []
    for (src, tgt), cnt in counts.items():
        rows.append({
            "edge":                f"{src} - {tgt}",
            "source":              src,
            "target":              tgt,
            "bootstrap_stability": round(cnt / n_boot, 4),
            "stable":              cnt / n_boot >= 0.7,
        })

    if not rows:
        return pd.DataFrame(columns=["edge","source","target",
                                     "bootstrap_stability","stable"])

    return (pd.DataFrame(rows)
              .sort_values("bootstrap_stability", ascending=False)
              .reset_index(drop=True))


# ── aliases ───────────────────────────────────────────────────────────────────
correlationdag         = correlation_dag
directindirecteffects  = direct_indirect_effects
bootstrapedgestability = bootstrap_edge_stability