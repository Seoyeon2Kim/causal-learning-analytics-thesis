# src/datautils.py
import pandas as pd
import numpy as np
from pathlib import Path

# ── OULAD column aliases ──────────────────────────────────────────────────────
_ALIAS_ID   = ["id_student", "student_id", "user_id", "userId"]
_ALIAS_WEEK = ["week", "date", "week_number", "week_num"]
_ALIAS_ACT  = ["sum_click", "clicks", "activity", "sum_clicks"]

def _col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_generic_education_dataset(path: str, dataset_name: str = "oulad") -> pd.DataFrame:
    """
    Load OULAD (or compatible) event logs and return a merged weekly DataFrame.
    Automatically maps common column aliases.
    """
    path = Path(path)

    # ── Try OULAD layout first ────────────────────────────────────────────────
    vle_path  = path / "studentVle.csv"
    info_path = path / "studentInfo.csv"
    asmt_path = path / "studentAssessment.csv"
    asmt_def  = path / "assessments.csv"

    if vle_path.exists() and info_path.exists():
        vle  = pd.read_csv(vle_path)
        info = pd.read_csv(info_path)

        # normalise student-id column
        sid_vle  = _col(vle,  _ALIAS_ID) or vle.columns[0]
        sid_info = _col(info, _ALIAS_ID) or info.columns[0]
        vle  = vle.rename(columns={sid_vle:  "student_id"})
        info = info.rename(columns={sid_info: "student_id"})

        # week column  (OULAD uses 'date' as day-offset; convert to week)
        date_col = _col(vle, _ALIAS_WEEK + ["date"])
        if date_col and date_col != "week":
            vle["week"] = (vle[date_col] // 7).clip(lower=0)
        elif "week" not in vle.columns:
            vle["week"] = 0

        # clicks
        click_col = _col(vle, _ALIAS_ACT)
        if click_col and click_col != "sum_click":
            vle = vle.rename(columns={click_col: "sum_click"})
        elif "sum_click" not in vle.columns:
            vle["sum_click"] = 1

        # weekly aggregation per student
        weekly_vle = (
            vle.groupby(["student_id", "week", "code_module", "code_presentation"],
                        dropna=False)
               .agg(sum_click=("sum_click", "sum"),
                    n_activities=("sum_click", "count"))
               .reset_index()
        )

        # assessment features
        if asmt_path.exists() and asmt_def.exists():
            sa  = pd.read_csv(asmt_path)
            adf = pd.read_csv(asmt_def)
            sa  = sa.merge(adf[["id_assessment","date","weight"]],
                           on="id_assessment", how="left")
            sa["week"]       = (sa["date"] // 7).clip(lower=0)
            sa["is_late"]    = (sa["date_submitted"] > sa["date"]).astype(int)
            sa["score_norm"] = sa["score"].fillna(0) / 100.0
            sa = sa.rename(columns={"id_student": "student_id"})
            asmt_weekly = (
                sa.groupby(["student_id","week"])
                  .agg(n_submissions=("score_norm","count"),
                       mean_score=("score_norm","mean"),
                       late_rate=("is_late","mean"))
                  .reset_index()
            )
            weekly_vle = weekly_vle.merge(asmt_weekly,
                                          on=["student_id","week"],
                                          how="left")

        # merge student-level outcomes
        info["dropout"] = (info["final_result"] == "Withdrawn").astype(int)
        info["failure"] = (info["final_result"] == "Fail").astype(int)
        keep = ["student_id","code_module","code_presentation",
                "gender","age_band","imd_band","highest_education",
                "num_of_prev_attempts","studied_credits",
                "dropout","failure","final_result"]
        keep = [c for c in keep if c in info.columns]
        merged = weekly_vle.merge(info[keep],
                                  on=["student_id","code_module","code_presentation"],
                                  how="left")
        return merged

    # ── Fallback: single CSV ──────────────────────────────────────────────────
    csvs = list(path.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {path}")
    df = pd.read_csv(csvs[0])
    for alias in _ALIAS_ID:
        if alias in df.columns:
            df = df.rename(columns={alias: "student_id"})
            break
    if "week" not in df.columns:
        df["week"] = 0
    if "dropout" not in df.columns:
        df["dropout"] = 0
    if "failure" not in df.columns:
        df["failure"] = 0
    return df


def build_weekly_timelines(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing weeks so every student has a contiguous timeline."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in ["student_id","week","dropout","failure"]:
        if col in numeric_cols:
            numeric_cols.remove(col)
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def derive_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dropout / failure columns are integer dtype."""
    df = df.copy()
    df["dropout"] = df["dropout"].fillna(0).astype(int)
    df["failure"] = df["failure"].fillna(0).astype(int)
    return df


def cumulative_snapshot(df: pd.DataFrame, up_to_week: int) -> pd.DataFrame:
    """Aggregate all events up to (and including) up_to_week per student."""
    sub = df[df["week"] <= up_to_week].copy()
    numeric_cols = sub.select_dtypes(include=[np.number]).columns.tolist()
    for col in ["student_id","week","dropout","failure"]:
        if col in numeric_cols:
            numeric_cols.remove(col)
    snap = (
        sub.groupby("student_id")[numeric_cols]
           .mean()
           .reset_index()
    )
    labels = (
        df.groupby("student_id")[["dropout","failure"]]
          .max()
          .reset_index()
    )
    snap = snap.merge(labels, on="student_id", how="left")
    return snap


def make_sequence_tensor(df: pd.DataFrame,
                         feature_cols: list,
                         max_len: int = 12,
                         up_to_week: int = None):
    """
    Return (X, y_drop, y_fail) where X has shape (N, max_len, F).
    df should already be filtered to up_to_week before calling,
    OR pass up_to_week to filter inside.
    """
    if up_to_week is not None:
        df = df[df["week"] <= up_to_week].copy()

    students = df["student_id"].unique()
    F = len(feature_cols)
    X      = np.zeros((len(students), max_len, F), dtype=np.float32)
    y_drop = np.zeros(len(students), dtype=np.float32)
    y_fail = np.zeros(len(students), dtype=np.float32)

    labels = df.groupby("student_id")[["dropout", "failure"]].max()

    for i, sid in enumerate(students):
        rows = df[df["student_id"] == sid].sort_values("week")
        vals = rows[feature_cols].fillna(0).values[-max_len:]
        X[i, :len(vals), :] = vals
        if sid in labels.index:
            y_drop[i] = labels.loc[sid, "dropout"]
            y_fail[i] = labels.loc[sid, "failure"]

    return X, y_drop, y_fail

# ── aliases ──────────────────────────────────────────────
loadgenericeducationdataset = load_generic_education_dataset
buildweeklytimelines        = build_weekly_timelines
derivetargets               = derive_targets
cumulativesnapshot          = cumulative_snapshot
makesequencetensor          = make_sequence_tensor