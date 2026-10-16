"""Load and clean the IBM Telco Customer Churn data.

Every other module gets its data from here, so the cleaning rules live in one place.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_COPY = PROJECT_ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DOWNLOADED_COPY = PROJECT_ROOT / "data" / "telco_churn.csv"


def data_path() -> Path:
    """Prefer the original Kaggle/IBM file if it sits in the project root, else the downloaded copy."""
    return LOCAL_COPY if LOCAL_COPY.exists() else DOWNLOADED_COPY


def load_raw() -> pd.DataFrame:
    """Read the CSV exactly as stored, with no type fixes."""
    return pd.read_csv(data_path())


def blank_total_charges(df: pd.DataFrame) -> pd.Series:
    """True for rows where TotalCharges is an empty or whitespace-only string."""
    return df["TotalCharges"].astype(str).str.strip() == ""


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return an analysis-ready copy.

    The only change is TotalCharges: the 11 blank strings belong to customers with
    tenure 0 who have never been billed, so their true total is 0. We fill them with 0
    and keep every row.
    """
    df = raw.copy()
    blanks = blank_total_charges(df)
    df.loc[blanks, "TotalCharges"] = "0"
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])  # fails loudly if any other text remains
    assert len(df) == len(raw), "cleaning must never drop rows"
    return df


def load_clean() -> pd.DataFrame:
    return clean(load_raw())


def churned(df: pd.DataFrame) -> pd.Series:
    """1 for customers who left, 0 for customers who stayed."""
    return (df["Churn"] == "Yes").astype(int)
