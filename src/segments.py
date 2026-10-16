"""Phase 2: churn rate by segment, chi-square test, and Cramer's V."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from data import churned

TENURE_BINS = [-1, 6, 12, 24, 48, np.inf]  # -1 so that tenure 0 lands in the first band
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49+"]
SEGMENT_COLUMNS = ["Contract", "PaymentMethod", "InternetService", "TenureBand"]


def add_tenure_band(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TenureBand"] = pd.cut(out["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS)
    return out


def churn_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Customers, customers who left, and churn rate (%) for each value of `column`."""
    left = churned(df)
    table = (
        pd.DataFrame({column: df[column], "left": left})
        .groupby(column, observed=True)["left"]
        .agg(customers="size", customers_who_left="sum")
    )
    table["churn_rate_pct"] = (100 * table["customers_who_left"] / table["customers"]).round(2)
    return table


def chi_square_with_cramers_v(df: pd.DataFrame, column: str) -> dict:
    """Is churn independent of `column`? Chi-square answers yes/no; Cramer's V says how strongly.

    Cramer's V = sqrt(chi2 / (n * (min(rows, cols) - 1))), from 0 (no link) to 1 (perfect link).
    p-values shrink with sample size, so V is the fairer way to rank segments against each other.
    """
    contingency = pd.crosstab(df[column], df["Churn"])
    chi2, p_value, dof, _ = chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    k = min(contingency.shape) - 1
    return {
        "segment": column,
        "chi2": round(float(chi2), 1),
        "dof": int(dof),
        "p_value": float(p_value),
        "cramers_v": round(float(np.sqrt(chi2 / (n * k))), 3),
    }


def effect_size_ranking(df: pd.DataFrame) -> pd.DataFrame:
    results = [chi_square_with_cramers_v(df, col) for col in SEGMENT_COLUMNS]
    return pd.DataFrame(results).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def finding_sentence(table: pd.DataFrame, column: str, overall_pct: float) -> str:
    """One plain-English sentence from the highest- and lowest-churn values of a segment."""
    worst = table["churn_rate_pct"].idxmax()
    best = table["churn_rate_pct"].idxmin()
    ratio = table.loc[worst, "churn_rate_pct"] / table.loc[best, "churn_rate_pct"]
    return (
        f"By {column}: '{worst}' customers left at {table.loc[worst, 'churn_rate_pct']:.2f}% "
        f"versus {table.loc[best, 'churn_rate_pct']:.2f}% for '{best}' "
        f"({ratio:.1f}x; overall rate {overall_pct:.2f}%)."
    )
