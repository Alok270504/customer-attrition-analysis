"""Phase 3: the high-risk cohort and a Wilson confidence interval for its churn rate."""
import math

import pandas as pd

from data import churned

Z_95 = 1.959964  # standard normal quantile for a two-sided 95% interval


def cohort_mask(df: pd.DataFrame) -> pd.Series:
    """Month-to-month contract AND pays by electronic check AND in the first 6 months."""
    return (
        (df["Contract"] == "Month-to-month")
        & (df["PaymentMethod"] == "Electronic check")
        & (df["tenure"] <= 6)
    )


def wilson_ci(successes: int, n: int, z: float = Z_95) -> dict:
    """Wilson score interval for a proportion, written out step by step.

    centre = (p + z^2 / 2n) / (1 + z^2 / n)
    margin = z * sqrt(p(1-p)/n + z^2 / 4n^2) / (1 + z^2 / n)
    Unlike the simple p +/- z*sqrt(p(1-p)/n), it never goes below 0 or above 1 and
    behaves well when p is far from 50%.
    """
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return {"p": p, "z": z, "denominator": denominator, "centre": centre,
            "margin": margin, "lower": centre - margin, "upper": centre + margin}


def pilot_customers_per_group(p_control: float, p_treatment: float,
                              alpha: float = 0.05, power: float = 0.80) -> int:
    """Customers needed in EACH of treatment and control to detect p_control -> p_treatment.

    Standard two-proportion formula:
    n = (z_{1-a/2} * sqrt(2 p_bar (1 - p_bar)) + z_{power} * sqrt(p1(1-p1) + p2(1-p2)))^2 / (p1 - p2)^2
    """
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    p_bar = (p_control + p_treatment) / 2
    numerator = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
                 + z_power * math.sqrt(p_control * (1 - p_control) + p_treatment * (1 - p_treatment))) ** 2
    return math.ceil(numerator / (p_control - p_treatment) ** 2)


def cohort_summary(df: pd.DataFrame) -> dict:
    mask = cohort_mask(df)
    left = churned(df)
    size = int(mask.sum())
    cohort_left = int(left[mask].sum())
    ci = wilson_ci(cohort_left, size)
    return {
        "size": size,
        "pct_of_customers": 100 * size / len(df),
        "customers_who_left": cohort_left,
        "churn_rate_pct": 100 * cohort_left / size,
        "pct_of_all_churners": 100 * cohort_left / int(left.sum()),
        "overall_churn_pct": 100 * left.mean(),
        "rest_churn_pct": 100 * left[~mask].mean(),
        "ci_lower_pct": 100 * ci["lower"],
        "ci_upper_pct": 100 * ci["upper"],
        "ci_steps": ci,
    }
