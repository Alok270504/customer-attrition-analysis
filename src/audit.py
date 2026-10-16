"""Phase 1: data audit. Run with `python src/audit.py`."""
import pandas as pd

from data import blank_total_charges, clean, load_raw

# Columns where "No internet service" / "No phone service" is a third value next to "Yes"/"No"
SERVICE_COLUMNS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def is_text(series: pd.Series) -> bool:
    """True if a column is stored as text (object dtype in pandas 2, str dtype in pandas 3)."""
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def basic_profile(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "null_counts": df.isnull().sum(),
    }


def total_charges_issue(df: pd.DataFrame) -> dict:
    """Show that TotalCharges is text and that the blanks hide from null checks."""
    blanks = blank_total_charges(df)
    blank_rows = df.loc[blanks, ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]]
    return {
        "total_charges_is_text": is_text(df["TotalCharges"]),
        "total_charges_nulls": int(df["TotalCharges"].isnull().sum()),
        "blank_count": int(blanks.sum()),
        "blank_rows": blank_rows,
        "all_blanks_tenure_zero": bool((blank_rows["tenure"] == 0).all()),
        # The reverse check: is every tenure-0 customer a blank? If so, blank == "never billed"
        "tenure_zero_count": int((df["tenure"] == 0).sum()),
    }


def duplicate_ids(df: pd.DataFrame) -> int:
    return int(df["customerID"].duplicated().sum())


def category_values(df: pd.DataFrame) -> dict:
    """Unique values of every text column except the ID and TotalCharges."""
    text_cols = [c for c in df.columns if is_text(df[c]) and c not in ("customerID", "TotalCharges")]
    return {c: sorted(df[c].unique()) for c in text_cols}


def no_service_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Count how often each service column says 'No internet/phone service' instead of 'No'."""
    rows = []
    for col in SERVICE_COLUMNS:
        counts = df[col].value_counts()
        no_service = counts.filter(like="service").sum()
        rows.append({"column": col, "No": counts.get("No", 0), "No ... service": no_service})
    return pd.DataFrame(rows)


def no_service_is_consistent(df: pd.DataFrame) -> dict:
    """'No internet service' should appear exactly where InternetService == 'No' (same for phone).

    If true, the third value is not a data-entry error: it is a structural 'No' that tells us
    *why* the customer lacks the add-on. We keep it as-is for modeling.
    """
    no_internet = df["InternetService"] == "No"
    internet_ok = all(((df[c] == "No internet service") == no_internet).all()
                      for c in SERVICE_COLUMNS if c != "MultipleLines")
    phone_ok = bool(((df["MultipleLines"] == "No phone service") == (df["PhoneService"] == "No")).all())
    return {"internet_consistent": internet_ok, "phone_consistent": phone_ok}


def main() -> None:
    raw = load_raw()
    profile = basic_profile(raw)
    print("Shape:", profile["shape"])
    print("\nDtypes:\n", profile["dtypes"].to_string())
    print("\nNull counts (total):", int(profile["null_counts"].sum()))

    issue = total_charges_issue(raw)
    print("\nTotalCharges stored as text:", issue["total_charges_is_text"])
    print("TotalCharges nulls reported by isnull():", issue["total_charges_nulls"])
    print("Blank-string TotalCharges:", issue["blank_count"])
    print("All blanks have tenure 0:", issue["all_blanks_tenure_zero"])
    print("Customers with tenure 0 in total:", issue["tenure_zero_count"])
    print(issue["blank_rows"].to_string(index=False))

    print("\nDuplicate customerIDs:", duplicate_ids(raw))
    print("\nCategory values:")
    for col, values in category_values(raw).items():
        print(f"  {col}: {values}")
    print("\n'No' vs 'No ... service' in service columns:\n", no_service_overlap(raw).to_string(index=False))
    print("'No ... service' lines up with InternetService/PhoneService:", no_service_is_consistent(raw))

    df = clean(raw)
    print("\nAfter cleaning: rows =", len(df), "| TotalCharges dtype =", df["TotalCharges"].dtype,
          "| zeros filled =", int((df["TotalCharges"] == 0).sum()))


if __name__ == "__main__":
    main()
