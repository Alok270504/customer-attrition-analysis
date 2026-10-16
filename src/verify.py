"""Phase 8: check every resume number against what the code actually produces.

Run with `python src/verify.py`. Exits with code 1 if anything fails.
Each actual value is rounded to the precision the resume states, then compared exactly.
"""
import sys

from audit import is_text, total_charges_issue
from cohort import cohort_summary
from data import churned, load_clean, load_raw
from model import run_all


def build_checks() -> tuple[list[tuple[str, str, str]], dict]:
    raw = load_raw()
    df = load_clean()
    issue = total_charges_issue(raw)
    cohort = cohort_summary(df)
    models = run_all()
    b = models["baseline"]

    pct2 = lambda x: f"{x:.2f}%"
    pct1 = lambda x: f"{x:.1f}%"
    checks = [
        # (metric, expected, actual)
        ("Rows", "7,043", f"{len(raw):,}"),
        ("Overall churn rate", "26.54%", pct2(100 * churned(df).mean())),
        ("Majority-class accuracy (all customers)", "73.46%", pct2(100 * b["accuracy_all"])),
        ("TotalCharges stored as text", "True", str(is_text(raw["TotalCharges"]))),
        ("TotalCharges blank strings", "11", str(issue["blank_count"])),
        ("Blanks with tenure = 0", "11", str(int((issue["blank_rows"]["tenure"] == 0).sum()))),
        ("Rows kept after cleaning", "7,043", f"{len(df):,}"),
        ("Cohort size", "680", str(cohort["size"])),
        ("Cohort % of customers", "9.65%", pct2(cohort["pct_of_customers"])),
        ("Cohort churn rate", "67.65%", pct2(cohort["churn_rate_pct"])),
        ("Cohort % of all churners", "24.61%", pct2(cohort["pct_of_all_churners"])),
        ("Cohort 95% Wilson CI lower", "64.0%", pct1(cohort["ci_lower_pct"])),
        ("Cohort 95% Wilson CI upper", "71.1%", pct1(cohort["ci_upper_pct"])),
        ("Logistic regression test AUC", "0.846", f"{models['results']['Logistic regression']['auc']:.3f}"),
        ("Gradient boosting test AUC", "0.845", f"{models['results']['Gradient boosting']['auc']:.3f}"),
        ("Baseline AUC", "0.500", f"{b['auc']:.3f}"),
    ]
    return checks, {"test_accuracy": b["accuracy_test"], "test_size": len(models["y_test"]), "n": len(df)}


def print_table(checks) -> bool:
    width = max(len(m) for m, _, _ in checks)
    print(f"{'Metric':<{width}} | Expected | Actual   | Result")
    print("-" * (width + 32))
    all_pass = True
    for metric, expected, actual in checks:
        ok = expected == actual
        all_pass &= ok
        print(f"{metric:<{width}} | {expected:<8} | {actual:<8} | {'PASS' if ok else 'FAIL'}")
    return all_pass


if __name__ == "__main__":
    checks, extra = build_checks()
    passed = print_table(checks)
    # Not a resume number, printed so nobody is surprised by it in the notebook
    print(f"\nNote: on the {extra['test_size']:,}-customer test split alone, always predicting 'stayed' "
          f"scores {100 * extra['test_accuracy']:.2f}%;\nthe resume figure is the share of all "
          f"{extra['n']:,} customers who stayed.")
    print("\nALL PASS" if passed else "\nSOME CHECKS FAILED - do not publish until explained")
    sys.exit(0 if passed else 1)
