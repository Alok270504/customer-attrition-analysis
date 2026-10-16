"""Phase 4: baseline, logistic regression, gradient boosting. Run with `python src/model.py`."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import churned, load_clean

RANDOM_STATE = 42
NOT_FEATURES = ["customerID", "Churn"]  # an ID carries no signal, and Churn is the answer itself


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One-hot encode every column except the ID and the target.

    drop_first=True drops one category per column (e.g. keeps only 'gender_Male'),
    because the dropped one is implied when all the others are 0.
    """
    X = pd.get_dummies(df.drop(columns=NOT_FEATURES), drop_first=True).astype(float)
    y = churned(df)
    return X, y


def split(X: pd.DataFrame, y: pd.Series):
    # stratify=y keeps the 26.54% churn share the same in train and test
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)


def make_models() -> dict:
    return {
        # Scaling puts every feature on the same footing so coefficients are comparable
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "Gradient boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def baseline(y: pd.Series, y_train: pd.Series, y_test: pd.Series) -> dict:
    """Always predict 'stayed'. Its score is the same for everyone, so it cannot rank: AUC = 0.5.

    Accuracy is reported twice: on all 7,043 customers (= share who stayed) and on the
    1,761-customer test split, which differs slightly because 25% of 7,043 is not a whole
    number of each class.
    """
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
    test_preds = dummy.predict(np.zeros((len(y_test), 1)))
    test_scores = dummy.predict_proba(np.zeros((len(y_test), 1)))[:, 1]
    return {
        "accuracy_all": accuracy_score(y, dummy.predict(np.zeros((len(y), 1)))),
        "accuracy_test": accuracy_score(y_test, test_preds),
        "auc": roc_auc_score(y_test, test_scores),
    }


def fit_and_score(models: dict, X_train, X_test, y_train, y_test) -> dict:
    """Fit each model and return its test AUC and ROC curve points."""
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, scores)
        results[name] = {"model": model, "auc": roc_auc_score(y_test, scores), "fpr": fpr, "tpr": tpr}
    return results


def cross_validated_auc(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """5-fold stratified CV on all 7,043 rows: does the test-set AUC hold up on other splits?"""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in make_models().items():
        aucs = cross_val_score(model, X, y, cv=folds, scoring="roc_auc")
        rows.append({"model": name, "cv_auc_mean": aucs.mean(), "cv_auc_std": aucs.std(),
                     "fold_aucs": np.round(aucs, 3).tolist()})
    return pd.DataFrame(rows)


def top_coefficients(logreg_pipeline, feature_names, n: int = 10) -> pd.DataFrame:
    """Largest coefficients by size. Each is the change in log-odds of leaving per +1 standard
    deviation of that feature, holding the others fixed; exp(coef) is the odds multiplier."""
    coefs = logreg_pipeline.named_steps["logisticregression"].coef_[0]
    table = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    table["odds_multiplier"] = np.exp(table["coefficient"])
    table["direction"] = np.where(table["coefficient"] > 0, "more likely to leave", "less likely to leave")
    order = table["coefficient"].abs().sort_values(ascending=False).index
    return table.loc[order].head(n).reset_index(drop=True)


def reference_categories(df: pd.DataFrame) -> dict:
    """The category get_dummies(drop_first=True) dropped for each text column (the first, sorted)."""
    text_cols = df.drop(columns=NOT_FEATURES).select_dtypes(exclude="number").columns
    return {c: sorted(df[c].unique())[0] for c in text_cols}


def plain_english(feature: str, coefficient: float, references: dict) -> str:
    """Turn one coefficient into a sentence a non-technical manager can read."""
    direction = "higher" if coefficient > 0 else "lower"
    odds = np.exp(coefficient)
    if "_" in feature and feature.split("_", 1)[0] in references:
        # Yes/no indicators were standardised too, so the size is per SD of the indicator,
        # not the effect of switching from one category to another: use it to rank, not literally.
        column, value = feature.split("_", 1)
        return (f"{column} = '{value}' (vs. '{references[column]}') is linked with {direction} odds of leaving "
                f"(coefficient {coefficient:+.2f})")
    return (f"Each extra standard deviation of {feature} is linked with {direction} odds of leaving "
            f"(odds x {odds:.2f}, other features held fixed)")


def run_all() -> dict:
    df = load_clean()
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = split(X, y)
    results = fit_and_score(make_models(), X_train, X_test, y_train, y_test)
    return {
        "X": X, "y": y, "y_test": y_test,
        "baseline": baseline(y, y_train, y_test),
        "results": results,
        "cv": cross_validated_auc(X, y),
        "top_coefs": top_coefficients(results["Logistic regression"]["model"], X.columns),
    }


def main() -> None:
    out = run_all()
    print(f"Features: {out['X'].shape[1]} columns after one-hot encoding")
    b = out["baseline"]
    print(f"Baseline (always 'stayed'): accuracy {b['accuracy_all']:.2%} on all customers, "
          f"{b['accuracy_test']:.2%} on the test split; AUC {b['auc']:.3f}")
    for name, r in out["results"].items():
        print(f"{name}: test AUC {r['auc']:.3f}")
    print("\n5-fold stratified CV AUC:")
    for _, row in out["cv"].iterrows():
        print(f"  {row['model']}: {row['cv_auc_mean']:.3f} +/- {row['cv_auc_std']:.3f}  {row['fold_aucs']}")
    print("\nTop 10 logistic regression coefficients:")
    print(out["top_coefs"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
