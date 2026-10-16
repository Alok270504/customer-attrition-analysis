"""Phase 5: save the five charts to outputs/charts/. Run with `python src/charts.py`."""
import matplotlib

matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

from cohort import cohort_summary
from data import PROJECT_ROOT, churned, load_clean
from segments import add_tenure_band, churn_by

CHART_DIR = PROJECT_ROOT / "outputs" / "charts"
HIGHLIGHT = "#2a78d6"  # the bar the title talks about
MUTED = "#c4c3bd"      # every other bar
SECOND = "#eb6834"     # second series on the ROC chart
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": TEXT_SECONDARY, "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "text.color": TEXT,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def _bar_chart(labels, rates, highlight_index, title, subtitle, filename, overall_pct=None):
    """Vertical bars with 1-decimal % labels; one highlighted bar; optional dashed overall line."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    colors = [HIGHLIGHT if i == highlight_index else MUTED for i in range(len(labels))]
    bars = ax.bar(labels, rates, color=colors, width=0.6, edgecolor=SURFACE, linewidth=2)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 1, f"{rate:.1f}%",
                ha="center", va="bottom", fontsize=11, color=TEXT)
    if overall_pct is not None:
        ax.axhline(overall_pct, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.text(len(labels) - 0.5, overall_pct + 1, f"All customers: {overall_pct:.1f}%",
                ha="right", va="bottom", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel("Share of customers who left (%)")
    ax.set_ylim(0, max(rates) * 1.18)
    ax.set_title(title, pad=24)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(CHART_DIR / filename, dpi=150)
    plt.close(fig)


def _segment_chart(df, column, readable_name, make_title, filename, short_labels=None):
    """Bar chart for one segment column; make_title(worst, best, ratio) builds the takeaway title."""
    short_labels = short_labels or {}
    table = churn_by(df, column)
    worst = table["churn_rate_pct"].idxmax()
    best = table["churn_rate_pct"].idxmin()
    ratio = table.loc[worst, "churn_rate_pct"] / table.loc[best, "churn_rate_pct"]
    labels = [short_labels.get(x, str(x)) for x in table.index]
    _bar_chart(labels, table["churn_rate_pct"].tolist(), list(table.index).index(worst),
               make_title(worst, best, ratio), f"Share who left, by {readable_name} ({len(df):,} customers)",
               filename, 100 * churned(df).mean())


def contract_chart(df):
    _segment_chart(df, "Contract", "contract type",
                   lambda w, b, r: f"{w} customers leave at {r:.0f}x the rate of {b.lower()} customers",
                   "churn_by_contract.png")


def tenure_chart(df):
    _segment_chart(df, "TenureBand", "months as a customer",
                   lambda w, b, r: f"Customers in months {w} leave at {r:.1f}x the rate of those at {b} months",
                   "churn_by_tenure_band.png", short_labels={x: f"{x} months" for x in ["0-6", "7-12", "13-24", "25-48", "49+"]})


def payment_chart(df):
    short = {"Bank transfer (automatic)": "Bank transfer\n(automatic)",
             "Credit card (automatic)": "Credit card\n(automatic)",
             "Electronic check": "Electronic\ncheck", "Mailed check": "Mailed\ncheck"}
    _segment_chart(df, "PaymentMethod", "payment method",
                   lambda w, b, r: f"{w} payers leave at {r:.1f}x the rate of {b.lower()} payers",
                   "churn_by_payment.png", short_labels=short)


def cohort_chart(df):
    s = cohort_summary(df)
    labels = [f"Risk cohort\n({s['size']:,} customers)",
              f"Everyone else\n({len(df) - s['size']:,} customers)",
              f"All customers\n({len(df):,})"]
    rates = [s["churn_rate_pct"], s["rest_churn_pct"], s["overall_churn_pct"]]
    title = (f"The risk cohort is {s['pct_of_customers']:.1f}% of customers "
             f"but {s['pct_of_all_churners']:.1f}% of everyone who left")
    subtitle = ("Month-to-month + electronic check + tenure <= 6 months; "
                f"cohort 95% CI {s['ci_lower_pct']:.1f}%-{s['ci_upper_pct']:.1f}%")
    _bar_chart(labels, rates, 0, title, subtitle, "cohort_vs_baseline.png")


def roc_chart(model_output: dict):
    fig, ax = plt.subplots(figsize=(6.4, 6))
    for (name, r), color in zip(model_output["results"].items(), [HIGHLIGHT, SECOND]):
        ax.plot(r["fpr"], r["tpr"], color=color, linewidth=2, label=f"{name} (AUC {r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linestyle="--", linewidth=1,
            label="Random ranking / always 'stayed' (AUC 0.500)")
    ax.set_xlabel("False positive rate (stayers wrongly flagged)")
    ax.set_ylabel("True positive rate (leavers correctly flagged)")
    aucs = [r["auc"] for r in model_output["results"].values()]
    ax.set_title(f"Both models rank customers about equally well\n(test AUC {aucs[0]:.3f} vs {aucs[1]:.3f})")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)


def save_all(model_output: dict | None = None) -> list:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    df = add_tenure_band(load_clean())
    contract_chart(df)
    tenure_chart(df)
    payment_chart(df)
    cohort_chart(df)
    if model_output is None:
        from model import run_all
        model_output = run_all()
    roc_chart(model_output)
    return sorted(p.name for p in CHART_DIR.glob("*.png"))


if __name__ == "__main__":
    print(save_all())
