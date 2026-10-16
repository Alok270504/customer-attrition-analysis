# Interview Prep: Customer Attrition Analysis

The 12 questions an interviewer is most likely to ask, with short, honest answers. Every number is
reproduced by `python src/verify.py`.

---

**1. Walk me through the project in 30 seconds.**
I took IBM's Telco dataset of 7,043 customers, 26.54% of whom left. I audited the data and found a hidden
text-type problem. I ranked segments by how strongly they're linked to leaving, and found a cohort of 680
new, month-to-month, electronic-check customers: 9.65% of the base, but 24.61% of everyone who left. A
logistic regression ranks customers by risk with test AUC 0.846. I proposed a randomized retention pilot
on that cohort. I didn't implement it.

**2. Why did you fill the 11 blank TotalCharges with 0 instead of dropping them?**
All 11 belong to customers with tenure 0. They'd signed up but hadn't been billed, so $0 is the true
value, not a guess. Dropping them would remove real (brand-new) customers and nudge every rate. The
blanks also didn't show up in a null check, because a blank string isn't "missing" to pandas. That's why
I check data types, not just null counts.

**3. What does AUC measure?**
Pick one customer who left and one who stayed at random. AUC is the probability that the model gives the
leaver the higher risk score. 0.5 is a coin flip and 1.0 is perfect. It measures **ranking**, not whether
any single prediction is right, which suits a retention team deciding whom to call first.

**4. Your baseline is 73.46% accurate. Why don't you compare 0.846 to 73.46%?**
They measure different things on different scales. 73.46% is the accuracy of always saying "stayed".
That's just the share of customers who stayed, and it catches zero leavers. Because that baseline gives
everyone the same score, its AUC is 0.5. The fair comparison is AUC vs. AUC: 0.846 vs 0.5. (Small detail:
on the 1,761-customer test split alone, the always-"stayed" accuracy is 73.48%. 73.46% is the
full-dataset figure.)

**5. Why is the cohort not causal?**
It's an observed pattern in historical data. Customers who pick month-to-month and electronic check may
just be less committed to begin with, and that could drive both their choices and their leaving.
Switching someone's payment method might change nothing. To claim cause, you need to randomly assign an
intervention and compare against a control group.

**6. How would you test a retention offer?**
I'd run a randomized pilot. As new customers enter the cohort, randomly assign them to offer or no offer.
Measure 90-day churn in each group, and compare with a two-proportion test. Decide in advance the smallest
effect worth acting on. For a drop from 67.65% to 57.65%, at 5% significance and 80% power, that's 367
customers per group. Guardrails: offer cost below the margin saved, and track whether the discount mostly
goes to people who'd have stayed anyway.

**7. Why chi-square and Cramér's V, and why rank by V rather than p-value?**
Chi-square tests whether leaving is independent of a segment. With 7,043 customers every p-value is tiny,
so p-values can't tell you which segment matters *most*. Cramér's V is an effect size from 0 to 1. Contract
came out strongest (0.410), then tenure band (0.363), internet service (0.322) and payment method (0.303).

**8. Why a Wilson interval instead of the usual ± 1.96 × standard error?**
The simple interval behaves badly when a rate is far from 50% or the sample is small, and can even go
below 0 or above 100%. Wilson stays accurate in those cases. For the cohort it gives 64.0% to 71.1%
around 67.65%. The whole interval sits far above the 26.54% overall rate, so the gap isn't sampling noise.

**9. Why recommend logistic regression over gradient boosting?**
They tie: test AUC 0.846 vs 0.845, and 5-fold CV 0.845 ± 0.013 vs 0.847 ± 0.011. The gap is smaller than
the fold-to-fold spread. When accuracy ties, I pick the model I can explain to a manager, audit, and run
cheaply. Logistic regression gives one readable coefficient per feature.

**10. Why stratify the split, and why `drop_first=True`?**
Stratifying keeps the share of customers who left the same in train and test, so the test score isn't
skewed by a lucky split. `drop_first=True` drops one category per column, e.g. keeping only
"Contract = One year" and "Two year". The dropped one (month-to-month) becomes the reference, because it's
implied when the others are 0. That avoids perfectly redundant columns.

**11. Any coefficient that surprised you? Can you trust them?**
MonthlyCharges came out *negative*, even though higher-paying customers leave more in the raw data. That
happens because tenure, monthly and total charges overlap heavily (total ≈ tenure × monthly), and the
add-on services already explain much of the price. So the model splits credit between them. I read
coefficients as the model's ranking logic, not as separate causal effects. The stable messages are that
longer tenure and longer contracts go with staying, and fiber internet goes with leaving.

**12. What are the limitations, and what would you do next?**
Limitations: it's a fictional snapshot with no timeline, cost or margin data, and every finding is
associational. Next steps:
- uplift modeling, to target customers an offer would actually change
- survival analysis, to model *when* customers leave
- SHAP, for per-customer explanations
- CLV-weighted risk, to prioritise by revenue at stake
- a live risk-score dashboard for the retention team
