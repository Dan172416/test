"""
FIFA World Cup 2026 - Disciplinary Cards Analysis
Comparing average yellow+red cards per 90 minutes between players aged 30+
and players under 23 (non-goalkeepers, >=1 match played).

REVISED VERSION: Uses random sample of n=30 per group (lecturer's requirement:
max sample size 30, population must be at least double, i.e. >= 60).

Run this script on your own machine (Jupyter, VS Code, or Google Colab) with
worldcup2026_stats.html saved in the same folder.

Pipeline:
1. Load saved FBref HTML table
2. Flatten multi-level headers, drop duplicated embedded header rows, fix dtypes
3. Define population per age group, check population >= 60, draw random sample n=30
4. Descriptive statistics
5. 95% CI via DescrStatsW.tconfint_mean()
6. Levene's test -> two-sample t-test via statsmodels ttest_ind
7. Generate descriptive statistics plots (bar chart with CI, boxplot)
"""

import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.stats.weightstats import DescrStatsW, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42
SAMPLE_SIZE = 30
HTML_PATH = "worldcup2026_stats.html"   # <-- change to your saved file's name

# ---------------------------------------------------------------------------
# STEP 1: Read the saved HTML file
# ---------------------------------------------------------------------------
with open(HTML_PATH, "r", encoding="utf-8") as f:
    html = f.read()

# FBref wraps some tables in HTML comments; strip them so pandas can see the table
html_clean = html.replace("<!--", "").replace("-->", "")
tables = pd.read_html(html_clean, header=[0, 1])

print(f"Total tables found on page: {len(tables)}")

# The correct "Player Standard Stats" table needs to be identified.
# Based on the original report, it was found at index 11 out of 13 tables.
# IMPORTANT: verify this yourself by inspecting a few tables, since the page
# layout may differ slightly for your saved copy:
#
# for i, t in enumerate(tables):
#     print(i, t.columns.tolist()[:6])
#
# Look for the table whose columns include Player, Pos, Age, Min, CrdY, CrdR.
df = tables[11].copy()

# ---------------------------------------------------------------------------
# STEP 2: Clean the data
# ---------------------------------------------------------------------------
# Flatten the two-row MultiIndex header, keeping only the bottom level
df.columns = [col[-1] if isinstance(col, tuple) else col for col in df.columns]

# Keep only the columns we need
df = df[["Player", "Pos", "Squad", "Age", "Born", "MP", "Min", "90s", "CrdY", "CrdR"]]

# Remove duplicate embedded header rows (FBref re-inserts "Player" header every ~25 rows)
df = df[df["Player"] != "Player"]
df = df.dropna(subset=["Player"])

# Fix data types
for col in ["MP", "Min", "90s", "CrdY", "CrdR"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Age on FBref is "YY-DDD" format (years-days); keep only the year part
df["Age"] = df["Age"].astype(str).str.split("-").str[0]
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

# Drop rows with missing key values, and players with 0 minutes
df = df.dropna(subset=["Age", "Min", "MP", "CrdY", "CrdR"])
df = df[df["Min"] > 0]

print(f"Total cleaned records: {len(df)}")

# ---------------------------------------------------------------------------
# STEP 3: Define population, check size, and draw random sample n=30
# ---------------------------------------------------------------------------
# Exclude goalkeepers (exact match, consistent with the original report)
mask_outfield = df["Pos"] != "GK"
mask_played = df["MP"] >= 1
base = df[mask_outfield & mask_played].copy()

# Cards per 90 minutes = (Yellow + Red) / 90s played
base["Cards90"] = (base["CrdY"] + base["CrdR"]) / base["90s"]
base = base.replace([np.inf, -np.inf], np.nan).dropna(subset=["Cards90"])

pop_30plus = base[base["Age"] >= 30]
pop_u23 = base[base["Age"] < 23]

print(f"Population size, 30+: {len(pop_30plus)}")
print(f"Population size, under 23: {len(pop_u23)}")

# Lecturer's rule: population must be at least double the sample size (30),
# i.e. at least 60 players in each group, before we may sample.
for name, pop in [("30+", pop_30plus), ("under 23", pop_u23)]:
    if len(pop) < 2 * SAMPLE_SIZE:
        raise ValueError(
            f"Population for group '{name}' has only {len(pop)} players; "
            f"need at least {2 * SAMPLE_SIZE} to draw a valid n={SAMPLE_SIZE} sample."
        )
    print(f"Population check passed for '{name}': {len(pop)} >= {2 * SAMPLE_SIZE}")

sample_30plus = pop_30plus.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
sample_u23 = pop_u23.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

s1 = sample_30plus["Cards90"].to_numpy()
s2 = sample_u23["Cards90"].to_numpy()

# ---------------------------------------------------------------------------
# STEP 4: Descriptive statistics
# ---------------------------------------------------------------------------
def describe(sample, label):
    return {
        "group": label, "n": len(sample), "mean": np.mean(sample),
        "median": np.median(sample), "std": np.std(sample, ddof=1),
        "min": np.min(sample), "max": np.max(sample)
    }

desc_30plus = describe(s1, "30+")
desc_u23 = describe(s2, "Under 23")
desc_table = pd.DataFrame([desc_30plus, desc_u23])
print("\n=== Descriptive Statistics ===")
print(desc_table)

# ---------------------------------------------------------------------------
# STEP 5: 95% confidence interval for each group's mean
# ---------------------------------------------------------------------------
d1 = DescrStatsW(s1)
d2 = DescrStatsW(s2)
ci_30plus = d1.tconfint_mean(alpha=0.05)
ci_u23 = d2.tconfint_mean(alpha=0.05)

print(f"\n95% CI, 30+ group:      {ci_30plus}")
print(f"95% CI, under-23 group: {ci_u23}")

# ---------------------------------------------------------------------------
# STEP 6: Levene's test, then two-sample t-test
# ---------------------------------------------------------------------------
levene_stat, levene_p = st.levene(s1, s2)
usevar = "pooled" if levene_p > 0.05 else "unequal"
tstat, pval, dof = ttest_ind(s1, s2, usevar=usevar)

print(f"\nLevene's test: statistic={levene_stat:.4f}, p-value={levene_p:.4f}")
print(f"Variance assumption used for t-test: {usevar}")
print(f"t-statistic: {tstat:.4f}")
print(f"p-value:     {pval:.4f}")
print(f"df:          {dof:.2f}")

conclusion = (
    "Reject H0: there IS a statistically significant difference in mean cards "
    "per 90 minutes between players 30+ and players under 23."
    if pval < 0.05 else
    "Fail to reject H0: no statistically significant difference detected in "
    "mean cards per 90 minutes between the two age groups."
)
print(conclusion)

# ---------------------------------------------------------------------------
# STEP 7: Descriptive statistics visualizations
# ---------------------------------------------------------------------------
# --- Chart 1: Bar chart of means with 95% CI (unchanged) ---
plot_data = pd.DataFrame({
    "Age Group": ["30+", "Under 23"],
    "Mean Cards per 90": [np.mean(s1), np.mean(s2)],
})
means = [np.mean(s1), np.mean(s2)]
ci_lower = [ci_30plus[0], ci_u23[0]]
ci_upper = [ci_30plus[1], ci_u23[1]]
error_lower = np.clip(np.array(means) - np.array(ci_lower), 0, None)
error_upper = np.array(ci_upper) - np.array(means)

plt.figure(figsize=(8, 5))
plt.bar(plot_data["Age Group"], plot_data["Mean Cards per 90"],
        yerr=[error_lower, error_upper], capsize=6,
        color=["steelblue", "darkorange"], edgecolor="black", linewidth=1.2)
plt.title("Mean Disciplinary Cards per 90 Minutes by Age Group (n=30 per group)")
plt.xlabel("Age Group")
plt.ylabel("Mean Yellow + Red Cards per 90 Minutes")
plt.grid(axis="y", alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig("descriptive_statistics_mean_ci.png", dpi=300, bbox_inches="tight")
plt.show()

# --- Chart 2: HISTOGRAM (replaces boxplot+stripplot) ---
# Use common bin edges so both groups are directly comparable on the same scale.
bin_edges = np.linspace(0, 10, 11)  # bins: 0-1, 1-2, ..., 9-10

# Option A: Side-by-side histograms (two panels) - clearer when scales differ a lot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(s1, bins=bin_edges, color="steelblue", edgecolor="black", alpha=0.85)
axes[0].set_title("30+ Group (n=30)")
axes[0].set_xlabel("Cards per 90 Minutes")
axes[0].set_ylabel("Frequency (number of players)")
axes[0].grid(axis="y", alpha=0.3, linestyle="--")

axes[1].hist(s2, bins=bin_edges, color="darkorange", edgecolor="black", alpha=0.85)
axes[1].set_title("Under 23 Group (n=30)")
axes[1].set_xlabel("Cards per 90 Minutes")
axes[1].set_ylabel("Frequency (number of players)")
axes[1].grid(axis="y", alpha=0.3, linestyle="--")

plt.suptitle("Distribution of Disciplinary Cards per 90 Minutes (n=30 per group)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("descriptive_statistics_histogram_sidebyside.png", dpi=300, bbox_inches="tight")
plt.show()

# Option B: Overlay histogram (single panel, both groups on one chart)
plt.figure(figsize=(8, 5))
plt.hist(s1, bins=bin_edges, alpha=0.6, color="steelblue", edgecolor="black", label="30+")
plt.hist(s2, bins=bin_edges, alpha=0.6, color="darkorange", edgecolor="black", label="Under 23")
plt.title("Distribution of Disciplinary Cards per 90 Minutes (n=30 per group)")
plt.xlabel("Cards per 90 Minutes")
plt.ylabel("Frequency (number of players)")
plt.legend()
plt.grid(axis="y", alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig("descriptive_statistics_histogram_overlay.png", dpi=300, bbox_inches="tight")
plt.show()
# ---------------------------------------------------------------------------
# Save tidy summaries for the report
# ---------------------------------------------------------------------------
summary = pd.DataFrame([
    {**desc_30plus, "ci_lower": ci_30plus[0], "ci_upper": ci_30plus[1]},
    {**desc_u23, "ci_lower": ci_u23[0], "ci_upper": ci_u23[1]},
])
summary.to_csv("cards_per_90_summary.csv", index=False)

test_results = pd.DataFrame([{
    "levene_stat": levene_stat, "levene_p": levene_p,
    "variance_assumption": usevar, "t_stat": tstat, "p_value": pval, "df": dof,
}])
test_results.to_csv("ttest_results.csv", index=False)

print("\nSaved: cards_per_90_summary.csv, ttest_results.csv, and two PNG charts.")
