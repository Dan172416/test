"""
PRT551 - Objective 1 - Analytic Task
Question: On average, how many shots (total) and shots on target (SOT) does
a team register per match at the FIFA World Cup 2026 group stage, and is
there a significant difference in EACH of these between winning teams and
losing teams? (i.e. do winners out-shoot losers, out-accurate them, or both?)

Sections:
  1. Data wrangling      - load & clean the team-match dataset
  2. Data prep & sampling- define population, draw a simple random sample
  3. Descriptive stats   - mean/median/sd/IQR for shots & SOT, by result
  4. Confidence interval - 95% CI for population mean SOT
  5. Hypothesis tests    - two-sample (Welch) t-tests, winners vs losers,
                            run separately for shots_total and shots_on_target

NOTE ON DATA:
  This script expects a CSV called 'team_match_data.csv' with one row per
  team per WC2026 group-stage match, produced by clean_group_stage_data.py
  from thestatsdontlie.com data:
      match_id, team, opponent, result, shots_on_target, shots_total, goals
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import os

RNG_SEED = 42
np.random.seed(RNG_SEED)

# ============================================================
# 1. DATA WRANGLING
# ============================================================
df = pd.read_csv("team_match_data.csv")

# Basic cleaning: drop rows with missing shots/SOT, enforce valid result labels
df = df.dropna(subset=["shots_on_target", "shots_total"])
df = df[df["result"].isin(["Win", "Loss", "Draw"])]

print(f"Population size (all team-match rows): {len(df)}")
print(df["result"].value_counts(), "\n")

# ============================================================
# 2. DATA PREPARATION & SAMPLING
# ============================================================
# Population  : all team-match observations, WC2026 group stage
# Unit        : one team's performance in one match
# Variables   : shots_total, shots_on_target
# Sample      : simple random sample of n = 40 team-match rows
SAMPLE_SIZE = 40
sample = df.sample(n=SAMPLE_SIZE, random_state=RNG_SEED).reset_index(drop=True)
sample.to_csv("sample_shots_on_target.csv", index=False)
print(f"Drew a simple random sample of n = {len(sample)} team-match rows.\n")

# For the two-sample comparisons, drop draws (no winner/loser to compare)
sample_decisive = sample[sample["result"].isin(["Win", "Loss"])]
win_mask = sample_decisive["result"] == "Win"
loss_mask = sample_decisive["result"] == "Loss"
print(f"Decisive-result rows kept for t-tests: {len(sample_decisive)} "
      f"(winners={win_mask.sum()}, losers={loss_mask.sum()})\n")

# ============================================================
# 3. DESCRIPTIVE STATISTICS - for BOTH shots_total and shots_on_target
# ============================================================
def describe(x, label):
    print(f"--- {label} (n={len(x)}) ---")
    print(f"mean   : {x.mean():.2f}")
    print(f"median : {x.median():.2f}")
    print(f"sd     : {x.std(ddof=1):.2f}")
    q1, q3 = x.quantile([0.25, 0.75])
    print(f"IQR    : {q3 - q1:.2f}  (Q1={q1:.2f}, Q3={q3:.2f})\n")

VARS = {"shots_total": "Total shots", "shots_on_target": "Shots on target"}
groups = {}  # groups[var]["Winners"/"Losers"] = pandas Series

for var, label in VARS.items():
    winners = sample_decisive.loc[win_mask, var]
    losers = sample_decisive.loc[loss_mask, var]
    groups[var] = {"Winners": winners, "Losers": losers}
    describe(sample[var], f"All sampled team-matches - {label}")
    describe(winners, f"Winning teams - {label}")
    describe(losers, f"Losing teams - {label}")

# --- Combined boxplot: Shots vs SOT, each split by Winners/Losers ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
for ax, (var, label) in zip(axes, VARS.items()):
    ax.boxplot(
        [groups[var]["Winners"], groups[var]["Losers"]],
        tick_labels=["Winners", "Losers"],
    )
    ax.set_title(label)
    ax.set_ylabel(label)
fig.suptitle("Shots vs Shots on Target: Winners vs Losers (sample)")
plt.tight_layout()
plt.savefig("shots_vs_sot_boxplot.png", dpi=150)
print("Saved combined boxplot to shots_vs_sot_boxplot.png\n")

# --- Combined histogram: same layout, overlaid Winners/Losers per panel ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, (var, label) in zip(axes, VARS.items()):
    bins = range(0, int(sample[var].max()) + 2)
    ax.hist(groups[var]["Winners"], bins=bins, alpha=0.6, label="Winners",
            color="#4C72B0", edgecolor="black")
    ax.hist(groups[var]["Losers"], bins=bins, alpha=0.6, label="Losers",
            color="#DD8452", edgecolor="black")
    ax.set_title(label)
    ax.set_xlabel(label)
    ax.set_ylabel("Frequency")
    ax.legend()
fig.suptitle("Distribution of Shots vs Shots on Target: Winners vs Losers")
plt.tight_layout()
plt.savefig("shots_vs_sot_histogram.png", dpi=150)
print("Saved combined histogram to shots_vs_sot_histogram.png\n")

# ============================================================
# 4. CONFIDENCE INTERVAL (95%, population mean SOT, sigma unknown)
# ============================================================
x = sample["shots_on_target"]
n = len(x)
mean = x.mean()
se = x.std(ddof=1) / np.sqrt(n)
t_crit = stats.t.ppf(0.975, df=n - 1)
ci_low, ci_high = mean - t_crit * se, mean + t_crit * se

print("=== 95% Confidence Interval for population mean SOT ===")
print(f"x̄ = {mean:.3f}, s = {x.std(ddof=1):.3f}, n = {n}, t* = {t_crit:.3f}")
print(f"95% CI: ({ci_low:.3f}, {ci_high:.3f})\n")

# ============================================================
# 5. HYPOTHESIS TESTS - two-sample Welch's t-test, run for BOTH variables
#    H0: mu_win = mu_loss   Ha: mu_win > mu_loss   (one-tailed, each variable)
# ============================================================
def interpret_d(d):
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

def run_two_sample_test(var, label):
    winners = groups[var]["Winners"]
    losers = groups[var]["Losers"]
    n1, n2 = len(winners), len(losers)

    t_stat, p_two = stats.ttest_ind(winners, losers, equal_var=False)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2

    s1, s2 = winners.std(ddof=1), losers.std(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    d = (winners.mean() - losers.mean()) / pooled_sd

    print(f"=== Two-sample t-test (Welch): {label}, winners vs losers ===")
    print(f"mean winners = {winners.mean():.3f} (n={n1}), "
          f"mean losers = {losers.mean():.3f} (n={n2})")
    print(f"t = {t_stat:.3f}, one-tailed p = {p_one:.4f}, "
          f"Cohen's d = {d:.3f} ({interpret_d(d)} effect)")
    alpha = 0.05
    if p_one < alpha:
        print(f"p < {alpha}: reject H0 -> winners have significantly higher mean {label}.\n")
    else:
        print(f"p >= {alpha}: fail to reject H0 -> no significant evidence for {label}.\n")
    return t_stat, p_one, d

results = {}
for var, label in VARS.items():
    results[var] = run_two_sample_test(var, label)

print("Summary - which matters more, shot VOLUME or shot ACCURACY?")
for var, label in VARS.items():
    t_stat, p_one, d = results[var]
    verdict = "significant" if p_one < 0.05 else "not significant"
    print(f"  {label:18s}: p = {p_one:.4f} ({verdict}), Cohen's d = {d:.3f} ({interpret_d(d)})")

print("\nNote: draws are excluded from these group comparisons (no winner/loser "
      "to assign). Sub-sample sizes are small and unequal, which limits "
      "statistical power - Cohen's d is reported alongside each p-value.")