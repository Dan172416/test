import pandas as pd

# TASK 4  (Higher/lower Possession )

# 1. EXTRACT RAW DATA


df = pd.read_csv("worldcup2026_raw.csv")

print("--RAW DATA--")
print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())



# 2. CHECK  QUALITY OF DATA



print("--DATA QUALITY CHECK--")
print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nNumber of unique matches:")
print(df["match_id"].nunique())


# 3. DATA CLEAN (NUMERIC VARIABLES)


df["possession"] = pd.to_numeric(
    df["possession"],
    errors="coerce"
)

df["goals"] = pd.to_numeric(
    df["goals"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "match_id",
        "team",
        "opponent",
        "possession",
        "goals"
    ]
).copy()



# 4. VALIDATE MATCH STRUCTURE ( one observation for each team that way each world cup
 # match will have two observations):

match_counts = df.groupby("match_id").size()
print("--MATCH VALIDATION--")
print("\nRows per match:")
print(match_counts.value_counts().sort_index())

valid_match_ids = match_counts[
    match_counts == 2
].index

df = df[
    df["match_id"].isin(valid_match_ids)
].copy()

print("\nValid matches:")
print(df["match_id"].nunique())

print("\nValid team-match observations:")
print(len(df))



# 5. HIGHER AND LOWER POSSESSION DIFFERENTIATION WITHIN EACH MATCH


df["max_possession"] = (
    df.groupby("match_id")["possession"]
      .transform("max")
)

df["min_possession"] = (
    df.groupby("match_id")["possession"]
      .transform("min")
)


def possession_group(row):

    if row["possession"] == row["max_possession"] and \
       row["possession"] != row["min_possession"]:
        return "Higher"

    elif row["possession"] == row["min_possession"] and \
         row["possession"] != row["max_possession"]:
        return "Lower"

    else:
        return "Equal"


df["possession_group"] = df.apply(
    possession_group,
    axis=1
)



# 6. CHECK GROUPS


print("--POSSESSION GROUPS--")


print(df["possession_group"].value_counts())

equal_matches = df[
    df["possession_group"] == "Equal"
]["match_id"].nunique()

print("\nMatches with equal possession:")
print(equal_matches)



# 7. REMOVE EQUAL-POSSESSION MATCHES since they cannot be classified as either of them.

equal_ids = df.loc[
    df["possession_group"] == "Equal",
    "match_id"
].unique()

analysis_population = df[
    ~df["match_id"].isin(equal_ids)
].copy()


analysis_population = analysis_population.drop(
    columns=["max_possession", "min_possession"]
)

# 8. FINAL POPULATION CHECK


print("ANALYSIS POPULATION")
print("\nNumber of matches:")
print(
    analysis_population["match_id"].nunique()
)

print("\nNumber of team-match observations:")
print(len(analysis_population))

print("\nGroup counts:")
print(
    analysis_population[
        "possession_group"
    ].value_counts()
)

print("\nFirst 10 observations:")
print(
    analysis_population.head(10).to_string(index=False)
)



# 9. SAVE CLEAN POPULATION
analysis_population.to_csv(
    "worldcup2026_clean_population.csv",
    index=False
)

print("\nClean population saved to:")
print("worldcup2026_clean_population.csv")


# 10. RANDOM SAMPLING done within match instead of individual team observations.

SAMPLE_MATCHES = 30  # 30 samples taken
RANDOM_SEED = 42

unique_matches = (
    analysis_population["match_id"]
    .drop_duplicates()
)

sampled_match_ids = unique_matches.sample(
    n=SAMPLE_MATCHES,
    random_state=RANDOM_SEED
)

sample = analysis_population[
    analysis_population["match_id"].isin(sampled_match_ids)
].copy()


print("SAMPLING")

print("\nPopulation matches:")
print(analysis_population["match_id"].nunique())

print("\nSampled matches:")
print(sample["match_id"].nunique())

print("\nSampled team-match observations:")
print(len(sample))

print("\nSample group counts:")
print(sample["possession_group"].value_counts())


# Save sample
sample.to_csv(
    "worldcup2026_sample.csv",
    index=False
)

# 11. DESCRIPTIVE STATISTICS

print("--DESCRIPTIVE STATISTICS--")


descriptive = sample.groupby(
    "possession_group"
)["goals"].agg(
    [
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max"
    ]
)

print("\nGoals scored by possession group:")
print(descriptive.round(3))


# 12. CREATE HIGHER AND LOWER POSSESSION SAMPLES

higher = sample[
    sample["possession_group"] == "Higher"
]["goals"]

lower = sample[
    sample["possession_group"] == "Lower"
]["goals"]


print("\nHigher possession mean goals:")
print(round(higher.mean(), 3))

print("\nLower possession mean goals:")
print(round(lower.mean(), 3))

print("\nDifference in sample means:")
print(round(higher.mean() - lower.mean(), 3))


# 13. 95% CONFIDENCE INTERVALS

from scipy import stats
import numpy as np


def mean_confidence_interval(data, confidence=0.95):

    n = len(data)

    mean = data.mean()

    standard_error = stats.sem(data)

    margin_error = stats.t.ppf(
        (1 + confidence) / 2,
        df=n - 1
    ) * standard_error

    lower_bound = mean - margin_error
    upper_bound = mean + margin_error

    return mean, lower_bound, upper_bound


higher_mean, higher_ci_low, higher_ci_high = (
    mean_confidence_interval(higher)
)

lower_mean, lower_ci_low, lower_ci_high = (
    mean_confidence_interval(lower)
)


print("--95% CONFIDENCE INTERVALS--")


print(
    "\nHigher-possession teams:",
    f"Mean = {higher_mean:.3f}, "
    f"95% CI = ({higher_ci_low:.3f}, {higher_ci_high:.3f})"
)

print(
    "\nLower-possession teams:",
    f"Mean = {lower_mean:.3f}, "
    f"95% CI = ({lower_ci_low:.3f}, {lower_ci_high:.3f})"
)


# 14. ONE-SAMPLE T-TEST OF WITHIN-MATCH GOAL DIFFERENCES

match_goals = sample.pivot(
    index="match_id",
    columns="possession_group",
    values="goals"
)

# Calculate within-match goal difference
match_goals["goal_difference"] = (
    match_goals["Higher"] - match_goals["Lower"]
)

goal_differences = match_goals["goal_difference"]



print("--WITHIN-MATCH GOAL DIFFERENCES--")


print("\nNumber of sampled matches:")
print(len(goal_differences))

print("\nMean goal difference:")
print(round(goal_differences.mean(), 3))

print("\nStandard deviation of goal differences:")
print(round(goal_differences.std(), 3))


# 95% CI FOR MEAN GOAL DIFFERENCE


n_diff = len(goal_differences)

mean_diff = goal_differences.mean()

se_diff = stats.sem(goal_differences)

margin_diff = stats.t.ppf(
    0.975,
    df=n_diff - 1
) * se_diff

ci_diff_low = mean_diff - margin_diff
ci_diff_high = mean_diff + margin_diff


print("\n95% CI for mean goal difference:")

print(
    f"({ci_diff_low:.3f}, {ci_diff_high:.3f})"
)



# 14.ONE-SAMPLE T-TEST

alpha = 0.05

t_statistic, p_value = stats.ttest_1samp(
    goal_differences,
    popmean=0
)

print("--ONE-SAMPLE T-TEST--")

print("\nH0: Mean goal difference = 0")
print("H1: Mean goal difference != 0")

print(f"\nt-statistic = {t_statistic:.4f}")
print(f"p-value = {p_value:.4f}")
print(f"Significance level = {alpha}")


if p_value < alpha:

    print("\nDecision: Reject the null hypothesis.")

    print(
        "Conclusion: There is statistically clear "
        "evidence that the mean number of goals scored does differ "
        "between higher-possession and lower-possession teams "
        "within the same match."
    )

else:

    print("\nDecision: Failed to reject the null hypothesis.")

    print(
        "Conclusion: There is insufficient statistical "
        "evidence that the mean number of goals scored differs "
        "between higher-possession and lower-possession teams "
        "within the same match."
    )


# Save match-level differences
match_goals.to_csv(
    "within_match_goal_differences.csv"
)


# 15. VISUALISATION (bar graph)


import matplotlib.pyplot as plt


group_means = sample.groupby(
    "possession_group"
)["goals"].mean()

# higher then lower in order
group_means = group_means.reindex(
    ["Higher", "Lower"]
)


plt.figure(figsize=(7, 5))

group_means.plot(
    kind="bar"
)

plt.title(
    "Mean Goals Scored by Possession Group\n"
    "FIFA World Cup 2026 Sample"
)

plt.xlabel("Possession Group")
plt.ylabel("Mean Goals Scored")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "mean_goals_by_possession.png",
    dpi=300
)

plt.show()



# 16. SAVE DESCRIPTIVE STATISTICS

descriptive.to_csv(
    "descriptive_statistics.csv"
)
plt.show()

print("--ANALYSIS COMPLETE--")

print("\nFiles created:")

print(
    "worldcup2026_clean_population.csv"
)

print(
    "worldcup2026_sample.csv"
)

print(
    "descriptive_statistics.csv"
)

print(
    "mean_goals_by_possession.png"
)
