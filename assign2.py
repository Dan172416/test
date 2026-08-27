import pandas as pd
import statistics as stats
import numpy as np
import scipy.stats as st
import math

# Read the datasets
goalkeeper_data = pd.read_csv(
    r"C:\Users\User\Documents\VSCODE_DEMO\Data science\fifa_goalkeeper_statistics.csv",
    encoding="utf-8-sig"
)

match_counts = pd.read_csv(
    r"C:\Users\User\Documents\VSCODE_DEMO\Data science\fifa_2026_team_match_counts.csv",
    encoding="utf-8-sig"
)

# Standardise team names
match_counts["Team"] = match_counts["Team"].replace({
    "Bosnia–Herz": "Bosnia and Herzegovina"
})

# Merge the two datasets
fifa_data = goalkeeper_data.merge(
    match_counts[
        ["Team", "Matches_Played", "Home_Matches", "Away_Matches"]
    ],
    on="Team",
    how="left",
    validate="one_to_one"
)

#  check unmatched_teams
unmatched_teams = fifa_data.loc[
    fifa_data["Matches_Played"].isna(),
    "Team"
]

print("Number of unmatched teams:", len(unmatched_teams))

if not unmatched_teams.empty:
    print(unmatched_teams.tolist())


# Calculate goalkeeper saves per match
fifa_data["Saves_per_match"] = (
    fifa_data["Goalkeeper Saves"] /
    fifa_data["Matches_Played"]
)


# Classify European and non-European teams
european_teams = [
    "Austria",
    "Belgium",
    "Bosnia and Herzegovina",
    "Croatia",
    "Czechia",
    "England",
    "France",
    "Germany",
    "Netherlands",
    "Norway",
    "Portugal",
    "Scotland",
    "Spain",
    "Sweden",
    "Switzerland",
    "Türkiye"
]

fifa_data["Region"] = np.where(
    fifa_data["Team"].isin(european_teams),
    "European",
    "Non-European"
)


# Remove observations with missing analytic values
population_data = fifa_data.dropna(
    subset=["Saves_per_match", "Region"]
).copy()


# Proportionate stratified random sampling
european_sample = population_data[
    population_data["Region"] == "European"
].sample(
    n=10,
    random_state=42
)

non_european_sample = population_data[
    population_data["Region"] == "Non-European"
].sample(
    n=20,
    random_state=42
)


# Combine the two groups
sample_data = pd.concat(
    [european_sample, non_european_sample],
    ignore_index=True
)


# Randomise the order of the final sample
sample_data = sample_data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# Select the analytic variable
sample = np.array(
    sample_data["Saves_per_match"]
)


# Sample size
n = len(sample)

print("Sample size:", n)
print("\nSample size by region:")
print(sample_data["Region"].value_counts())


# Mean
x_bar = stats.mean(sample)
print("Mean saves per match: %.2f" % x_bar)


# Median
median = stats.median(sample)
print("Median saves per match: %.2f" % median)


# Mode
mode = stats.mode(sample)
print("Mode saves per match: %.2f" % mode)


# Minimum and maximum
minimum = np.min(sample)
maximum = np.max(sample)

print("Minimum saves per match: %.2f" % minimum)
print("Maximum saves per match: %.2f" % maximum)


# Range
the_range = maximum - minimum
print("Range: %.2f" % the_range)


# Sample variance
s_square = sample.var(ddof=1)
print("Sample variance: %.2f" % s_square)


# Sample standard deviation
s = sample.std(ddof=1)
print("Sample standard deviation: %.2f" % s)

# Compute standard error
std_err = s / math.sqrt(n)

print(
    "Standard error: %.2f"
    % std_err
)

# 95% confidence level
confidence_level = 0.95
alpha = 1 - confidence_level

# Use a t-critical value because population SD is unknown
t_score = st.t.ppf(
    1 - alpha / 2,
    df=n - 1
)

print("T-statistic: %.2f" % t_score)

# Margin of error
mrg_err = t_score * std_err
print("Margin of error: %.2f" % mrg_err)

# Confidence interval — method 1
ci_low = x_bar - mrg_err
ci_upp = x_bar + mrg_err

print(
    "95%% confidence interval: %.2f to %.2f"
    % (ci_low, ci_upp)
)

# Confidence interval — method 2
ci_low_check, ci_upp_check = st.t.interval(
    confidence=confidence_level,
    df=n - 1,
    loc=x_bar,
    scale=std_err
)

print(
    "95%% confidence interval check: %.2f to %.2f"
    % (ci_low_check, ci_upp_check)
)
# Select the two samples
sample1 = np.array(
    sample_data.loc[
        sample_data["Region"] == "European",
        "Saves_per_match"
    ]
)

sample2 = np.array(
    sample_data.loc[
        sample_data["Region"] == "Non-European",
        "Saves_per_match"
    ]
)


# Basic statistics of sample 1: European teams
x_bar1 = stats.mean(sample1)
s1 = sample1.std(ddof=1)
n1 = len(sample1)


# Basic statistics of sample 2: Non-European teams
x_bar2 = stats.mean(sample2)
s2 = sample2.std(ddof=1)
n2 = len(sample2)


# Display the sample statistics
print("\nSample 1: European teams")
print(
    "\tMean: %.2f. Standard deviation: %.2f. Size: %d."
    % (x_bar1, s1, n1)
)

print("\nSample 2: Non-European teams")
print(
    "\tMean: %.2f. Standard deviation: %.2f. Size: %d."
    % (x_bar2, s2, n2)
)


# Perform Welch's two-sample t-test
# Null hypothesis:
# Mean of European teams = mean of Non-European teams
#
# Alternative hypothesis:
# Mean of European teams != mean of Non-European teams

t_stats, p_val = st.ttest_ind_from_stats(
    mean1=x_bar1,
    std1=s1,
    nobs1=n1,
    mean2=x_bar2,
    std2=s2,
    nobs2=n2,
    equal_var=False,
    alternative="two-sided"
)


# Display the t-statistic
print("\nComputing t* ...")
print(
    "\tt-statistic (t*): %.2f"
    % t_stats
)


# Display the p-value
print("\nComputing p-value ...")
print(
    "\tp-value: %.4f"
    % p_val
)


# Statistical conclusion
alpha = 0.05

print("\nConclusion:")

if p_val < alpha:
    print(
        "\tWe reject the null hypothesis."
    )

    print(
        "\tThere is statistically significant evidence "
        "of a difference in mean goalkeeper saves per "
        "match between European and Non-European teams."
    )

else:
    print(
        "\tWe fail to reject the null hypothesis."
    )

    print(
        "\tThere is insufficient statistical evidence "
        "of a difference in mean goalkeeper saves per "
        "match between European and Non-European teams."
    )