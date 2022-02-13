"""
Humanities Data Analysis: Case studies with Python
--------------------------------------------------
Folgert Karsdorp, Mike Kestemont & Allen Riddell
Chapter 5: Statistics Essentials: Who Reads Novels?
"""


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
import random, numpy; random.seed(1); numpy.random.seed(1);


# %%
# Dataset GSS7214_R5.DTA is stored in compressed form as GSS7214_R5.DTA.gz
import gzip
import pandas as pd

with gzip.open('data/GSS7214_R5.DTA.gz', 'rb') as infile:
    # we restrict this (very large) dataset to the variables of interest
    columns = ['id', 'year', 'age', 'sex', 'race', 'reg16', 'degree',
               'realrinc', 'readfict']
    df = pd.read_stata(infile, columns=columns)

# further limit dataset to the years we are interested in
df = df.loc[df['year'].isin({1998, 2000, 2002})]


# %%
# limit dataset to exclude records from individuals who refused
# to report their income
df = df.loc[df['realrinc'].notnull()]


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
# REALITY CHECK, using GSS provided values
df.loc[df['year'] == 2000, 'realrinc'].astype(float).min() == 333;
round(df.loc[df['year'] == 2000, 'realrinc'].astype(float).mean(), 2) == 22110.13;


# %%
# inflation measured via US Consumer Price Index (CPI), source:
# http://www.dlt.ri.gov/lmi/pdf/cpi.pdf
cpi2015_vs_1986 = 236.7 / 109.6
assert df['realrinc'].astype(float).median() < 24000  # reality check
df['realrinc2015'] = cpi2015_vs_1986 * df['realrinc'].astype(float)


# %%
import matplotlib.pyplot as plt
df.groupby('race')['realrinc2015'].plot(kind='hist', bins=30)
plt.xlabel('Income')
plt.legend();


# %%
import numpy as np

df['realrinc2015_log10'] = np.log10(df['realrinc2015'])
df.groupby('race')['realrinc2015_log10'].plot(kind='hist', bins=30)
plt.xlabel(r'$\log10(\mathrm{Income})$')
plt.legend();


# %%
print(df['realrinc2015'].max() / df['realrinc2015'].min())


# %%
print(df['realrinc2015'].mean())


# %%
print(df['realrinc2015'].median())


# %%
df_bachelor = df[df['degree'] == 'bachelor']
# observed=True instructs pandas to ignore categories
# without any observations
df_bachelor.groupby(['year', 'degree'], observed=True)['realrinc2015'].agg(['size', 'mean', 'median'])


# %%
realrinc2015_corrupted = [11159, 13392, 31620, 40919, 53856, 60809, 118484, 1436180]
print(np.mean(realrinc2015_corrupted))


# %%
print(np.median(realrinc2015_corrupted))


# %%
readfict_sample = df.loc[df['readfict'].notnull()].sample(8)['readfict']
readfict_sample = readfict_sample.replace(['no', 'yes'], [0, 1])
readfict_sample


# %%
print("Mean:", readfict_sample.mean())
print("Median:", readfict_sample.median())


# %%
import matplotlib.pyplot as plt

df['realrinc2015'].plot(kind='hist', bins=30);


# %%
# simulate incomes from a gamma distribution with identical mean
alpha = 5
sim = np.random.gamma(alpha, df['realrinc2015'].mean() / alpha, size=df.shape[0])
sim = pd.Series(sim, name='realrinc2015_simulated')
sim.plot(kind='hist', bins=30);


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
# REALITY CHECK
import math
math.isclose(df['realrinc2015'].mean(), np.mean(sim), rel_tol=0.01);


# %%
# Name this function `range_` to avoid colliding with the built-in
# function `range`.
def range_(series):
    """Difference between the maximum value and minimum value."""
    return series.max() - series.min()


print(f"Observed range: {range_(df['realrinc2015']):,.0f}\n"
      f"Simulated range: {range_(sim):,.0f}")


# %%
print(f"Observed variance: {df['realrinc2015'].var():.2f}\n"
      f"Simulated variance: {sim.var():.2f}")


# %%
print(f"Observed std: {df['realrinc2015'].std():.2f}\n"
      f"Simulated std: {sim.std():.2f}")


# %%
# The many standard deviation functions in Python:
import statistics

print(f"statistics.stdev: {statistics.stdev(sim):.1f}\n"
      f"         sim.std: {sim.std():.1f}\n"
      f"          np.std: {np.std(sim):.1f}\n"
      f"  np.std(ddof=1): {np.std(sim, ddof=1):.1f}")


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
fig, axes = plt.subplots(2, sharex=True)
df['realrinc2015'].plot(kind='box', vert=False, ax=axes[0])
axes[0].set_yticklabels(["Respondent's income"])
sim.plot(kind='box', vert=False, ax=axes[1])
axes[1].set_yticklabels(["Respondent's income (simulated)"]);


# %%
df.groupby('degree')['realrinc2015'].mad().round()


# %%
df_bachelor_or_more = df[df['degree'].isin(['bachelor', 'graduate'])]
df_bachelor_or_more.groupby(['degree', 'readfict'], observed=True)['realrinc2015'].mad().round()


# %%
df_bachelor_or_more.groupby(['degree', 'readfict'], observed=True)['realrinc2015'].mean().round()


# %%
group1 = ['high school', 'high school', 'high school', 'high school', 'high school',
          'high school', 'bachelor', 'bachelor']
group2 = ['lt high school', 'lt high school', 'lt high school', 'lt high school',
          'high school', 'junior college', 'bachelor', 'graduate']
group3 = ['lt high school', 'lt high school', 'high school', 'high school',
          'junior college', 'junior college', 'bachelor', 'graduate']


# %%
# calculate the number of unique values in each group
print([len(set(group)) for group in [group1, group2, group3]])
# calculate the ratio of observed categories to total observations
print([len(set(group)) / len(group) for group in [group1, group2, group3]])


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
# East South Central States are Alabama, Kentucky, Mississippi, Tennessee
regions_oi = sorted(['pacific', 'e. sou. central', 'new england'])
df_regions = df.loc[df['reg16'].isin(regions_oi)].copy()
df_regions['reg16'] = df_regions['reg16'].cat.remove_unused_categories()
df_regions.groupby('reg16')['degree'].value_counts(normalize=True).round(1).to_frame();


# %%
import collections
import scipy.stats

# Calculate the entropy of the empirical distribution over degree
# types for each group
for n, group in enumerate([group1, group2, group3], 1):
    degree_counts = list(collections.Counter(group).values())
    H = scipy.stats.entropy(degree_counts)
    print(f'Group {n} entropy: {H:.1f}')


# %%
df.groupby('reg16')['degree'].apply(lambda x: scipy.stats.entropy(x.value_counts()))


# %%
df_subset_columns = ['age', 'realrinc2015_log10', 'reg16', 'degree']
min_income = 10_000
df_subset_index_mask = ((df['age'] >= 23) & (df['age'] <= 30) &
                        (df['degree'] == 'bachelor') &
                        (df['realrinc2015'] > min_income))
df_subset = df.loc[df_subset_index_mask, df_subset_columns]
# discard rows with NaN values
df_subset = df_subset[df_subset.notnull().all(axis=1)]
# age is an integer, not a float
df_subset['age'] = df_subset['age'].astype(int)


# %%
# Small amount of noise ("jitter") to respondents' ages makes
# discrete points easier to see
_jitter = np.random.normal(scale=0.1, size=len(df_subset))
df_subset['age_jitter'] = df_subset['age'].astype(float) + _jitter
ax = df_subset.plot(x='age_jitter', y='realrinc2015_log10', kind='scatter', alpha=0.4)
ax.set(ylabel="Respondent's income (log10)", xlabel="Age");


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
# calculate values used in paragraph below
x, y = np.arange(23, 31), df_subset.groupby('age')['realrinc2015_log10'].median()
slope, intercept = np.polyfit(x, y, deg=1)
[slope.round(3), intercept.round(2), (10**slope - 1).round(2), y[[23, 30]].round(1), (10**y[[23, 30]]).round(-3)];


# %%
ax = df_subset.plot(x='age_jitter', y='realrinc2015_log10', kind='scatter', alpha=0.4)
slope, intercept = 0.035, 3.67
xs = np.linspace(23 - 0.2, 30 + 0.2)
label = f'y = {slope:.3f}x + {intercept:.2f}'
ax.plot(xs, slope * xs + intercept, label=label)
ax.set(ylabel="Respondent's income (log10)", xlabel="Age")
ax.legend();


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
ax = df_subset.plot(x='age_jitter', y='realrinc2015_log10', kind='scatter', alpha=0.4)
coef2, coef1, intercept = np.polyfit(x, y, deg=2)
label = f'$y = {coef2:.2f}x^2 + {coef1:.2f}x + {intercept:.2f}$'
ax.plot(xs, coef1 * xs + coef2 * xs**2 + intercept, label=label)
ax.set(ylabel="Respondent's income (log10)", xlabel="Age")
ax.legend();


# %%
df_subset[['age', 'realrinc2015_log10']].corr('kendall')


# %%
df_subset = df.loc[df['readfict'].notnull(), ['reg16', 'readfict']]
pd.crosstab(df_subset['reg16'], df_subset['readfict'], margins=True)


# %%
pd.crosstab(df_subset['reg16'], df_subset['readfict']).plot.barh(stacked=True);
# The pandas.crosstab call above accomplishes the same thing as the call:
# df_subset.groupby('reg16')['readfict'].value_counts().unstack()


# %%
pd.crosstab(df_subset['reg16'], df_subset['readfict'], normalize='index').sort_values(
    by='yes', ascending=False)


# %%
pd.crosstab(
    df_subset['reg16'], df_subset['readfict'], normalize='index').plot.barh(stacked=True);
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, title="Read fiction?");


# %%
# Strategy:
# 1. Calculate the table of Pr(X=x, Y=y) from empirical frequencies
# 2. Calculate the marginal distributions Pr(X=x)Pr(Y=y)
# 3. Combine above quantities to calculate the mutual information.

joint = pd.crosstab(df_subset['reg16'], df_subset['readfict'], normalize='all')

# construct a table of the same shape as joint with the relevant
# values of Pr(X = x)Pr(Y = y)
proba_readfict, proba_reg16 = joint.sum(axis=0), joint.sum(axis=1)
denominator = np.outer(proba_reg16, proba_readfict)

mutual_information = (joint * np.log(joint / denominator)).sum().sum()
print(mutual_information)


# %%
tate = pd.read_csv("data/tate.csv.gz")
# remove objects for which no suitable year information is given:
tate = tate[tate['year'].notnull()]
tate = tate[tate['year'].str.isdigit()]
tate['year'] = tate['year'].astype('int')


