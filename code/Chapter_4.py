"""
Humanities Data Analysis: Case studies with Python
--------------------------------------------------
Folgert Karsdorp, Mike Kestemont & Allen Riddell
Chapter 4: Processing Tabular Data
"""


# %%
import csv

with open('data/names.csv') as infile:
    data = list(csv.DictReader(infile))


# %%
print(data[0])


# %%
data = []

with open('data/names.csv') as infile:
    for row in csv.DictReader(infile):
        row['frequency'] = int(row['frequency'])
        row['year'] = int(row['year'])
        data.append(row)

print(data[0])


# %%
starting_year = min(row['year'] for row in data)
print(starting_year)


# %%
ending_year = max(row['year'] for row in data)
print(ending_year)


# %%
print(ending_year - starting_year + 1)


# %%
import collections

# Step 1: create counter objects to store counts of girl and boy names
# See Chapter 2 for an introduction of Counter objects
name_counts_girls = collections.Counter()
name_counts_boys = collections.Counter()

# Step 2: iterate over the data and increment the counters
for row in data:
    if row['sex'] == 'F':
        name_counts_girls[row['year']] += 1
    else:
        name_counts_boys[row['year']] += 1

# Step 3: Loop over all years and assert the presence of at least
# 10 girl and boy names
for year in range(starting_year, ending_year + 1):
    assert name_counts_girls[year] >= 10
    assert name_counts_boys[year] >= 10


# %%
import pandas as pd


# %%
df = pd.read_csv('data/names.csv')


# %%
df.head(n=5)


# %%
print(df.dtypes)


# %%
df['name'].head()


# %%
print(type(df['sex'].values))


# %%
df[['name', 'sex']].head()


# %%
df.iloc[3:7]


# %%
df.iloc[30]


# %%
print(df.columns)


# %%
print(df.index)


# %%
df = pd.read_csv('data/names.csv', index_col=0)
df.head()


# %%
#  first reload the data without index specification
df = pd.read_csv('data/names.csv')
df = df.set_index('year')
df.head()


# %%
df.loc[1899].head()


# %%
df.loc[1921, 'name'].head()


# %%
print(df.iloc[10, 2])


# %%
df.loc[1921, ['name', 'sex']].head()


# %%
df.loc[[1921, 1967], ['name', 'sex']].head()


# %%
df.loc[[1921, 1967], ['name', 'sex']].tail()


# %%
df.iloc[[3, 10], [0, 2]]


# %%
df.iloc[1000:1100, -2:].head()


# %%
array = df.values
array_slice = array[1000:1100, -2:]


# %%
def df2ranking(df, rank_col='frequency', cutoff=20):
    """Transform a data frame into a popularity index."""
    df = df.sort_values(by=rank_col, ascending=False)
    df = df.reset_index()
    return df['name'][:cutoff]


# %%
girl_ranks, boy_ranks = [], []
for year in df.index.unique():
    for sex in ('F', 'M'):
        if sex == 'F':
            year_df = df.loc[year]
            ranking = df2ranking(year_df.loc[year_df['sex'] == sex])
            ranking.name = year
            girl_ranks.append(ranking)
        else:
            year_df = df.loc[year]
            ranking = df2ranking(year_df.loc[year_df['sex'] == sex])
            ranking.name = year
            boy_ranks.append(ranking)

girl_ranks = pd.DataFrame(girl_ranks)
boy_ranks = pd.DataFrame(boy_ranks)


# %%
# we use reset_index() to make `year` available as a column
df.reset_index().groupby('name')['year'].median().head()


# %%
df.reset_index().groupby('name')['year'].median().sort_values().head()


# %%
boy_ranks = df.loc[df.sex == 'M'].groupby(level=0).apply(df2ranking)
girl_ranks = df.loc[df.sex == 'F'].groupby(level=0).apply(df2ranking)


# %%
import numpy as np

data = pd.DataFrame({'name': ['Jennifer', 'Claire', 'Matthew', 'Richard',
                              'Richard', 'Claire', 'Matthew', 'Jennifer'],
                     'sex': ['F', 'F', 'M', 'M',
                             'M', 'F', 'M', 'F'],
                     'value': np.random.rand(8)})
data


# %%
grouped = data.groupby('sex')


# %%
grouped.get_group('F')


# %%
for grouper, group in grouped:
    print('grouper:', grouper)
    print(group)


# %%
grouped.sum()


# %%
grouped['value'].mean()


# %%
grouped['value'].agg(np.sum)


# %%
grouped['value'].agg(['size', 'mean'])


# %%
def combine_unique(names):
    return ' '.join(set(names))

grouped['name'].agg(combine_unique)


# %%
A = {'Frank', 'Henry', 'James', 'Richard'}
B = {'Ryan', 'James', 'Logan', 'Frank'}

diff_1 = A.difference(B)
diff_2 = A - B

print(f"Difference of A and B = {diff_1}")
print(f"Difference of A and B = {diff_2}")


# %%
print(len(A.difference(B)))


# %%
boy_ranks.apply(set, axis=1).head()


# %%
def turnover(df):
    """Compute the 'turnover' for popularity rankings."""
    df = df.apply(set, axis=1)
    turnovers = {}
    for year in range(df.index.min() + 1, df.index.max() + 1):
        name_set, prior_name_set = df.loc[year], df.loc[year - 1]
        turnovers[year] = len(name_set.difference(prior_name_set))
    return pd.Series(turnovers)


# %%
boy_turnover = turnover(boy_ranks)
boy_turnover.head()


# %%
def turnover(df):
    """Compute the 'turnover' for popularity rankings."""
    df = df.apply(set, axis=1)
    return (df.iloc[1:] - df.shift(1).iloc[1:]).apply(len)


# %%
A = np.array([{'Isaac', 'John', 'Mark'}, {'Beth', 'Rose', 'Claire'}])
B = np.array([{'John', 'Mark', 'Benjamin'}, {'Sarah', 'Anna', 'Susan'}])

C = A - B
print(C)


# %%
s = boy_ranks.apply(set, axis=1)
s.shift(1).head()


# %%
differences = (s.iloc[1:] - s.shift(1).iloc[1:])
differences.head()


# %%
turnovers = differences.apply(len)
turnovers.head()


# %%
boy_turnover = turnover(boy_ranks)
boy_turnover.head()


# %%
girl_turnover = turnover(girl_ranks)
girl_turnover.head()


# %%
ax = girl_turnover.plot(
    style='o', ylim=(-0.1, 3.1), alpha=0.7,
    title='Annual absolute turnover (girls)'
)
ax.set_ylabel("Absolute turnover");


# %%
girl_turnover.plot(kind='hist');


# %%
girl_rm = girl_turnover.rolling(25).mean()
ax = girl_rm.plot(title="Moving average turnover (girls; window = 25)")
ax.set_ylabel("Absolute turnover");


# %%
boy_rm = boy_turnover.rolling(25).mean()
ax = boy_rm.plot(title="Moving average turnover (boys; window = 25)")
ax.set_ylabel("Absolute turnover");


# %%
def type_token_ratio(frequencies):
    """Compute the type-token ratio of the frequencies."""
    return len(frequencies) / frequencies.sum()


# %%
ax = df.loc[df['sex'] == 'F'].groupby(level=0)['frequency'].apply(type_token_ratio).plot();
ax.set_ylabel("type-token ratio");


# %%
import matplotlib.pyplot as plt

# create an empty plot
fig, ax = plt.subplots()

for sex in ['F', 'M']:
    counts = df.loc[df['sex'] == sex, 'frequency']
    tt_ratios = counts.groupby(level=0).apply(type_token_ratio)
    # Use the same axis to plot both sexes (i.e. ax=ax)
    tt_ratios.plot(label=sex, legend=True, ax=ax)
ax.set_ylabel("type-token ratio");


# %%
def max_relative_frequency(frequencies):
    return (frequencies / frequencies.sum()).max()

# create an empty plot
fig, ax = plt.subplots()

for sex in ['F', 'M']:
    counts = df.loc[df['sex'] == sex, 'frequency']
    div = counts.groupby(level=0).apply(max_relative_frequency)
    div.plot(label=sex, legend=True, ax=ax)
ax.set_ylabel("Relative frequency");


# %%
boys_names = df.loc[df['sex'] == 'M', 'name']
boys_names.head()


# %%
boys_names.str.lower().head()


# %%
boys_names.loc[boys_names.str.match('[^aeiou]+o[^aeiou]', case=False)].head()


# %%
boys_names.str.get(0).head()


# %%
boys_coda = boys_names.str.get(-1)
boys_coda.head()


# %%
boys_fd = boys_coda.groupby('year').value_counts(normalize=True)
boys_fd.head()


# %%
boys_fd.loc[1940].sort_index().head()


# %%
boys_fd.loc[[(1960, 'n'), (1960, 'p'), (1960, 'r')]]


# %%
boys_fd = boys_fd.unstack()
boys_fd.head()


# %%
boys_fd = boys_fd.fillna(0)


# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(
    nrows=2, ncols=4, sharey=True, figsize=(12, 6))

letters = ["a", "d", "e", "i", "n", "o", "s", "t"]
axes = boys_fd[letters].plot(
    subplots=True, ax=axes, title=letters, color='C0', grid=True, legend=False)

# The x-axis of each subplots is labeled with 'year'.
# We remove those and add one main x-axis label
for ax in axes.flatten():
    ax.xaxis.label.set_visible(False)
fig.text(0.5, 0.04, "year", ha="center", va="center", fontsize="x-large")
# Reserve some additional height for space between subplots
fig.subplots_adjust(hspace=0.5);


# %%
d = df.loc[1910]
duplicates = d[d.duplicated(subset='name')]['name']
duplicates.head()


# %%
d = d.loc[d.duplicated(subset='name', keep=False)]
d.sort_values('name').head()


# %%
d = d.pivot_table(values='frequency', index='name', columns='sex')
d.head()


# %%
(d['F'] / (d['F'] + d['M'])).head()


# %%
def usage_ratio(df):
    """Compute the usage ratio for unixes names."""
    df = df.loc[df.duplicated(subset='name', keep=False)]
    df = df.pivot_table(values='frequency', index='name', columns='sex')
    return df['F'] / (df['F'] + df['M'])


# %%
d = df.groupby(level=0).apply(usage_ratio)
d.head()


# %%
d = d.unstack(level='name')
d.tail()


# %%
unisex_ranking = abs(d - 0.5).fillna(0.5).mean().sort_values().index


# %%
# Create a figure and subplots
fig, axes = plt.subplots(
    nrows=2, ncols=4, sharey=True, sharex=True, figsize=(12, 6))

# Plot the time series into the subplots
names = unisex_ranking[:8].tolist()
d[names].plot(
    subplots=True, color="C0", ax=axes, legend=False, title=names)

# Clean up some redudant labels and adjust spacing
for ax in axes.flatten():
    ax.xaxis.label.set_visible(False)
    ax.axhline(0.5, ls='--', color="grey", lw=1)
fig.text(0.5, 0.04, "year", ha="center", va="center", fontsize="x-large");
fig.subplots_adjust(hspace=0.5);


# %%
# Create a figure and subplots
fig, axes = plt.subplots(
    nrows=2, ncols=4, sharey=True, sharex=True, figsize=(12, 6))

# Plot the time series into the subplots
d[names].rolling(window=10).mean().plot(
    color='C0', subplots=True, ax=axes, legend=False, title=names);

# Clean up some redundant labels and adjust spacing
for ax in axes.flatten():
    ax.xaxis.label.set_visible(False)
    ax.axhline(0.5, ls='--', color="grey", lw=1)
fig.text(0.5, 0.04, "year", ha="center", va="center", fontsize="x-large");
fig.subplots_adjust(hspace=0.5);


