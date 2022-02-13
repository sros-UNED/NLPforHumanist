"""
Humanities Data Analysis: Case studies with Python
--------------------------------------------------
Folgert Karsdorp, Mike Kestemont & Allen Riddell
Chapter 1: Introduction
"""


# %%
x = 100


# %%
saying = "It's turtles all the way down"


# %%
print(saying[0])


# %%
string = "Python"
for character in string:
    print(character)


# %%
numbers = [1, 1, 2, 3, 5, 8]
words = ["This", "is", "a", "list", "of", "strings"]


# %%
print(numbers[0])
print(numbers[-1])  # use -1 to retrieve the last item in a sequence
print(words[3:])  # use slice syntax to retrieve a subsequence


# %%
packages = {'matplotlib': 'Matplotlib is a Python 2D plotting library',
            'pandas': 'Pandas is a Python library for data analysis',
            'scikit-learn': 'Scikit-learn helps with Machine Learning in Python'}


# %%
print(packages['pandas'])


# %%
packages = {"matplotlib", "pandas", "scikit-learn"}


# %%
words = ["move", "slowly", "and", "fix", "things"]
for word in words:
    if "i" in word:
        print(word)


# %%
import math


# %%
print(math.log(2.7183))
print(math.sqrt(2))


# %%
def count_ing(strings):
    count = 0
    for string in strings:
        if string.endswith("ing"):
            count += 1
    return count

words = [
    "coding", "is", "about", "developing", "logical", "event", "sequences"
]
print(count_ing(words))


# %%
f = open("data/aesop-wolf-dog.txt")  # open a file
text = f.read()  # read the contents of a file
f.close()  # close the connection to the file
print(text)  # print the contents of the file


# %%
import pandas as pd

df = pd.read_csv("data/feeding-america.csv", index_col='date')


# %%
df.head()


# %%
print(len(df))


# %%
print(df['recipe_class'].unique())


# %%
df['recipe_class'].value_counts()


# %%
df['recipe_class'].value_counts().plot(kind='bar', color="C0", width=0.1);


# %%
import matplotlib.pyplot as plt

grouped = df.groupby('date')  # group all rows from the same year
recipe_counts = grouped.size()  # compute the size of each group
recipe_counts.plot(style='o', xlim=(1810, 1930))  # plot the group size
plt.ylabel("number of recipes")  # add a label to the y-axis
plt.xlabel("year of publication") # add a label to the x-axis


# %%
# split ingredient strings into lists
ingredients = df['ingredients'].str.split(';')
# group all rows from the same year
groups = ingredients.groupby('date')
# merge the lists from the same year
ingredients = groups.sum()
# compute counts per year
ingredients = ingredients.apply(pd.Series.value_counts)
# normalise the counts
ingredients = ingredients.divide(recipe_counts, 0)


# %%
ingredients.head()


# %%
ax = ingredients['tomato'].plot(style='o', xlim=(1810, 1930))
ax.set_ylabel("fraction of recipes")
ax.set_xlabel("year of publication");


# %%
import scipy.stats

def plot_trend(column, df, line_color='grey', xlim=(1810, 1930)):
    slope, intercept, _, _, _ = scipy.stats.linregress(
        df.index, df[column].fillna(0).values)
    ax = df[column].plot(style='o', label=column)
    ax.plot(df.index, intercept + slope * df.index, '--',
             color=line_color, label='_nolegend_')
    ax.set_ylabel("fraction of recipes")
    ax.set_xlabel("year of publication")
    ax.set_xlim(xlim)


# %%
plot_trend('tomato', ingredients)


# %%
plot_trend('baking powder', ingredients)
plot_trend('yeast', ingredients)
plt.legend();  # add a legend to the plot


# %%
plot_trend('nutmeg', ingredients)


# %%
from sklearn.feature_selection import chi2

# Transform the index into a list of labels, in which each label
# indicates whether a row stems from before or after the Civil War:
labels = ['Pre-Civil War' if year < 1864 else 'Post-Civil War' for year in ingredients.index]
# replace missing values with zero (.fillna(0)),
# and compute the chi2 statistic:
keyness, _ = chi2(ingredients.fillna(0), labels)
# Turn keyness values into a Series, and sort in descending order:
keyness = pd.Series(keyness, index=ingredients.columns).sort_values(ascending=False)


# %%
keyness.head(n=10)


# %%
# step 1: compute summed ingredient counts per year
counts = df['ingredients'].str.split(';').groupby(
    'date').sum().apply(pd.Series.value_counts).fillna(0)

# step 2: construct frequency rankings for pre- and post-war years
pre_cw = counts[counts.index < 1864].sum().rank(method='dense', pct=True)
post_cw = counts[counts.index > 1864].sum().rank(method='dense', pct=True)

# step 3: merge the pre- and post-war data frames
rankings = pd.DataFrame({'Pre-Civil War': pre_cw, 'Post-Civil War': post_cw})

# step 4: produce the plot
fig = plt.figure(figsize=(10, 6))
plt.scatter(rankings['Post-Civil War'], rankings['Pre-Civil War'],
            c=rankings['Pre-Civil War'] - rankings['Post-Civil War'],
            alpha=0.7)

# Add annotations of the 20 most distinctive ingredients
for i, row in rankings.loc[keyness.head(20).index].iterrows():
    plt.annotate(i, xy=(row['Post-Civil War'], row['Pre-Civil War']))

plt.xlabel("Frequency rank Post-Civil War")
plt.ylabel("Frequency rank Pre-Civil War");


# %%
df['ethnicgroup'].value_counts(dropna=False).head(10)


# %%
grouped = df.groupby(level='date')
# compute the number of unique ethnic groups per year,
# divided by the number of books
n_groups = grouped['ethnicgroup'].nunique() / grouped['book_id'].nunique()
n_groups.plot(style='o')

# add a least square line as reference
slope, intercept, _, _, _ = scipy.stats.linregress(
    n_groups.index, n_groups.fillna(0).values)

# create the plot
plt.plot(
    n_groups.index, intercept + slope * n_groups.index, '--', color="grey")
plt.xlim(1810, 1930)
plt.ylabel("Average number of ethnic groups")
plt.xlabel("Year of publication");


# %%
# step 1: add a new column indicating for each recipe whether
#         we have information about its ethnic group
df['foreign'] = df['ethnicgroup'].notnull()

# step 2: construct frequency rankings for foreign and general recipes
counts = df.groupby('foreign')['ingredients'].apply(
    ';'.join).str.split(';').apply(pd.Series.value_counts).fillna(0)

foreign_counts = counts.iloc[1].rank(method='dense', pct=True)
general_counts = counts.iloc[0].rank(method='dense', pct=True)

# step 3: merge the foreign and general data frames
rankings = pd.DataFrame({'foreign': foreign_counts, 'general': general_counts})

# step 4: compute the keyness of ingredients in foreign recipes
#         as the difference in frequency ranks
keyness = (rankings['foreign'] - rankings['general']).sort_values(ascending=False)

# step 5: produce the plot
fig = plt.figure(figsize=(10, 6))
plt.scatter(rankings['general'], rankings['foreign'],
            c=rankings['foreign'] - rankings['general'],
            alpha=0.7)

for i, row in rankings.loc[keyness.head(10).index].iterrows():
    plt.annotate(i, xy=(row['general'], row['foreign']))

plt.xlabel("Frequency rank general recipes")
plt.ylabel("Frequency rank foreign recipes");


