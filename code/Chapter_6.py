"""
Humanities Data Analysis: Case studies with Python
--------------------------------------------------
Folgert Karsdorp, Mike Kestemont & Allen Riddell
Chapter 6: Introduction to Probability
"""


# %%
pr_pynchon = 0.00001
pr_positive = 0.90
pr_false_positive = 0.01
print(pr_positive * pr_pynchon / (pr_positive * pr_pynchon +
                                  pr_false_positive * (1 - pr_pynchon)))


# %%
import pandas as pd
import numpy as np; np.random.seed(1)  # fix a seed for reproducible random sampling

# only show counts for these words:
words_of_interest = ['upon', 'the', 'state', 'enough', 'while']
df = pd.read_csv('data/federalist-papers.csv', index_col=0)
df[words_of_interest].sample(6)


# %%
# values associated with the column 'AUTHOR' are one of the following:
# {'HAMILTON', 'MADISON', 'JAY', 'HAMILTON OR MADISON',
#  'HAMILTON AND MADISON'}
# essays with the author 'HAMILTON OR MADISON' are the 12 disputed essays.
disputed_essays = df[df['AUTHOR'] == 'HAMILTON OR MADISON'].index
assert len(disputed_essays) == 12  # there are twelve disputed essays
# numbers widely used to identify the essays
assert set(disputed_essays) == {49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 62, 63}


# %%
# gather essays with known authorship: the undisputed essays of
# Madison and Hamilton
df_known = df.loc[df['AUTHOR'].isin(('HAMILTON', 'MADISON'))]
print(df_known['AUTHOR'].value_counts())


# %%
df_known.groupby('AUTHOR')['upon'].plot.hist(
    rwidth=0.9, alpha=0.6, range=(0, 22), legend=True);


# %%
df_known.groupby('AUTHOR')['upon'].describe()


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
np.testing.assert_array_almost_equal(
    df_known.groupby('AUTHOR')['upon'].mean().values, [7.33333333, 1.25])


# %%
# The expression below applies `mean` to a sequence of binary observations
# to get a proportion. For example,
# np.mean([False, False, True]) == np.mean([0, 0, 1]) == 1/3
proportions = df_known.groupby('AUTHOR')['upon'].apply(
    lambda upon_counts: (upon_counts > 0).mean())
print(proportions)


# %%
proportions.plot.bar(rot=0);


# %%
proportions


# %%
df = pd.read_csv('data/federalist-papers.csv', index_col=0)
author = df['AUTHOR']  # save a copy of the author column
df = df.drop('AUTHOR', axis=1)  # remove the author column
df = df.divide(df.sum(axis=0))  # rate per 1 word
df *= 1000  # transform from rate per 1 word to rate per 1,000 words
df = df.round()  # round to nearest integer
df['AUTHOR'] = author  # put author column back
df_known = df[df['AUTHOR'].isin({'HAMILTON', 'MADISON'})]

df_known.groupby('AUTHOR')['by'].describe()


# %%
df_known.groupby('AUTHOR')['by'].plot.hist(
    alpha=0.6, range=(0, 35), rwidth=0.9, legend=True);


# %%
print(df_known.loc[df_known['by'] > 30, 'by'])


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
assert df_known.loc[df_known['by'] > 30, 'by'].index == 83


# %%
with open('data/federalist-83.txt') as infile:
    text = infile.read()

# a regular expression here would be more robust
by_jury_count = text.count(' by jury')
by_count = text.count(' by ')
word_count = len(text.split())  # crude word count
by_rate = 1000 * (by_count - by_jury_count) / word_count

print('In Federalist No. 83 (by Hamilton), without "by jury", '
      f'"by" occurs {by_rate:.0f} times per 1,000 words on average.')


# %%
import scipy.special

def negbinom_pmf(x, alpha, beta):
    """Negative binomial probability mass function."""
    # In practice this calculation should be performed on the log
    # scale to reduce the risk of numeric underflow.
    return (
        scipy.special.binom(x + alpha - 1, alpha - 1)
        * (beta / (beta + 1)) ** alpha
        * (1 / (beta + 1)) ** x
    )

print('Pr(X = 6):', negbinom_pmf(6, alpha=5, beta=1))
print('Pr(X = 14):', negbinom_pmf(14, alpha=5, beta=1))


# %%
df_known[df_known['AUTHOR'] == 'HAMILTON']['by'].plot.hist(
    range=(0, 35), density=True, rwidth=0.9);


# %%
df_known[df_known['AUTHOR'] == 'HAMILTON']['by'].describe()


# %%
import itertools
import matplotlib.pyplot as plt

x = np.arange(60)
alphas, betas = [5, 6.5, 12, 16], [1.5, 0.5, 0.7]
params = list(itertools.product(alphas, betas))
pmfs = [negbinom_pmf(x, alpha, beta) for alpha, beta in params]

fig, axes = plt.subplots(4, 3, sharey=True, figsize=(10, 8))
axes = axes.flatten()

for ax, pmf, (alpha, beta) in zip(axes, pmfs, params):
    ax.bar(x, pmf)
    ax.set_title(fr'$\alpha$ = {alpha}, $\beta$ = {beta}')
plt.tight_layout();


# %%
def negbinom(alpha, beta, size=None):
    """Sample from a negative binomial distribution.

    Uses `np.random.negative_binomial`, which makes use of a
    different parameterization than the one used in the text.
    """
    n = alpha
    p = beta / (beta + 1)
    return np.random.negative_binomial(n, p, size)

samples = negbinom(5, 0.7, 10000)
# put samples in a pandas Series in order to calculate summary statistics
pd.Series(samples).describe()


# %%
df_known[df_known['AUTHOR'] == 'MADISON']['by'].plot.hist(
    density=True, rwidth=0.9, range=(0, 35)  # same scale as with Hamilton
);


# %%
x = np.arange(60)
alphas, betas = [100, 50, 28, 10], [23, 4, 1.3]
params = list(itertools.product(alphas, betas))
pmfs = [negbinom_pmf(x, alpha, beta) for alpha, beta in params]

fig, axes = plt.subplots(4, 3, sharey=True, figsize=(10, 8))
axes = axes.flatten()

for ax, pmf, (alpha, beta) in zip(axes, pmfs, params):
    ax.bar(x, pmf)
    ax.set_title(fr'$\alpha$ = {alpha}, $\beta$ = {beta}')
plt.tight_layout()


# %%
authors = ('HAMILTON', 'MADISON')
alpha_hamilton, beta_hamilton = 5, 0.7
alpha_madison, beta_madison = 50, 4

# observed
fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
df_known.groupby('AUTHOR')['by'].plot.hist(
    ax=axes[0], density=True, range=(0, 35), rwidth=0.9, alpha=0.6,
    title='Hamilton v. Madison (observed)', legend=True)

# model
simulations = 10000
for author, (alpha, beta) in zip(authors, [(alpha_hamilton, beta_hamilton),
                                           (alpha_madison, beta_madison)]):
    pd.Series(negbinom(alpha, beta, size=simulations)).plot.hist(
        label=author, density=True, rwidth=0.9, alpha=0.6, range=(0, 35), ax=axes[1])
axes[1].set_xlim((0, 40))
axes[1].set_title('Hamilton v. Madison (model)')
axes[1].legend()
plt.tight_layout();


# %%
likelihood_hamilton = negbinom_pmf(14, alpha_hamilton, beta_hamilton)
print(likelihood_hamilton)


# %%
likelihood_madison = negbinom_pmf(14, alpha_madison, beta_madison)
print(likelihood_madison)


# %%
pr_hamilton = likelihood_hamilton * 0.5 / (likelihood_hamilton * 0.5 + likelihood_madison * 0.5)
print(pr_hamilton)


# %%
# `x` is a sample of Hamilton's rates of 'by' (per 1,000 words)
x = np.array([13, 6, 4, 8, 16, 9, 10, 7, 18, 10, 7, 5, 8, 5, 6, 14, 47])
pd.Series(x).describe()


# %%
import scipy.optimize
import scipy.special

# The function `negbinom_pmf` is defined in the text.


def estimate_negbinom(x):
    """Estimate the parameters of a negative binomial distribution.

    Maximum-likelihood estimates of the parameters are calculated.
    """
    def objective(x, alpha):
        beta = alpha / np.mean(x)  # MLE for beta has closed-form solution
        return -1 * np.sum(np.log(negbinom_pmf(x, alpha, beta)))
    alpha = scipy.optimize.minimize(
        lambda alpha: objective(x, alpha), x0=np.mean(x), bounds=[(0, None)], tol=1e-7).x[0]
    return alpha, alpha / np.mean(x)


alpha, beta = estimate_negbinom(x)
print(alpha, beta)


