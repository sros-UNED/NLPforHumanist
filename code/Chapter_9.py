"""
Humanities Data Analysis: Case studies with Python
--------------------------------------------------
Folgert Karsdorp, Mike Kestemont & Allen Riddell
Chapter 9: A Topic Model of United States Supreme Court Opinions, 1900-2000
"""


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
# NOTE: fixed random seed for normal mixture model
import numpy.random; numpy.random.seed(1)
import random; random.seed(1)


# %%
import pandas as pd

df = pd.read_csv("data/tate.csv.gz", index_col='artId')
df[["artist", "acquisitionYear", "accession_number", "medium", "width", "height"]].head(2)


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
assert df.shape[0] == 63298, df.shape[0]


# %%
df = df.loc[(df['width'] < 8000) & (df['height'] < 8000)]
df = df.loc[(df['width'] >= 20) & (df['height'] >= 20)]


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
assert df.shape[0] == (63298 - 18), df.shape[0]


# %%
import numpy as np

df["width_log10"] = np.log10(df["width"])
df["height_log10"] = np.log10(df["height"])


# %%
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(10, 6))
df.plot(x='width_log10', y='height_log10', kind='scatter', alpha=0.02, ax=axes[0])
axes[0].set(xlabel=r'Width ($\log_{10}$)', ylabel=r'Height ($\log_{10}$)')

df['width_log10'].plot(kind='density', title=r'Width ($\log_{10}$)', ax=axes[1])
df['height_log10'].plot(kind='density', title=r'Height ($\log_{10}$)', ax=axes[2])
xlim = (1.2, 3.8); axes[1].set_xlim(*xlim); axes[2].set_xlim(*xlim)
plt.tight_layout();


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
np.set_printoptions(suppress=True)  # supress scientific notation


# %%
import sklearn.mixture as mixture

gmm = mixture.BayesianGaussianMixture(n_components=3, max_iter=200)
gmm.fit(df[['width_log10', 'height_log10']])

# order of components is arbitrary, sort by mixing proportions (decending)
order = np.argsort(gmm.weights_)[::-1]
means, covariances, weights = gmm.means_[order], gmm.covariances_[order], gmm.weights_[order]

# mu_1, mu_2, mu_3 in the equation above
print("μ's =", means.round(2))
# Sigma_1, Sigma_2, Sigma_3 in the equation above
print("Σ's =", covariances.round(4))
# theta_1, theta_2, theta_3 in the equation above
print("θ's =", weights.round(2))


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
assert (means[0].round(2) == [2.45, 2.4]).all()


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
# code not for public consumption, just for plot
import scipy.stats
import matplotlib.patches

# reorder responsibilties
responsibilities = gmm.predict_proba(df[['width_log10', 'height_log10']])[:, order]
assert (responsibilities.argmax(axis=1) == 0).mean() > 0.5
color_predicted = [f'C{i}' for i in responsibilities.argmax(axis=1)]
#color_true = [f'C{i}' for i in (df['mf'] == 'F').astype(int)]
x = np.linspace(df['width_log10'].min(), df['width_log10'].max(), 1000)
y = np.linspace(df['height_log10'].min(), df['height_log10'].max(), 1000)
X, Y = np.meshgrid(x, y)
fig, ax = plt.subplots()

for i, (mu, Sigma) in enumerate(zip(means, covariances)):
    pdf = scipy.stats.multivariate_normal(mean=mu, cov=Sigma).pdf
    Z = pdf(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    CS = ax.contour(X, Y, Z, 10, alpha=0.8, cmap="Greys")

ax.scatter(df['width_log10'], df['height_log10'],
           color=color_predicted, alpha=0.05)
ax.set(xlabel=r'Width ($\log_{10}$)', ylabel=r'Height ($\log_{10}$)')

patches = [
    matplotlib.patches.Patch(color=f'C{i}', label=rf'latent class {i+1} ($\theta_{i+1} = {theta:.2f}$)')
    for i, theta in enumerate(weights)
]
plt.legend(handles=patches)
plt.plot([0, 1], [0, 1], transform=ax.transAxes, alpha=0.2, linestyle='--', color="black") # diagonal line
plt.tight_layout();


# %%
import os
import gzip
import pandas as pd

with gzip.open('data/supreme-court-opinions-by-author.jsonl.gz', 'rt') as fh:
    df = pd.read_json(fh, lines=True).set_index(['us_reports_citation', 'authors'])


# %%
df.loc['323 US 214']


# %%
print(df.loc['323 US 214'].loc['murphy', 'text'][:500])


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
assert len(df) == 34_677


# %%
df['year'].hist(bins=50)


# %%
df['year'].describe()


# %%
import sklearn.feature_extraction.text as text

# min_df: ignore words occurring in fewer than `n` documents
# stop_words: ignore very common words ("the", "and", "or", "to", ...)
vec = text.CountVectorizer(lowercase=True, min_df=100, stop_words='english')
dtm = vec.fit_transform(df['text'])


# %%
print(f'Shape of document-term matrix: {dtm.shape}. '
      f'Number of tokens {dtm.sum()}')


# %%
import sklearn.decomposition as decomposition
model = decomposition.LatentDirichletAllocation(
    n_components=100, learning_method='online', random_state=1)


# %%
document_topic_distributions = model.fit_transform(dtm)


# %%
vocabulary = vec.get_feature_names()
# (# topics, # vocabulary)
assert model.components_.shape == (100, len(vocabulary))
# (# documents, # topics)
assert document_topic_distributions.shape == (dtm.shape[0], 100)


# %%
topic_names = [f'Topic {k}' for k in range(100)]
topic_word_distributions = pd.DataFrame(
    model.components_, columns=vocabulary, index=topic_names)
document_topic_distributions = pd.DataFrame(
    document_topic_distributions, columns=topic_names, index=df.index)


# %%
document_topic_distributions.loc['323 US 214'].loc['murphy'].head(10)


# %%
murphy_dissent = document_topic_distributions.loc['323 US 214'].loc['murphy']
murphy_dissent.sort_values(ascending=False).head(10)


# %%
topic_word_distributions.loc['Topic 8'].sort_values(ascending=False).head(18)


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
assert topic_word_distributions.loc['Topic 8'].shape == (13_231,)


# %%
weights_sample = document_topic_distributions.sample(8)
weights_ordered = weights_sample.apply(
    lambda row: row.sort_values(ascending=False).reset_index(drop=True), axis=1)
# transpose DataFrame so pandas plots weights on correct axis
ax = weights_ordered.T.plot()
ax.set(xlabel='Rank of mixing weight', ylabel='Probability', xlim=(0, 25));


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
# lauderdale_clark_label, our_topic_number, our_topic_words
lauderdale_clark_figure_3_mapping = (
    ('lands, indian, land', 59, 'indian, territory, indians'),
    ('tax, commerce, interstate', 89, 'commerce, interstate, state'),
    ('federal, immunity, law', 2, 'suit, action, states, ..., immunity'),
    ('military, aliens, aliens', 22, '..., alien,..., aliens, ..., deportation, immigration'),
    ('property, income, tax', 79, 'tax, taxes, property'),
    ('district, habeas, appeal', 43, 'court, federal, district, appeals, review, courts, habeas'),
    ('negligence, maritime, admiralty', 7, 'vessel, ship, admiralty'),
    ('patent, copyright, cable', 86, 'patent, ..., invention, patents'),
    ('search, fourth, warrant', 37, 'search, warrant, fourth'),
    ('jury, death, penalty', 3, 'sentence, death, sentencing, penalty'),
    ('school, religious, schools', 73, 'religious, funds, ... government, ..., establishment'),
    ('trial, counsel, testimony', 13, 'counsel, trial, defendant'),
    ('epa, waste, safety', 95, 'regulations, ..., agency, ..., safety, ..., air, epa' ),
    ('speech, ordinance, public', 58, 'speech, amendment, .., public'),
    ('antitrust, price, securities', 39, 'market, price, competition, act, antitrust'),
    ('child, abortion, children', 14, 'child, children, medical, ..., woman, ... abortion'),
    ('prison, inmates, parole', 67, 'prison, release, custody, parole' ),
    ('political, election, party', 23, 'speech, amendment, ..., political, party'),
    ('title, vii, employment', 55, 'title, discrimination, ..., vii'),
    ('offense, criminal, jeopardy', 78, 'criminal, ..., crime, offense'),
    ('union, labor, board', 24, 'board, union, labor'),
    ('damages, fees, attorneys', 87, 'attorney, fees, ..., costs'),
    ('commission, rates, gas', 97, 'rate, ..., gas, ..., rates'),
    ('congress, act, usc', 41, 'federal, congress, act, law'),
)
import pickle
import hashlib
assert len(lauderdale_clark_figure_3_mapping) == 24, len(lauderdale_clark_figure_3_mapping)
assert hashlib.sha256(pickle.dumps(lauderdale_clark_figure_3_mapping)).hexdigest() == '02e57c243457f5bc7f06f2bffa6d1bac68cc32d537db29d5dc6519ee11726a16'


# %%
# issueArea is coded as int but pandas does not allow us to mix int and
# values indicating NaN, so we represent the issueArea codes as `float`
# as a compromise.
scdb = pd.read_csv('data/SCDB_2016_01_caseCentered_Citation.csv.zip',
                   dtype={'issueArea': float}, encoding='latin1', index_col='caseId')
df_after_1945 = df.loc[df.case_id.isin(scdb.index)]


# %%
df_after_1945 = df_after_1945.join(scdb['issueArea'], on='case_id')


# %%
# for issueArea labels see SCDB documentation
# Exclude label 14 ("Private Action") as none of the opinions are
# assigned this label
spaeth_issue_areas = {
    1: "Criminal Procedure",
    2: "Civil Rights",
    3: "First Amendment",
    4: "Due Process",
    5: "Privacy",
    6: "Attorneys",
    7: "Unions",
    8: "Economic Activity",
    9: "Judicial Power",
    10: "Federalism",
    11: "Interstate Relations",
    12: "Federal Taxation",
    13: "Miscellaneous",
}
df_after_1945["issueArea"] = pd.Categorical(
    df_after_1945["issueArea"].replace(spaeth_issue_areas),
    categories=spaeth_issue_areas.values())


# %%
import collections

[(issue_area, count)] = collections.Counter(df_after_1945['issueArea']).most_common(1)
print(f'Issue area `{issue_area}` associated with {count} opinions, '
      f'{count / len(df_after_1945):.0%} of all opinions.')


# %%
document_word_counts = dtm.toarray().sum(axis=1)
document_topic_word_counts = document_topic_distributions.multiply(
    document_word_counts, axis='index'
)
df_after_1945 = df_after_1945.join(document_topic_word_counts)


# %%
df_after_1945.groupby('issueArea')["Topic 3"].sum()


# %%
topic_word_distributions.loc['Topic 3'].sort_values(ascending=False).head()


# %%
figure_3_topic_names = [f'Topic {t}' for _, t, _ in lauderdale_clark_figure_3_mapping]
df_plot = df_after_1945.groupby('issueArea')[figure_3_topic_names].sum()
df_plot = df_plot.rename(columns={
    f'Topic {t}': f'{t}: {figure_3_words}'
    for figure_3_words, t, _ in lauderdale_clark_figure_3_mapping
})
# heatmap code adapted from matplotlib documentation:
# https://matplotlib.org/gallery/images_contours_and_fields/
# image_annotated_heatmap.html

# `numpy.flipud` flips y-axis (to align with Lauderdale and Clark)
fig, ax = plt.subplots()
im = ax.imshow(np.flipud(df_plot.values), cmap="Greys")

ax.set_xticks(np.arange(len(df_plot.columns)))
ax.set_yticks(np.arange(len(df_plot.index)))
ax.set_xticklabels(df_plot.columns)
ax.set_yticklabels(reversed(df_plot.index))

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax.set_title('Topic model and expert label alignment')
fig.tight_layout()


# %%
minor_top_topics = topic_word_distributions['minor'].sort_values(ascending=False).head(5)


# %%
minor_top_topics_top_words = topic_word_distributions.loc[minor_top_topics.index].apply(
        lambda row: ', '.join(row.sort_values(ascending=False).head().index),
        axis=1,
)
minor_top_topics_top_words.name = 'topic_top_words'
minor_top_topics.to_frame().join(minor_top_topics_top_words)


# %%
opinion_of_interest = ('483 US 587', 'brennan')
document_topic_distributions.loc[opinion_of_interest, minor_top_topics.index]


# %%
print(df.loc[opinion_of_interest, 'text'].values[0][1000:2000])


# %%
opinion_of_interest = ('479 US 189', 'white_b')
print(f'"minor" count in 479 US 189:', sum('minor' in word.lower()
          for word in df.loc[opinion_of_interest, 'text'].values[0].split()))


# %%
document_topic_distributions.loc[opinion_of_interest, minor_top_topics.index.tolist() + ['Topic 23']]


# %%
topic_oi = 'Topic 23'
topic_oi_words = ', '.join(
    topic_word_distributions.loc[topic_oi].sort_values(ascending=False).head(8).index)
print(f'Topic 23 top words:\n  {topic_oi_words}')


# %%
print(df.loc[opinion_of_interest, 'text'][0][1000:1500])


# %%
import itertools

opinion_text = df.loc[opinion_of_interest, 'text'][0]
window_width, num_words = 3, len(opinion_text.split())
words = iter(opinion_text.split())
windows = [
    ' '.join(itertools.islice(words, 0, window_width))
    for _ in range(num_words // window_width)
]
print([window for window in windows if 'minor' in window])


# %%
labor_topic = 'Topic 24'
topic_word_distributions.loc[labor_topic].sort_values(ascending=False).head(10)


# %%
topic_top_words = topic_word_distributions.loc[labor_topic].sort_values(
    ascending=False).head(10).index
topic_top_words_joined = ', '.join(topic_top_words)
print(topic_top_words_joined)


# %%
# convert `dtm` (matrix) into an array:
opinion_word_counts = np.array(dtm.sum(axis=1)).ravel()
word_counts_by_year = pd.Series(opinion_word_counts).groupby(df.year.values).sum()
topic_word_counts = document_topic_distributions.multiply(opinion_word_counts,
                                                          axis='index')
topic_word_counts_by_year = topic_word_counts.groupby(df.year.values).sum()
topic_proportion_by_year = topic_word_counts_by_year.divide(word_counts_by_year,
                                                            axis='index')


# %%
topic_proportion_by_year.head()


# %%
import matplotlib.pyplot as plt

window = 3
topic_proportion_rolling = topic_proportion_by_year.loc[1900:, labor_topic].rolling(
    window=window).mean()
topic_proportion_rolling.plot()
plt.title(f'Prevalence of {labor_topic} ({window} year rolling average)'
          f'\n{topic_top_words_joined}');


# %%
# each tuple records the following in the order used by Lauderdale and Clark:
# (<Lauderdale and Clark top words>, <our topic number>, <our top words>)
lauderdale_clark_figure_3_mapping = (
    ('lands, indian, land', 59, 'indian, territory, indians'),
    ('tax, commerce, interstate', 89, 'commerce, interstate, state'),
    ('federal, immunity, law', 2, 'suit, action, states, ..., immunity'),
    ('military, aliens, aliens', 22, '..., alien,..., aliens, ..., deportation, immigration'),
    ('property, income, tax', 79, 'tax, taxes, property'),
    ('district, habeas, appeal', 43, 'court, federal, district, appeals, review, courts, habeas'),
    ('negligence, maritime, admiralty', 7, 'vessel, ship, admiralty'),
    ('patent, copyright, cable', 86, 'patent, ..., invention, patents'),
    ('search, fourth, warrant', 37, 'search, warrant, fourth'),
    ('jury, death, penalty', 3, 'sentence, death, sentencing, penalty'),
    ('school, religious, schools', 73, 'religious, funds, ... government, ..., establishment'),
    ('trial, counsel, testimony', 13, 'counsel, trial, defendant'),
    ('epa, waste, safety', 95, 'regulations, ..., agency, ..., safety, ..., air, epa' ),
    ('speech, ordinance, public', 58, 'speech, amendment, .., public'),
    ('antitrust, price, securities', 39, 'market, price, competition, act, antitrust'),
    ('child, abortion, children', 14, 'child, children, medical, ..., woman, ... abortion'),
    ('prison, inmates, parole', 67, 'prison, release, custody, parole' ),
    ('political, election, party', 23, 'speech, amendment, ..., political, party'),
    ('title, vii, employment', 55, 'title, discrimination, ..., vii'),
    ('offense, criminal, jeopardy', 78, 'criminal, ..., crime, offense'),
    ('union, labor, board', 24, 'board, union, labor'),
    ('damages, fees, attorneys', 87, 'attorney, fees, ..., costs'),
    ('commission, rates, gas', 97, 'rate, ..., gas, ..., rates'),
    ('congress, act, usc', 41, 'federal, congress, act, law'),
)


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
# integrity check. We define this table in two places. The definitions must match.
import pickle
import hashlib
assert len(lauderdale_clark_figure_3_mapping) == 24, len(lauderdale_clark_figure_3_mapping)
assert hashlib.sha256(pickle.dumps(lauderdale_clark_figure_3_mapping)).hexdigest() == '02e57c243457f5bc7f06f2bffa6d1bac68cc32d537db29d5dc6519ee11726a16'


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
import matplotlib.pyplot as plt
discrimination_topic = 'Topic 15'
topic_top_words = topic_word_distributions.loc[discrimination_topic].sort_values(ascending=False).head(12).index
topic_top_words_joined = f"{', '.join(topic_top_words[:6])}\n{', '.join(topic_top_words[6:])}"
window = 3
topic_proportion_by_year.loc[1900:, discrimination_topic].rolling(window=window).mean().plot()
plt.suptitle(f'Prevalence of {discrimination_topic} ({window} year rolling average)', y=1.05, fontsize='x-large')
plt.title(topic_top_words_joined, fontsize='medium', style='italic')
plt.axvline(x=1954, color='orange', label='Brown v. Board')
plt.ylabel("Topic prevalence")
plt.xlabel("Year")
plt.legend()
plt.savefig('figures/discrimination-topic.png')


