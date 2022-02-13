"""
Humanities Data Analysis: Case studies with Python
--------------------------------------------------
Folgert Karsdorp, Mike Kestemont & Allen Riddell
Chapter 8: Stylometry and the Voice of Hildegard
"""


# %%
import os
import tarfile

tf = tarfile.open('data/hildegard.tar.gz', 'r')
tf.extractall('data')


# %%
def load_directory(directory, max_length):
    documents, authors, titles = [], [], []
    for filename in os.scandir(directory):
        if not filename.name.endswith('.txt'):
            continue
        author, _ = os.path.splitext(filename.name)

        with open(filename.path) as f:
            contents = f.read()
        lemmas = contents.lower().split()
        start_idx, end_idx, segm_cnt = 0, max_length, 1

        # extract slices from the text:
        while end_idx < len(lemmas):
            documents.append(' '.join(lemmas[start_idx:end_idx]))
            authors.append(author[0])
            title = filename.name.replace('.txt', '').split('_')[1]
            titles.append(f"{title}-{segm_cnt}")

            start_idx += max_length
            end_idx += max_length
            segm_cnt += 1

    return documents, authors, titles


# %%
documents, authors, titles = load_directory('data/hildegard/texts', 10000)


# %%
import sklearn.feature_extraction.text as text

vectorizer = text.CountVectorizer(max_features=30, token_pattern=r"(?u)\b\w+\b")
v_documents = vectorizer.fit_transform(documents).toarray()

print(v_documents.shape)
print(vectorizer.get_feature_names()[:5])


# %%
vocab = [l.strip() for l in open('data/hildegard/wordlist.txt')
         if not l.startswith('#') and l.strip()][:65]

vectorizer = text.CountVectorizer(token_pattern=r"(?u)\b\w+\b", vocabulary=vocab)
v_documents = vectorizer.fit_transform(documents).toarray()

print(v_documents.shape)
print(vectorizer.get_feature_names()[:5])


# %%
import sklearn.preprocessing as preprocessing

v_documents = preprocessing.normalize(v_documents.astype(float), norm='l1')


# %%
import scipy.spatial.distance as scidist

scaler = preprocessing.StandardScaler()
s_documents = scaler.fit_transform(v_documents)

test_doc = s_documents[0]
distances = [
    scidist.cityblock(test_doc, train_doc) for train_doc in s_documents[1:]
]


# %%
import numpy as np

print(authors[np.argmin(distances) + 1])


# %%
import sklearn.model_selection as model_selection

test_size = len(set(authors)) * 2
(train_documents, test_documents,
 train_authors, test_authors) = model_selection.train_test_split(
    v_documents, authors, test_size=test_size, stratify=authors, random_state=1)

print(f'N={test_documents.shape[0]} test documents with '
      f'V={test_documents.shape[1]} features.')

print(f'N={train_documents.shape[0]} train documents with '
      f'V={train_documents.shape[1]} features.')


# %%
scaler = preprocessing.StandardScaler()
scaler.fit(train_documents)

train_documents = scaler.transform(train_documents)
test_documents = scaler.transform(test_documents)


# %%
distances = scidist.cdist(test_documents, train_documents, metric='cityblock')


# %%
nn_predictions = np.array(train_authors)[np.argmin(distances, axis=1)]
print(nn_predictions[:3])


# %%
class Delta:
    """Delta-Based Authorship Attributer."""

    def fit(self, X, y):
        """Fit (or train) the attributer.

        Arguments:
            X: a two-dimensional array of size NxV, where N represents
               the number of training documents, and V represents the
               number of features used.
            y: a list (or NumPy array) consisting of the observed author
                for each document in X.

        Returns:
            Delta: A trained (fitted) instance of Delta.

        """
        self.train_y = np.array(y)
        self.scaler = preprocessing.StandardScaler(with_mean=False)
        self.train_X = self.scaler.fit_transform(X)

        return self

    def predict(self, X, metric='cityblock'):
        """Predict the authorship for each document in X.

        Arguments:
            X: a two-dimensional (sparse) matrix of size NxV, where N
               represents the number of test documents, and V represents
               the number of features used during the fitting stage of
               the attributer.
            metric (str, optional): the metric used for computing
               distances between documents. Defaults to 'cityblock'.

        Returns:
            ndarray: the predicted author for each document in X.

        """
        X = self.scaler.transform(X)
        dists = scidist.cdist(X, self.train_X, metric=metric)
        return self.train_y[np.argmin(dists, axis=1)]


# %%
import sklearn.metrics as metrics

delta = Delta()
delta.fit(train_documents, train_authors)
preds = delta.predict(test_documents)

for true, pred in zip(test_authors, preds):
    _connector = 'WHEREAS' if true != pred else 'and'
    print(f'Observed author is {true} {_connector} {pred} was predicted.')

accuracy = metrics.accuracy_score(preds, test_authors)
print(f"\nAccuracy of predictions: {accuracy:.1f}")


# %%
with open('data/hildegard/texts/test/B_Mart.txt') as f:
    test_doc = f.read()

v_test_doc = vectorizer.transform([test_doc]).toarray()
v_test_doc = preprocessing.normalize(v_test_doc.astype(float), norm='l1')

print(delta.predict(v_test_doc)[0])


# %%
train_documents, train_authors, train_titles = load_directory('data/hildegard/texts', 3301)

vectorizer = text.CountVectorizer(token_pattern=r"(?u)\b\w+\b", vocabulary=vocab)
v_train_documents = vectorizer.fit_transform(train_documents).toarray()
v_train_documents = preprocessing.normalize(v_train_documents.astype(float), norm='l1')

delta = Delta().fit(v_train_documents, train_authors)


# %%
test_docs, test_authors, test_titles = load_directory('data/hildegard/texts/test', 3301)

v_test_docs = vectorizer.transform(test_docs).toarray()
v_test_docs = preprocessing.normalize(v_test_docs.astype(float), norm='l1')


# %%
predictions = delta.predict(v_test_docs)

for filename, prediction in zip(test_titles, predictions):
    print(f'{filename} -> {prediction}')


# %%
predictions = delta.predict(v_test_docs, metric='cosine')
for filename, prediction in zip(test_titles, predictions):
    print(f'{filename} -> {prediction}')


# %%
vectorizer = text.CountVectorizer(token_pattern=r"(?u)\b\w+\b", vocabulary=vocab)

v_documents = vectorizer.fit_transform(documents).toarray()
v_documents = preprocessing.normalize(v_documents.astype(np.float64), 'l1')
scaler = preprocessing.StandardScaler()
v_documents = scaler.fit_transform(v_documents)

print(f'N={v_documents.shape[0]} documents with '
      f'V={v_documents.shape[1]} features.')


# %%
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hierarchy

# 1. Calculate pairwise distances
dm = scidist.pdist(v_documents, 'cityblock')

# 2. Establish branch structure
linkage_object = hierarchy.linkage(dm, method='complete')

# 3. Visualize
def plot_tree(linkage_object, labels, figsize=(10, 5), ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    with plt.rc_context({'lines.linewidth': 1.0}):
        hierarchy.dendrogram(
            linkage_object, labels=labels, ax=ax,
            link_color_func=lambda c: 'black',
            leaf_font_size=10, leaf_rotation=90)
    # Remove ticks and spines
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    for s in ax.spines.values():
        s.set_visible(False)

plot_tree(linkage_object, authors)


# %%
v_test_docs = vectorizer.transform(test_docs[1:])
v_test_docs = preprocessing.normalize(v_test_docs.astype(float), norm='l1')
v_test_docs = scaler.transform(v_test_docs.toarray())

all_documents = np.vstack((v_documents, v_test_docs))

dm = scidist.pdist(all_documents, 'cityblock')
linkage_object = hierarchy.linkage(dm, method='complete')

plot_tree(linkage_object, authors + test_titles[1:], figsize=(12, 5))


# %%
words = vectorizer.get_feature_names()
authors = np.array(authors)
x = v_documents[:, words.index('super')]
y = v_documents[:, words.index('propter')]

fig, ax = plt.subplots()
for author in set(authors):
    ax.scatter(x[authors==author], y[authors==author], label=author)
ax.set(xlabel='super', ylabel='propter')
plt.legend();


# %%
fig, ax = plt.subplots()
ax.scatter(x, y, facecolors='none')
for p1, p2, author in zip(x, y, authors):
    ax.text(p1, p2, author[0], fontsize=12,
            ha='center', va='center')
ax.set(xlabel='super', ylabel='propter');


# %%
import sklearn.decomposition

pca = sklearn.decomposition.PCA(n_components=2)
documents_proj = pca.fit_transform(v_documents)

print(v_documents.shape)
print(documents_proj.shape)


# %%
c1, c2 = documents_proj[:, 0], documents_proj[:, 1]

fig, ax = plt.subplots()
ax.scatter(c1, c2, facecolors='none')

for p1, p2, author in zip(c1, c2, authors):
    ax.text(p1, p2, author[0], fontsize=12,
            ha='center', va='center')

ax.set(xlabel='PC1', ylabel='PC2');


# %%
fig, ax = plt.subplots(figsize=(4, 8))

for idx in range(pca.components_.shape[0]):
    ax.axvline(idx, linewidth=2, color='lightgrey')
    for score, author in zip(documents_proj[:, idx], authors):
        ax.text(
            idx, score, author[0], fontsize=10,
            va='center', ha='center')

ax.axhline(0, ls='dotted', c='black')
ax.set(
    xlim=(-0.5, 1.5), ylim=(-12, 12),
    xlabel='PCs', ylabel='Component scores',
    xticks=[0, 1], xticklabels=["PC1", "PC2"]);


# %%
pca = sklearn.decomposition.PCA(n_components=2)
pca.fit(v_documents)

print(pca.components_.shape)


# %%
pca = sklearn.decomposition.PCA(n_components=36)
pca.fit(v_documents)

print(len(pca.explained_variance_ratio_))


# %%
print(sum(pca.explained_variance_ratio_))


# %%
var_exp = pca.explained_variance_ratio_
cum_var_exp = np.cumsum(var_exp)

fig, ax = plt.subplots()

ax.bar(range(36), var_exp, alpha=0.5, align='center',
        label='individual explained variance')

ax.step(range(36), cum_var_exp, where='mid',
         label='cumulative explained variance')

ax.axhline(0.05, ls='dotted', color="black")
ax.set(ylabel='Explained variance ratio', xlabel='Principal components')
ax.legend(loc='best');


# %%
print(var_exp[0] + var_exp[1])


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
# SET PRINTING FOR NEXT CELL
np.set_printoptions(threshold=50)


# %%
pca = sklearn.decomposition.PCA(n_components=2).fit(v_documents)
print(pca.components_)


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
# RESET PRINTING
np.set_printoptions(threshold=1000)


# %%
X_centered = v_documents - np.mean(v_documents, axis=0)
X_bar1 = np.dot(X_centered, pca.components_.transpose())
X_bar2 = pca.transform(v_documents)


# %%
print(pca.components_.shape)
comps = pca.components_.transpose()
print(comps.shape)


# %%
vocab = vectorizer.get_feature_names()
vocab_weights = sorted(zip(vocab, comps[:, 0]))


# %%
print('Positive loadings:')
print('\n'.join(f'{w} -> {s}' for w, s in vocab_weights[:5]))


# %%
print('Negative loadings:')
print('\n'.join(f'{w} -> {s}' for w, s in vocab_weights[-5:]))


# %%
l1, l2 = comps[:, 0], comps[:, 1]

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(l1, l2, facecolors='none')

for x, y, l in zip(l1, l2, vocab):
    ax.text(x, y, l, ha='center', va='center', color='darkgrey', fontsize=12)


# %%
import mpl_axes_aligner.align

def plot_pca(document_proj, loadings, var_exp, labels):
    # first the texts:
    fig, text_ax = plt.subplots(figsize=(10, 10))
    x1, x2 = documents_proj[:, 0], documents_proj[:, 1]
    text_ax.scatter(x1, x2, facecolors='none')
    for p1, p2, author in zip(x1, x2, labels):
        color = 'red' if author not in ('H', 'G', 'B') else 'black'
        text_ax.text(p1, p2, author, ha='center',
                     color=color, va='center', fontsize=12)

    # add variance information to the axis labels:
    text_ax.set_xlabel(f'PC1 ({var_exp[0] * 100:.2f}%)')
    text_ax.set_ylabel(f'PC2 ({var_exp[1] * 100:.2f}%)')

    # now the loadings:
    loadings_ax = text_ax.twinx().twiny()
    l1, l2 = loadings[:, 0], loadings[:, 1]
    loadings_ax.scatter(l1, l2, facecolors='none');
    for x, y, loading in zip(l1, l2, vectorizer.get_feature_names()):
        loadings_ax.text(x, y, loading, ha='center', va='center',
                         color='darkgrey', fontsize=12)

    mpl_axes_aligner.align.yaxes(text_ax, 0, loadings_ax, 0)
    mpl_axes_aligner.align.xaxes(text_ax, 0, loadings_ax, 0)
    # add lines through origins:
    plt.axvline(0, ls='dashed', c='lightgrey', zorder=0)
    plt.axhline(0, ls='dashed', c='lightgrey', zorder=0);

# fit the pca:
pca = sklearn.decomposition.PCA(n_components=2)
documents_proj = pca.fit_transform(v_documents)
loadings = pca.components_.transpose()
var_exp = pca.explained_variance_ratio_

plot_pca(documents_proj, loadings, var_exp, authors)


# %%
all_documents = preprocessing.scale(np.vstack((v_documents, v_test_docs)))
pca = sklearn.decomposition.PCA(n_components=2)
documents_proj = pca.fit_transform(all_documents)
loadings = pca.components_.transpose()
var_exp = pca.explained_variance_ratio_

plot_pca(documents_proj, loadings, var_exp, list(authors) + test_titles[1:])


# %%
import tarfile

tf = tarfile.open('data/caesar.tar.gz', 'r')
tf.extractall('data')


