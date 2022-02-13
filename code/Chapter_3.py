"""
Humanities Data Analysis: Case studies with Python
--------------------------------------------------
Folgert Karsdorp, Mike Kestemont & Allen Riddell
Chapter 3: Exploring Texts using the Vector Space Model
"""


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
import numpy as np
import matplotlib.pyplot as plt

document_term_matrix = np.array([[1, 16], [2, 18], [35, 0], [39, 1]])
labels = '$d_1$', '$d_2$', '$d_3$', '$d_4$'
plt.quiver([0, 0, 0, 0], [0, 0, 0, 0],
           document_term_matrix[:, 0], document_term_matrix[:, 1],
           color=["C0", "C1", "C2", "C3"], angles='xy', scale_units='xy', scale=1)
for i, label in enumerate(labels):
    plt.annotate(label, xy=document_term_matrix[i], fontsize=15)
plt.ylim(-1, 20); plt.xlim(-1, 44)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 3");


# %%
corpus = ["D'où me vient ce désordre, Aufide, et que veut dire",
          "Madame, il était temps qu'il vous vînt du secours:",
          "Ah! Monsieur, c'est donc vous?",
          "Ami, j'ai beau rêver, toute ma rêverie",
          "Ne me parle plus tant de joie et d'hyménée;",
          "Il est vrai, Cléobule, et je veux l'avouer,",
          "Laisse-moi mon chagrin, tout injuste qu'il est;",
          "Ton frère, je l'avoue, a beaucoup de mérite;",
          "J'en demeure d'accord, chacun a sa méthode;",
          'Pour prix de votre amour que vous peignez extrême,']


# %%
document = corpus[2]
print(document.split())


# %%
import nltk
import nltk.tokenize

# download the most recent punkt package
nltk.download('punkt', quiet=True)

document = corpus[3]
print(nltk.tokenize.word_tokenize(document, language='french'))


# %%
import re


PUNCT_RE = re.compile(r'[^\w\s]+$')


def is_punct(string):
    """Check if STRING is a punctuation marker or a sequence of
       punctuation markers.

    Arguments:
        string (str): a string to check for punctuation markers.

    Returns:
        bool: True is string is a (sequence of) punctuation marker(s),
            False otherwise.

    Examples:
        >>> is_punct("!")
        True
        >>> is_punct("Bonjour!")
        False
        >>> is_punct("¿Te gusta el verano?")
        False
        >>> is_punct("...")
        True
        >>> is_punct("«»...")
        True

    """
    return PUNCT_RE.match(string) is not None


# %%
tokens = nltk.tokenize.word_tokenize(corpus[2], language='french')

# Loop with a standard for-loop
tokenized = []
for token in tokens:
    if not is_punct(token):
        tokenized.append(token)
print(tokenized)

# Loop with a list comprehension
tokenized = [token for token in tokens if not is_punct(token)]
print(tokenized)


# %%
def preprocess_text(text, language, lowercase=True):
    """Preprocess a text.

    Perform a text preprocessing procedure, which transforms a string
    object into a list of word tokens without punctuation markers.

    Arguments:
        text (str): a string representing a text.
        language (str): a string specifying the language of text.
        lowercase (bool, optional): Set to True to lowercase all
            word tokens. Defaults to True.

    Returns:
        list: a list of word tokens extracted from text, excluding
            punctuation.

    Examples:
        >>> preprocess_text("Ah! Monsieur, c'est donc vous?", 'french')
        ["ah", "monsieur", "c'est", "donc", "vous"]

    """
    if lowercase:
        text = text.lower()
    tokens = nltk.tokenize.word_tokenize(text, language=language)
    tokens = [token for token in tokens if not is_punct(token)]
    return tokens


# %%
for document in corpus[2:4]:
    print('Original:', document)
    print('Tokenized:', preprocess_text(document, 'french'))


# %%
import collections

vocabulary = collections.Counter()
for document in corpus:
    vocabulary.update(preprocess_text(document, 'french'))


# %%
print(vocabulary.most_common(n=5))


# %%
print('Original vocabulary size:', len(vocabulary))
pruned_vocabulary = {token for token, count in vocabulary.items() if count > 1}
print(pruned_vocabulary)
print('Pruned vocabulary size:', len(pruned_vocabulary))


# %%
n = 5
print('Original vocabulary size:', len(vocabulary))
pruned_vocabulary = {token for token, _ in vocabulary.most_common()[n:]}
print('Pruned vocabulary size:', len(pruned_vocabulary))


# %%
def extract_vocabulary(tokenized_corpus, min_count=1, max_count=float('inf')):
    """Extract a vocabulary from a tokenized corpus.

    Arguments:
        tokenized_corpus (list): a tokenized corpus represented, list
            of lists of strings.
        min_count (int, optional): the minimum occurrence count of a
            vocabulary item in the corpus.
        max_count (int, optional): the maximum occurrence count of a
            vocabulary item in the corpus. Defaults to inf.

    Returns:
        list: An alphabetically ordered list of unique words in the
            corpus, of which the frequencies adhere to the specified
            minimum and maximum count.

    Examples:
        >>> corpus = [['the', 'man', 'love', 'man', 'the'],
                      ['the', 'love', 'book', 'wise', 'drama'],
                      ['a', 'story', 'book', 'drama']]
        >>> extract_vocabulary(corpus, min_count=2)
        ['book', 'drama', 'love', 'man', 'the']

    """
    vocabulary = collections.Counter()
    for document in tokenized_corpus:
        vocabulary.update(document)
    vocabulary = {word for word, count in vocabulary.items()
                  if count >= min_count and count <= max_count}
    return sorted(vocabulary)


# %%
tokenized_corpus = [preprocess_text(document, 'french') for document in corpus]
vocabulary = extract_vocabulary(tokenized_corpus)


# %%
bags_of_words = []
for document in tokenized_corpus:
    tokens = [word for word in document if word in vocabulary]
    bags_of_words.append(collections.Counter(tokens))

print(bags_of_words[2])


# %%
def corpus2dtm(tokenized_corpus, vocabulary):
    """Transform a tokenized corpus into a document-term matrix.

    Arguments:
        tokenized_corpus (list): a tokenized corpus as a list of
        lists of strings. vocabulary (list): An list of unique words.

    Returns:
        list: A list of lists representing the frequency of each term
              in `vocabulary` for each document in the corpus.

    Examples:
        >>> tokenized_corpus = [['the', 'man', 'man', 'smart'],
                                ['a', 'the', 'man' 'love'],
                                ['love', 'book', 'journey']]
        >>> vocab = ['book', 'journey', 'man', 'love']
        >>> corpus2dtm(tokenized_corpus, vocabulary)
        [[0, 0, 2, 0], [0, 0, 1, 1], [1, 1, 0, 1]]

    """
    document_term_matrix = []
    for document in tokenized_corpus:
        document_counts = collections.Counter(document)
        row = [document_counts[word] for word in vocabulary]
        document_term_matrix.append(row)
    return document_term_matrix


document_term_matrix = corpus2dtm(tokenized_corpus, vocabulary)


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
import pandas as pd
pd.DataFrame(document_term_matrix, columns=vocabulary).iloc[0:5, 0:15]


# %%
import numpy as np

document_term_matrix = np.array(document_term_matrix)
print(document_term_matrix.shape)


# %%
import os
import lxml.etree
import tarfile

tf = tarfile.open('data/theatre-classique.tar.gz', 'r')
tf.extractall('data')

subgenres = ('Comédie', 'Tragédie', 'Tragi-comédie')

plays, titles, genres = [], [], []
for fn in os.scandir('data/theatre-classique'):
    # Only include XML files
    if not fn.name.endswith('.xml'):
        continue
    tree = lxml.etree.parse(fn.path)
    genre = tree.find('//genre')
    title = tree.find('//title')
    if genre is not None and genre.text in subgenres:
        lines = []
        for line in tree.xpath('//l|//p'):
            lines.append(' '.join(line.itertext()))
        text = '\n'.join(lines)
        plays.append(text)
        genres.append(genre.text)
        titles.append(title.text)


# %%
import matplotlib.pyplot as plt

counts = collections.Counter(genres)

fig, ax = plt.subplots()
ax.bar(counts.keys(), counts.values(), width=0.3)
ax.set(xlabel="genre", ylabel="count");


# %%
plays_tok = [preprocess_text(play, 'french') for play in plays]
vocabulary = extract_vocabulary(plays_tok, min_count=2)
document_term_matrix = np.array(corpus2dtm(plays_tok, vocabulary))

print(f"document-term matrix with "
      f"|D| = {document_term_matrix.shape[0]} documents and "
      f"|V| = {document_term_matrix.shape[1]} words.")


# %%
monsieur_idx = vocabulary.index('monsieur')
sang_idx = vocabulary.index('sang')

monsieur_counts = document_term_matrix[:, monsieur_idx]
sang_counts = document_term_matrix[:, sang_idx]


# %%
genres = np.array(genres)


# %%
fig, ax = plt.subplots()

for genre in ('Comédie', 'Tragédie', 'Tragi-comédie'):
    ax.scatter(monsieur_counts[genres == genre],
               sang_counts[genres == genre],
               label=genre, alpha=0.7)

ax.set(xlabel='monsieur', ylabel='sang')
plt.legend();


# %%
tr_means = document_term_matrix[genres == 'Tragédie'].mean(axis=0)
co_means = document_term_matrix[genres == 'Comédie'].mean(axis=0)
tc_means = document_term_matrix[genres == 'Tragi-comédie'].mean(axis=0)


# %%
print(tr_means.shape)


# %%
print('Mean absolute frequency of "monsieur"')
print(f'   in comédies: {co_means[monsieur_idx]:.2f}')
print(f'   in tragédies: {tr_means[monsieur_idx]:.2f}')
print(f'   in tragi-comédies: {tc_means[monsieur_idx]:.2f}')


# %%
fig, ax = plt.subplots()

ax.scatter(
    co_means[monsieur_idx], co_means[sang_idx], label='Comédies')
ax.scatter(
    tr_means[monsieur_idx], tr_means[sang_idx], label='Tragédie')
ax.scatter(
    tc_means[monsieur_idx], tc_means[sang_idx], label='Tragi-comédies')

ax.set(xlabel='monsieur', ylabel='sang')
plt.legend();


# %%
tragedy = np.array([tr_means[monsieur_idx], tr_means[sang_idx]])
comedy = np.array([co_means[monsieur_idx], co_means[sang_idx]])
tragedy_comedy = np.array([tc_means[monsieur_idx], tc_means[sang_idx]])


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
fig, ax = plt.subplots()

ax.plot([tr_means[monsieur_idx], tc_means[monsieur_idx]],
        [tr_means[sang_idx], tc_means[sang_idx]],
        'darkgrey', lw=2, ls='--')
ax.plot([tr_means[monsieur_idx], co_means[monsieur_idx]],
        [tr_means[sang_idx], co_means[sang_idx]],
        'darkgrey', lw=2, ls='--')
ax.plot([tc_means[monsieur_idx], co_means[monsieur_idx]],
        [tc_means[sang_idx], co_means[sang_idx]],
        'darkgrey', lw=2, ls='--')

ax.scatter(co_means[monsieur_idx], co_means[sang_idx],
           label='Comédies', zorder=3)
ax.scatter(tr_means[monsieur_idx], tr_means[sang_idx],
           label='Tragédie', zorder=3)
ax.scatter(tc_means[monsieur_idx], tc_means[sang_idx],
           label='Tragi-comédies', zorder=3)

ax.set(xlabel='monsieur', ylabel='sang')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3);


# %%
def euclidean_distance(a, b):
    """Compute the Euclidean distance between two vectors.

    Note: ``numpy.linalg.norm(a - b)`` performs the
    same calculation using a slightly faster method.

    Arguments:
        a (numpy.ndarray): a vector of floats or ints.
        b (numpy.ndarray): a vector of floats or ints.

    Returns:
        float: The euclidean distance between vector a and b.

    Examples:
        >>> import numpy as np
        >>> a = np.array([1, 4, 2, 8])
        >>> b = np.array([2, 1, 4, 7])
        >>> round(euclidean_distance(a, b), 2)
        3.87

    """
    return np.sqrt(np.sum((a - b) ** 2))


# %%
tc = euclidean_distance(tragedy, comedy)
print(f'tragédies - comédies:       {tc:.2f}')

ttc = euclidean_distance(tragedy, tragedy_comedy)
print(f'tragédies - tragi-comédies: {ttc:.2f}')

ctc = euclidean_distance(comedy, tragedy_comedy)
print(f' comédies - tragi-comédies: {ctc:.2f}')


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
# following two blocks with much appreciated help from:
# https://stackoverflow.com/questions/25227100/best-way-to-plot-an-angle-between-two-lines-in-matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Arc
import math

def get_angle_plot(line1, line2, offset = 1, color = None, origin = [0,0], len_x_axis = 1, len_y_axis = 1):

    l1xy = line1.get_xydata()

    # Angle between line1 and x-axis
    slope1 = (l1xy[1][1] - l1xy[0][1]) / float(l1xy[1][0] - l1xy[0][0])
    angle1 = abs(math.degrees(math.atan(slope1))) # Taking only the positive angle

    l2xy = line2.get_xydata()

    # Angle between line2 and x-axis
    slope2 = (l2xy[1][1] - l2xy[0][1]) / float(l2xy[1][0] - l2xy[0][0])
    angle2 = abs(math.degrees(math.atan(slope2)))

    theta1 = min(angle1, angle2)
    theta2 = max(angle1, angle2)

    angle = theta2 - theta1

    if color is None:
        color = line1.get_color() # Uses the color of line 1 if color parameter is not passed.

    return Arc(origin, len_x_axis*offset, len_y_axis*offset, 0, theta1, theta2, color=color)

fig, ax = plt.subplots()

ax.scatter(co_means[monsieur_idx], co_means[sang_idx],
           label='Comédies', zorder=3)
ax.scatter(tr_means[monsieur_idx], tr_means[sang_idx],
           label='Tragédie', zorder=3)
ax.scatter(tc_means[monsieur_idx], tc_means[sang_idx],
           label='Tragic-comédies', zorder=3)

# plot vectors
line_1 = Line2D([co_means[monsieur_idx], 0], [co_means[sang_idx], 0], 2, lw=2, ls='--', c='darkgrey')
line_2 = Line2D([tr_means[monsieur_idx], 0], [tr_means[sang_idx], 0], 1, lw=2, ls='--', c='darkgrey')
line_3 = Line2D([tc_means[monsieur_idx], 0], [tc_means[sang_idx], 0], 1, lw=2, ls='--', c='darkgrey')

ax = plt.gca()
ax.add_line(line_1)
ax.add_line(line_2)
ax.add_line(line_3)

angle_plot = get_angle_plot(line_1, line_2, 50)
ax.add_patch(angle_plot) # To display the angle arc

angle_plot = get_angle_plot(line_1, line_3, 12)
ax.add_patch(angle_plot) # To display the angle arc

angle_plot = get_angle_plot(line_2, line_3, 25)
ax.add_patch(angle_plot) # To display the angle arc

plt.xlabel('monsieur')
plt.ylabel('sang')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
plt.tight_layout()


# %%
def vector_len(v):
    """Compute the length (or norm) of a vector."""
    return np.sqrt(np.sum(v ** 2))


# %%
def cosine_distance(a, b):
    """Compute the cosine distance between two vectors.

    Arguments:
        a (numpy.ndarray): a vector of floats or ints.
        b (numpy.ndarray): a vector of floats or ints.

    Returns:
        float: cosine distance between vector a and b.

    Note:
        See also scipy.spatial.distance.cdist

    Examples:
        >>> import numpy as np
        >>> a = np.array([1, 4, 2, 8])
        >>> b = np.array([2, 1, 4, 7])
        >>> round(cosine_distance(a, b), 2)
        0.09

    """
    return 1 - np.dot(a, b) / (vector_len(a) * vector_len(b))


# %%
tc = cosine_distance(tragedy, comedy)
print(f'tragédies - comédies:       {tc:.2f}')

ttc = cosine_distance(tragedy, tragedy_comedy)
print(f'tragédies - tragi-comédies: {ttc:.2f}')

ctc = cosine_distance(comedy, tragedy_comedy)
print(f' comédies - tragi-comédies: {ctc:.2f}')


# %%
def city_block_distance(a, b):
    """Compute the city block distance between two vectors.

    Arguments:
        a (numpy.ndarray): a vector of floats or ints.
        b (numpy.ndarray): a vector of floats or ints.

    Returns:
        {int, float}: The city block distance between vector a and b.

    Examples:
        >>> import numpy as np
        >>> a = np.array([1, 4, 2, 8])
        >>> b = np.array([2, 1, 4, 7])
        >>> city_block_distance(a, b)
        7

    """
    return np.abs(a - b).sum()


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
fig, ax = plt.subplots()

monsieur_trag = tr_means[monsieur_idx]
sang_trag = tr_means[sang_idx]
monsieur_com = co_means[monsieur_idx]
sang_com = co_means[sang_idx]
monsieur_tc = tc_means[monsieur_idx]
sang_tc = tc_means[sang_idx]


# trag-tc
ax.plot([monsieur_trag, monsieur_tc], [sang_tc, sang_tc],
        'C2', lw=2, ls='--')
ax.plot([monsieur_trag, monsieur_trag], [sang_tc, sang_trag],
        'C2', lw=2, ls='--')

# com-tc
ax.plot([monsieur_tc, monsieur_tc], [sang_tc, sang_com],
        'C0', lw=2, ls='--')
ax.plot([monsieur_tc, monsieur_com], [sang_com, sang_com],
        'C0', lw=2, ls='--')

# trag-com
ax.plot([monsieur_trag, monsieur_com], [sang_trag, sang_trag],
        'C1', lw=2, ls='--')
ax.plot([monsieur_com, monsieur_com], [sang_trag, sang_com],
        'C1', lw=2, ls='--')

ax.scatter(co_means[monsieur_idx], co_means[sang_idx],
           label='Comédies', zorder=3)
ax.scatter(tr_means[monsieur_idx], tr_means[sang_idx],
           label='Tragédie', zorder=3)
ax.scatter(tc_means[monsieur_idx], tc_means[sang_idx],
           label='Tragic-comédies', zorder=3)

ax.set(xlabel='monsieur', ylabel='sang')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3);


# %%
tc = city_block_distance(tragedy, comedy)
print(f'tragédies - comédies:       {tc:.2f}')

ttc = city_block_distance(tragedy, tragedy_comedy)
print(f'tragédies - tragi-comédies: {ttc:.2f}')

ctc = city_block_distance(comedy, tragedy_comedy)
print(f' comédies - tragi-comédies: {ctc:.2f}')


# %%
import scipy.spatial.distance as dist

genre_vectors = {'tragédie': tr_means, 'comédie': co_means, 'tragi-comédie': tc_means}
metrics = {'cosine': dist.cosine, 'manhattan': dist.cityblock, 'euclidean': dist.euclidean}

import itertools

for metric_name, metric_fn in metrics.items():
    print(metric_name)
    for v1, v2 in itertools.combinations(genre_vectors, 2):
        distance = metric_fn(genre_vectors[v1], genre_vectors[v2])
        print(f'   {v1} - {v2}: {distance:.2f}')


# %%
def nearest_neighbors(X, metric='cosine'):
    """Retrieve the nearest neighbor for each row in a 2D array.

    Arguments:
        X (numpy.ndarray): a 2D array.
        metric (str): the distance metric to be used,
            one of: 'cosine', 'manhattan', 'euclidean'

    Returns:
        neighbors (list): A list of integers, corresponding to
            the index of each row's nearest neighbor.

    Examples:
        >>> X = np.array([[1, 4, 2], [5, 5, 1], [1, 2, 1]])
        >>> nearest_neighbors(X, metric='manhattan')
        [1, 0, 0]

    """
    distances = dist.pdist(X, metric=metric)
    distances = dist.squareform(distances)
    np.fill_diagonal(distances, np.inf)
    return distances.argmin(1)


# %%
neighbor_indices = nearest_neighbors(document_term_matrix)


# %%
nn_genres = genres[neighbor_indices]
print(nn_genres[:5])


# %%
overlap = np.sum(genres == nn_genres)
print(f'Maching pairs (normalized): {overlap / len(genres):.2f}')


# %%
print(collections.Counter(nn_genres[genres == 'Tragédie']).most_common())
print(collections.Counter(nn_genres[genres == 'Comédie']).most_common())
print(collections.Counter(nn_genres[genres == 'Tragi-comédie']).most_common())


# %%
t_dists, c_dists = [], []
for tc in document_term_matrix[genres == 'Tragi-comédie']:
    t_dists.append(cosine_distance(tc, tr_means))
    c_dists.append(cosine_distance(tc, co_means))

print(f'Mean distance to comédie vector: {np.mean(c_dists):.3f}')
print(f'Mean distance to tragédie vector: {np.mean(t_dists):.3f}')


# %%
fig, ax = plt.subplots()
ax.boxplot([t_dists, c_dists])
ax.set(xticklabels=('Tragédie', 'Comédie'), ylabel='Distances to genre means');


# %%
t_dists = np.array(t_dists)
outliers = t_dists.argsort()[::-1][:2]


# %%
tc_titles = np.array(titles)[genres == 'Tragi-comédie']
print('\n'.join(tc_titles[outliers]))


# %%
import csv

letters, years = [], []
with open("data/chain-letters.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        letters.append(row["letter"])
        years.append(int(row["year"]))


# %%
import numpy as np


# %%
a = np.array([1.0, 0.5, 0.33, 0.25, 0.2])


# %%
a = np.array([1, 3, 6, 10, 15])


# %%
a = np.array([0, 1, 1, 2, 3, 5], dtype='int32')
print(a.dtype)


# %%
a = a.astype('float32')
print(a.dtype)


# %%
a = np.array([0, 1, 1, 2, 3, 5])
print(a.ndim)


# %%
a = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]])
print(a.ndim)


# %%
a = np.array([[[1, 3, 3], [2, 5, 2]], [[2, 3, 7], [4, 5, 9]]])
print(a.ndim)


# %%
a = np.array([[0, 1, 2, 3], [1, 0, 2, 6], [2, 1, 0, 5]])
print(a.shape)


# %%
print(np.zeros((3, 5)))


# %%
print(np.zeros(10))


# %%
print(np.ones((3, 4), dtype='int64'))


# %%
print(np.empty((3, 2)))


# %%
print(np.random.random_sample(5))


# %%
print(np.random.random_sample((2, 3)))


# %%
a = np.arange(0, 2, 0.25)
print(a)


# %%
a = np.arange(10)
print(a[5])


# %%
print(a[3:8])


# %%
word_index = vocabulary.index('monsieur')
document_term_matrix = np.array(document_term_matrix)
print(document_term_matrix[2, word_index])


# %%
print(document_term_matrix[:5, word_index])


# %%
print(document_term_matrix[5, 10:40])


# %%
column_values = document_term_matrix[:, word_index]


# %%
print(document_term_matrix[5, :])


# %%
print(document_term_matrix[5])


# %%
print(document_term_matrix[(1, 8, 3), :])


# %%
words = 'monsieur', 'madame', 'amour'
word_indexes = [vocabulary.index(word) for word in words]
print(document_term_matrix[:, word_indexes])


# %%
numbers = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]


# %%
print([number * 10 for number in numbers])


# %%
numbers = np.array(numbers)
print(numbers * 10)


# %%
numbers = list(range(1000000))
%timeit [number * 10 for number in numbers]


# %%
numbers = np.arange(1000000)
%timeit numbers * 10


# %%
numbers = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
print([number for number in numbers if number < 10])


# %%
numbers = np.array(numbers)
print(numbers[numbers < 10])


# %%
print(numbers < 10)


# %%
print(document_term_matrix[document_term_matrix[:, vocabulary.index('de')] > 0])


# %%
numbers = np.random.random_sample(100000)
print(sum(numbers))


# %%
print(numbers.sum())  # equivalent to np.sum(numbers)


# %%
%timeit sum(numbers)
%timeit numbers.sum()


# %%
sums = document_term_matrix.sum(axis=1)


# %%
print(document_term_matrix.sum(axis=0))


# %%
print(document_term_matrix.sum())


# %%
a = np.array([1, 2, 3])
b = np.array([2, 4, 6])
print(a * b)


# %%
a = np.array([1, 2, 3])
print(a * 2)


