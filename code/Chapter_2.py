"""
Humanities Data Analysis: Case studies with Python
--------------------------------------------------
Folgert Karsdorp, Mike Kestemont & Allen Riddell
Chapter 2: Parsing and Manipulating Structured Data
"""


# %%
import tarfile
tf = tarfile.open('data/folger.tar.gz', 'r')
tf.extractall('data')


# %%
file_path = 'data/folger/txt/1H4.txt'
stream = open(file_path)
contents = stream.read()
stream.close()

print(contents[:300])


# %%
with open(file_path) as stream:
    contents = stream.read()

print(contents[:300])


# %%
with open('data/anna-karenina.txt', encoding='koi8-r') as stream:
    # Use stream.readline() to retrieve the next line from a file,
    # in this case the 1st one:
    line = stream.readline()

print(line)


# %%
csv_file = 'data/folger_shakespeare_collection.csv'
with open(csv_file) as stream:
    # call stream.readlines() to read all lines in the CSV file as a list.
    lines = stream.readlines()

print(lines[:3])


# %%
entries = []
for line in open(csv_file):
    entries.append(line.strip().split(','))

for entry in entries[:3]:
    print(entry)


# %%
import csv

entries = []
with open(csv_file) as stream:
    reader = csv.reader(stream, delimiter=',')
    for fname, author, title, editor, publisher, pubplace, date in reader:
        entries.append((fname, title))

for entry in entries[:5]:
    print(entry)


# %%
entries = []
with open(csv_file) as stream:
    reader = csv.reader(stream, delimiter=',')
    for fname, _, title, *_ in reader:
        entries.append((fname, title))

for entry in entries[:5]:
    print(entry)


# %%
a, _, c, _, _ = range(5)
print(a, c)


# %%
a, *l = range(5)
print(a, l)


# %%
seq = range(5)
a, l = seq[0], seq[1:]
print(a, l)


# %%
a, *l, b = range(5)
print(a, l, b)


# %%
entries = []

with open(csv_file) as stream:
    reader = csv.DictReader(stream, delimiter=',')
    for row in reader:
        entries.append(row)

for entry in entries[:5]:
    print(entry['fname'], entry['title'])


# %%
import PyPDF2 as PDF


# %%
file_path = 'data/folger/pdf/1H4.pdf'
pdf = PDF.PdfFileReader(file_path, overwriteWarnings=False)


# %%
n_pages = pdf.getNumPages()
print(f'PDF has {n_pages} pages.')


# %%
page = pdf.getPage(1)
content = page.extractText()
print(content[:150])


# %%
def pdf2txt(fname, page_numbers=None, concatenate=False):
    """Convert text from a PDF file into a string or list of strings.

    Arguments:
        fname: a string pointing to the filename of the PDF file
        page_numbers: an integer or sequence of integers pointing to the
            pages to extract. If None (default), all pages are extracted.
        concatenate: a boolean indicating whether to concatenate the
            extracted pages into a single string. When False, a list of
            strings is returned.

    Returns:
        A string or list of strings representing the text extracted
        from the supplied PDF file.

    """
    pdf = PDF.PdfFileReader(fname, overwriteWarnings=False)
    if page_numbers is None:
        page_numbers = range(pdf.getNumPages())
    elif isinstance(page_numbers, int):
        page_numbers = [page_numbers]
    texts = [pdf.getPage(n).extractText() for n in page_numbers]
    return '\n'.join(texts) if concatenate else texts


# %%
text = pdf2txt(file_path, concatenate=True)
sample = pdf2txt(file_path, page_numbers=[1, 4, 9])


# %%
import json

line = {
    'line_id': 12664,
    'play_name': 'Alls well that ends well',
    'speech_number': 1,
    'line_number': '1.1.1',
    'speaker': 'COUNTESS',
    'text_entry': 'In delivering my son from me, I bury a second husband.'
}

print(json.dumps(line))


# %%
with open('shakespeare.json', 'w') as f:
    json.dump(line, f)


# %%
with open('data/macbeth.json') as f:
    data = json.load(f)

print(data[3:5])


# %%
import collections

languages = collections.Counter()
for entry in data:
    languages[entry['lang']] += 1

print(languages.most_common())


# %%
with open('data/sonnets/18.xml') as stream:
    xml = stream.read()

print(xml)


# %%
import lxml.etree


# %%
tree = lxml.etree.parse('data/sonnets/18.xml')
print(tree)


# %%
# decoding is needed to transform the bytes object into an actual string
print(lxml.etree.tostring(tree).decode())


# %%
for rhyme in tree.iterfind('//rhyme'):
    print(f'element: {rhyme.tag} -> {rhyme.text}')


# %%
root = tree.getroot()
print(root.tag)


# %%
print(root.attrib['year'])


# %%
print(len(root))


# %%
children = [child.tag for child in root]


# %%
print('\n'.join(child.text or '' for child in root))


# %%
print(''.join(root[0].itertext()))


# %%
for node in root:
    if node.tag == 'line':
        print(f"line {node.attrib['n']: >2}: {''.join(node.itertext())}")


# %%
with open('data/sonnets/116.txt') as stream:
    text = stream.read()

print(text)


# %%
root = lxml.etree.Element('sonnet')
root.attrib['author'] = 'William Shakespeare'
root.attrib['year'] = '1609'


# %%
tree = lxml.etree.ElementTree(root)
stringified = lxml.etree.tostring(tree)
print(stringified)


# %%
print(type(stringified))


# %%
print(stringified.decode('utf-8'))


# %%
for nb, line in enumerate(open('data/sonnets/116.txt')):
    node = lxml.etree.Element('line')
    node.attrib['n'] = str(nb + 1)
    node.text = line.strip()
    root.append(node)
    # voltas typically, but not always occur between the octave and sextet
    if nb == 8:
        node = lxml.etree.Element('volta')
        root.append(node)


# %%
print(lxml.etree.tostring(tree, pretty_print=True).decode())


# %%
# Loop over all nodes in the tree
for node in root:
    # Leave the volta node alone. A continue statement instructs
    # Python to move on to the next item in the loop.
    if node.tag == 'volta':
        continue
    # We chop off and store verse-final punctuation:
    punctuation = ''
    if node.text[-1] in ',:;.':
        punctuation = node.text[-1]
        node.text = node.text[:-1]
    # Make a list of words using the split method
    words = node.text.split()
    # We split rhyme words and other words:
    other_words, rhyme = words[:-1], words[-1]
    # Replace the node's text with all text except the rhyme word
    node.text = ' '.join(other_words) + ' '
    # We create the rhyme element, with punctuation (if any) in its tail
    elt = lxml.etree.Element('rhyme')
    elt.text = rhyme
    elt.tail = punctuation
    # We add the rhyme to the line:
    node.append(elt)

tree = lxml.etree.ElementTree(root)
print(lxml.etree.tostring(tree, pretty_print=True).decode())


# %%
with open('data/sonnets/116.xml', 'w') as f:
    f.write(
        lxml.etree.tostring(
            root, xml_declaration=True, pretty_print=True, encoding='utf-8').decode())


# %%
root = lxml.etree.Element('sonnet')
# Add an author attribute to the root node
root.attrib['author'] = 'William Shakespeare'
# Add a year attribute to the root node
root.attrib['year'] = '1609'

for nb, line in enumerate(open('data/sonnets/116.txt')):
    line_node = lxml.etree.Element('line')
    # Add a line number attribute to each line node
    line_node.attrib['n'] = str(nb + 1)

    # Make different nodes for words and non-words
    word = ''
    for char in line.strip():
        if char.isalpha():
            word += char
        else:
            word_node = lxml.etree.Element('w')
            word_node.text = word
            line_node.append(word_node)
            word = ''

            char_node = lxml.etree.Element('c')
            char_node.text = char
            line_node.append(char_node)

    # don't forget last word:
    if word:
        word_node = lxml.etree.Element('w')
        word_node.text = word
        line_node.append(word_node)

    rhyme_node = lxml.etree.Element('rhyme')
    # We use xpath to find the final w-element in the line
    # and wrap it in a line element
    rhyme_node.append(line_node.xpath('//w')[-1])
    line_node.replace(line_node.xpath('//w')[-1], rhyme_node)

    root.append(line_node)

    # Add the volta node
    if nb == 8:
        node = lxml.etree.Element('volta')
        root.append(node)

tree = lxml.etree.ElementTree(root)
xml_string = lxml.etree.tostring(tree, pretty_print=True).decode()
# Print a snippet of the tree:
print(xml_string[:xml_string.find("</line>") + 8] + '  ...')


# %%
tree = lxml.etree.parse('data/folger/xml/Oth.xml')
print(tree.getroot().find('.//{http://www.tei-c.org/ns/1.0}title').text)


# %%
print(tree.getroot().find('title'))


# %%
NSMAP = {'tei': 'http://www.tei-c.org/ns/1.0'}
print(tree.getroot().find('.//tei:title', namespaces=NSMAP).text)


# %%
import bs4 as bs

html_doc = """
<html>
  <head>
    <title>Henry IV, Part I</title>
  </head>
  <body>
    <div>
      <p class="speaker">KING</p>
      <p id="line-1.1.1">
        <a id="ftln-0001">FTLN 0001</a>
        So shaken as we are, so wan with care,
      </p>
      <p id="line-1.1.2">
        <a id="ftln-0002">FTLN 0002</a>
        Find we a time for frighted peace to pant
      </p>
      <p id="line-1.1.3">
        <a id="ftln-0003">FTLN 0003</a>
        And breathe short-winded accents of new broils
      </p>
      <p id="line-1.1.4">
        <a id="ftln-0004">FTLN 0004</a>
        To be commenced in strands afar remote.
      </p>
    </div>
  </body>
</html>
"""

html = bs.BeautifulSoup(html_doc, 'html.parser')


# %%
# print the documents <title> (from head)
print(html.title)


# %%
# print the first <p> element and its content
print(html.p)


# %%
# print the text of a particular element, e.g. the <title>
print(html.title.text)


# %%
# print the parent tag (and its content) of the first <p> element
print(html.p.parent)


# %%
# print the parent tag name of the first <p> element
print(html.p.parent.name)


# %%
# find all occurrences of the <a> element
print(html.find_all('a'))


# %%
# find a <p> element with a specific ID
print(html.find('p', id='line-1.1.3'))


# %%
def html2txt(fpath):
    """Convert text from a HTML file into a string.

    Arguments:
        fpath: a string pointing to the filename of the HTML file

    Returns:
        A string representing the text extracted from the supplied
        HTML file.

    """
    with open(fpath) as f:
        html = bs.BeautifulSoup(f, 'html.parser')
    return html.get_text()


# %%
fp = 'data/folger/html/1H4.html'
text = html2txt(fp)
start = text.find('Henry V')
print(text[start:start + 500])


# %%
with open(fp) as f:
    html = bs.BeautifulSoup(f, 'html.parser')
toc = html.find('table', attrs={'class': 'contents'})


# %%
def toc_hrefs(html):
    """Return a list of hrefs from a document's table of contents."""
    toc = html.find('table', attrs={'class': 'contents'})
    hrefs = []
    for tr in toc.find_all('tr'):
        for td in tr.find_all('td'):
            for a in td.find_all('a'):
                hrefs.append(a.get('href'))
    return hrefs


# %%
items = toc_hrefs(html)
print(items[:5])


# %%
def get_href_div(html, href):
    """Retrieve the <div> element corresponding to the given href."""
    href = href.lstrip('#')
    div = html.find('div', attrs={'id': href})
    if div is None:
        div = html.find('a', attrs={'name': href}).findNext('div')
    return div


# %%
def html2txt(fname, concatenate=False):
    """Convert text from a HTML file into a string or sequence of strings.

    Arguments:
        fpath: a string pointing to the filename of the HTML file.
        concatenate: a boolean indicating whether to concatenate the
            extracted texts into a single string. If False, a list of
            strings representing the individual sections is returned.

    Returns:
        A string or list of strings representing the text extracted
        from the supplied HTML file.

    """
    with open(fname) as f:
        html = bs.BeautifulSoup(f, 'html.parser')
    # Use a concise list comprehension to create the list of texts.
    # The same list could be constructed using an ordinary for-loop:
    #    texts = []
    #    for href in toc_hrefs(html):
    #        text = get_href_div(html, href).get_text()
    #        texts.append(text)
    texts = [get_href_div(html, href).get_text() for href in toc_hrefs(html)]
    return '\n'.join(texts) if concatenate else texts


# %%
texts = html2txt(fp)
print(texts[6][:200])


# %%
import urllib.request

page = urllib.request.urlopen('https://en.wikipedia.org/wiki/William_Shakespeare')
html = page.read()


# %%
import bs4

soup = bs4.BeautifulSoup(html, 'html.parser')
print(soup.get_text().strip()[:300])


# %%
import re

for script in soup(['script', 'style']):
    script.extract()
text = soup.get_text()
text = re.sub('\s*\n+\s*', '\n', text)  # remove multiple linebreaks:
print(text[:300])


# %%
links = soup.find_all('a')
print(links[9].prettify())


# %%
V = {1, 2, 3, 4, 5}
E = {(1, 2), (1, 4), (2, 5), (3, 4), (4, 5)}


# %%
import networkx as nx

G = nx.Graph()
G.add_nodes_from(V)
G.add_edges_from(E)


# %%
import matplotlib.pyplot as plt

nx.draw_networkx(G, font_color="white")
plt.axis('off');


# %%
NSMAP = {'tei': 'http://www.tei-c.org/ns/1.0'}


def character_network(tree):
    """Construct a character interaction network.

    Construct a character interaction network for Shakespeare texts in
    the Folger Digital Text collection. Character interaction networks
    are constructed on the basis of successive speaker turns in the texts,
    and edges between speakers are created when their utterances follow
    one another.

    Arguments:
        tree: An lxml.ElementTree instance representing one of the XML
            files in the Folger Shakespeare collection.

    Returns:
        A character interaction network represented as a weighted,
        undirected NetworkX Graph.

    """
    G = nx.Graph()
    # extract a list of speaker turns for each scene in a play
    for scene in tree.iterfind('.//tei:div2[@type="scene"]', NSMAP):
        speakers = scene.findall('.//tei:sp', NSMAP)
        # iterate over the sequence of speaker turns...
        for i in range(len(speakers) - 1):
            # ... and extract pairs of adjacent speakers
            try:
                speaker_i = speakers[i].attrib['who'].split('_')[0].replace('#', '')
                speaker_j = speakers[i + 1].attrib['who'].split('_')[0].replace('#', '')
                # if the interaction between two speakers has already
                # been attested, update their interaction count
                if G.has_edge(speaker_i, speaker_j):
                    G[speaker_i][speaker_j]['weight'] += 1
                # else add an edge between speaker i and j to the graph
                else:
                    G.add_edge(speaker_i, speaker_j, weight=1)
            except KeyError:
                continue
    return G


# %%
tree = lxml.etree.parse('data/folger/xml/Ham.xml')
G = character_network(tree.getroot())


# %%
print(f"N nodes = {G.number_of_nodes()}, N edges = {G.number_of_edges()}")


# %%
import collections

interactions = collections.Counter()

for speaker_i, speaker_j, data in G.edges(data=True):
    interaction_count = data['weight']
    interactions[speaker_i] += interaction_count
    interactions[speaker_j] += interaction_count

nodesizes = [interactions[speaker] * 5 for speaker in G]


# %%
# Create an empty figure of size 15x15
fig = plt.figure(figsize=(15, 15))
# Compute the positions of the nodes using the spring layout algorithm
pos = nx.spring_layout(G, k=0.5, iterations=200)
# Then, add the edges to the visualization
nx.draw_networkx_edges(G, pos, alpha=0.4)
# Subsequently, add the weighted nodes to the visualization
nx.draw_networkx_nodes(G, pos, node_size=nodesizes, alpha=0.4)
# Finally, add the labels (i.e. the speaker IDs) to the visualization
nx.draw_networkx_labels(G, pos, fontsize=14)
plt.axis('off');


# %%
from copy import deepcopy
G0 = deepcopy(G)

for u, v, d in G0.edges(data=True):
    d['weight'] = 1

nodesizes = [interactions[speaker] * 5 for speaker in G0]

fig = plt.figure(figsize=(15, 15))
pos = nx.spring_layout(G0, k=0.5, iterations=200)
nx.draw_networkx_edges(G0, pos, alpha=0.4)
nx.draw_networkx_nodes(G0, pos, node_size=nodesizes, alpha=0.4)
nx.draw_networkx_labels(G0, pos, fontsize=14)
plt.axis('off');


# %%
G0.remove_node('Hamlet')


# %%
fig = plt.figure(figsize=(15, 15))
pos = nx.spring_layout(G0, k=0.5, iterations=200)
nodesizes = [interactions[speaker] * 5 for speaker in G0]
nx.draw_networkx_edges(G0, pos, alpha=0.4)
nx.draw_networkx_nodes(G0, pos, node_size=nodesizes, alpha=0.4)
nx.draw_networkx_labels(G0, pos, fontsize=14)
plt.axis('off');


# %%
import json
from networkx.readwrite import json_graph

with open('hamlet.json', 'w') as f:
    json.dump(json_graph.node_link_data(G), f)

with open('hamlet.json') as f:
    d = json.load(f)

G = json_graph.node_link_graph(d)
print(f"Graph with {len(G.nodes())} nodes and {len(G.edges())} edges.")


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
import functools
from copy import deepcopy
G1 = deepcopy(G)

for u, v, d in G.edges(data=True):
    if d["weight"] < 10:
        G1.remove_edge(u, v)

G1 = nx.relabel_nodes(G1, {"SOLDIERS.FORTINBRAS.Captain": "Fortinbras.Captain"})
# rename verbose name for Fortinbras' Captain
#SOLDIERS.FORTINBRAS.Captain

subgraphs = [G1.subgraph(c).copy() for c in nx.connected_components(G1)]
# functools.reduce is similar to foldl in Haskell and fold_left in OCaml
def larger_graph(graph1, graph2):
    return graph2 if len(graph2.nodes()) > len(graph1.nodes()) else graph1
G1 = functools.reduce(larger_graph, subgraphs, subgraphs[0])

fig = plt.figure(figsize=(9, 6))
pos = nx.spring_layout(G1, k=0.5, iterations=2000, seed=1)
nx.draw_networkx_edges(G1, pos, alpha=0.4)
nx.draw_networkx_nodes(G1, pos, node_size=[degree * 100 for _, degree in G1.degree()], alpha=0.4)
nx.draw_networkx_labels(G1, pos, fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig('img/hamlet-minimum-10-interactions.png')


