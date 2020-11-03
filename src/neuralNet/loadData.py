from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
import re
 
# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text
 
# turn a doc into clean tokens
def clean_doc(doc):
    # make doc lowercase
    doc = doc.lower()
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens
 
# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
    # load doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # update counts
    vocab.update(tokens)
 
# load all docs in a directory
def process_docs(directory, vocab, is_trian):
    # walk through all files in the folder
    for filename in listdir(directory):
        sent, num = re.match(r'([a-z]+)([0-9]+)', filename).groups()
        num = int(num)
        # skip any reviews in the test set
        if is_trian and ((sent == 'pos' and num >= 1263) or (sent == 'neg' and num >= 504) or (sent == 'neu' and num >= 2779)):
            continue
        if not is_trian and not ((sent == 'pos' and num >= 1263) or (sent == 'neg' and num >= 504) or (sent == 'neu' and num >= 2779)):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # add doc to vocab
        add_doc_to_vocab(path, vocab)

# save list to file
def save_list(lines, filename):
    # convert lines to a single blob of text
    data = '\n'.join(lines)
    # open file
    file = open(filename, 'w')
    # write text
    file.write(data)
    # close file
    file.close()
 
# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('../../data/finData/neg', vocab, True)
process_docs('../../data/finData/pos', vocab, True)
# process_docs('../../data/finData/neu', vocab, True)
# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))

# keep tokens with a min occurrence
min_occurance = 1
tokens = [k for k,c in vocab.items() if c >= min_occurance]
print(len(tokens))
 
# save tokens to a vocabulary file
save_list(tokens, 'vocab.txt')