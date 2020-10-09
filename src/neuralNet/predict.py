from string import punctuation
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np


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
def clean_doc(doc, vocab):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

model = keras.models.load_model("my_model")
path = ("testPos2.txt")
doc = load_doc(path)
tokens = clean_doc(doc, vocab)
x = list()
x.append(tokens)
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(x)
 
# sequence encode
x = tokenizer.texts_to_sequences(x)
x = pad_sequences(x, maxlen=1317, padding='post')
print(model.predict(x))