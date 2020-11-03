from string import punctuation
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from os import listdir

data_path = '../../data/'

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
    # make doc lowercase
    doc = doc.lower()
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
    documents = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_trian and filename.startswith('cv9'):
            continue
        if not is_trian and not filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc(doc, vocab)
        # add to list
        documents.append(tokens)
    return documents

def predict(row):
    doc = load_doc(row['File Name'])
    tokens = clean_doc(doc, vocab)
    seqs = list()
    seqs.append(tokens)
    print(row['Ticker'])
    print(seqs)
    seqs = tokenizer.texts_to_sequences(seqs)
    seqs = pad_sequences(seqs, maxlen=31, padding='post') #maxlen from training data
    return model.predict(seqs)

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

model = keras.models.load_model("my_model")
# path = 'testPos1.txt'
# doc = load_doc(path)
# tokens = clean_doc(doc, vocab)
# x = list()
# x.append(tokens)
# print(x)

# x = process_docs(data_path+'neg/', vocab, False)

tokenizer = Tokenizer()

# fit the tokenizer on the training data
tokenizer.fit_on_texts(process_docs(data_path+'/finData/neg/', vocab, True) + process_docs(data_path+'/finData/pos/', vocab, True))
 
# sequence encode
# x = tokenizer.texts_to_sequences(x)

# x = pad_sequences(x, maxlen=1317, padding='post')

# print(model.predict(x))

df = pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['MCFE'], 'File Name':[data_path+'testData/mcfe.txt']}) # clear positive
df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['TSLA'], 'File Name':[data_path+'testData/tsla.txt']})) #clear positive
df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['QUIB'], 'File Name':[data_path+'testData/quib.txt']})) #clear negative
df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['EQUI'], 'File Name':[data_path+'testData/equi.txt']})) #iffy negative
df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['XOM'], 'File Name':[data_path+'testData/xom.txt']})) #clear negative
df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['GM'], 'File Name':[data_path+'testData/gm.txt']})) #clear positive
df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['ALI'], 'File Name':[data_path+'testData/ali.txt']})) #clear positive
df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['PHAR'], 'File Name':[data_path+'testData/phar.txt']})) #clear negative
df['Prediction'] = df.apply(predict, axis = 1)

print(df)