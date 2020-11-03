from string import punctuation
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from os import listdir
import re
from sentimentTrainer import SentimentTrainer
from dataLoader import DataLoader

class SentimentPredictor:
    def __init__(self, vocab):
        self.model = keras.models.load_model('my_model')
        self.tokenizer = Tokenizer()
        self.vocab = vocab
        self.tokenizer.fit_on_texts(SentimentTrainer.process_docs('../../data/finData/neg/', self.vocab, True) + 
            SentimentTrainer.process_docs('../../data/finData/pos/', self.vocab, True))


    def predict(self, row):
        tokens = SentimentTrainer.clean_doc(row['Headline'], self.vocab)
        seqs = list()
        seqs.append(tokens)
        print(row['Ticker'])
        print(seqs)
        seqs = self.tokenizer.texts_to_sequences(seqs)
        print(seqs)
        seqs = pad_sequences(seqs, maxlen=31, padding='post') #maxlen from training data
        return self.model.predict(seqs)

    def predict_sentiment(self, dataframe):
        dataframe['Prediction'] = dataframe.apply(self.predict, axis = 1)
        return dataframe

# # load the vocabulary
# vocab_filename = 'vocab.txt'
# vocab = load_doc(vocab_filename)
# vocab = vocab.split()
# vocab = set(vocab)

# model = keras.models.load_model("my_model")
# # path = 'testPos1.txt'
# # doc = load_doc(path)
# # tokens = clean_doc(doc, vocab)
# # x = list()
# # x.append(tokens)
# # print(x)

# # x = process_docs(data_path+'neg/', vocab, False)

# tokenizer = Tokenizer()

# # fit the tokenizer on the training data
# tokenizer.fit_on_texts(process_docs(data_path+'/finData/neg/', vocab, True) + process_docs(data_path+'/finData/pos/', vocab, True))
 
# # sequence encode
# # x = tokenizer.texts_to_sequences(x)

# # x = pad_sequences(x, maxlen=1317, padding='post')

# # print(model.predict(x))

# df = pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['MCFE'], 'File Name':[data_path+'testData/mcfe.txt']}) # clear positive
# df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['TSLA'], 'File Name':[data_path+'testData/tsla.txt']})) #clear positive
# df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['QUIB'], 'File Name':[data_path+'testData/quib.txt']})) #clear negative
# df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['EQUI'], 'File Name':[data_path+'testData/equi.txt']})) #iffy negative
# df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['XOM'], 'File Name':[data_path+'testData/xom.txt']})) #clear negative
# df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['GM'], 'File Name':[data_path+'testData/gm.txt']})) #clear positive
# df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['ALI'], 'File Name':[data_path+'testData/ali.txt']})) #clear positive
# df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['PHAR'], 'File Name':[data_path+'testData/phar.txt']})) #clear negative
# df['Prediction'] = df.apply(predict, axis = 1)

# print(df)