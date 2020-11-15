from string import punctuation
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from os import listdir
import os
import re
from neuralNet.sentimentTrainer import SentimentTrainer
from neuralNet.dataLoader import DataLoader

class SentimentPredictor:
    def __init__(self, vocab):
        filepath = os.path.abspath(os.path.dirname(__file__))
        self.model = keras.models.load_model(filepath + '/my_model')
        self.tokenizer = Tokenizer()
        self.vocab = vocab
        self.tokenizer.fit_on_texts(SentimentTrainer.process_docs(filepath + '/../../data/finData/neg/', self.vocab, True) + 
            SentimentTrainer.process_docs(filepath + '/../../data/finData/pos/', self.vocab, True))


    def predict(self, row):
        tokens = SentimentTrainer.clean_doc(row['Headline'], self.vocab)
        seqs = list()
        seqs.append(tokens)
        seqs = self.tokenizer.texts_to_sequences(seqs)
        if len(seqs[0]) < 3:
            return np.nan
        seqs = pad_sequences(seqs, maxlen=31, padding='post') #maxlen from training data
        return self.model.predict(seqs)

    def predict_sentiment(self, dataframe):
        dataframe['Prediction'] = dataframe.apply(self.predict, axis = 1)
        return dataframe