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

class SentimentPredictor:
    def __init__(self, sentTrainer):
        self.st = sentTrainer

    def predict(self, row):
        tokens = self.st.clean_test_data(row['Headline'])
        seqs = list()
        seqs.append(tokens)
        seqs = self.st.tokenizer.texts_to_sequences(seqs)
        if len(seqs[0]) < 3:
            return np.nan
        seqs = pad_sequences(seqs, maxlen=self.st.max_length, padding='post')
        return self.st.model.predict(seqs)

    def predict_sentiment(self, dataframe):
        dataframe['Prediction'] = dataframe.apply(self.predict, axis = 1)
        return dataframe