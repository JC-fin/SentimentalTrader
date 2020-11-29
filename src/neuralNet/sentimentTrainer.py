from string import punctuation
import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import numpy as np
import re
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split

class SentimentTrainer:
    def __init__ (self):
        self.model = None
        self.max_length = 0
        self.tokenizer = Tokenizer()
        self.vocab = Counter()
        self.companies = ['nkla', 'nikola', 'msft', 'microsoft', 'aapl', 'apple', 
                    'nflx', 'netflix', 'wday', 'workday', 'nvda', 'nvidia',
                     'nlok', 'norton', 'xrx', 'xerox', 'hpq', 'hp', 'amd', 
                     'amd', 'mrna', 'moderna', 'pton', 'peloton', 'hd', 
                     'home depot']

    def clean_train_data(self, title):
        # make doc lowercase
        title = title.lower()
        
        # split into tokens by white space
        tokens = title.split()
        
        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        
        # remove remaining tokens that are not alphanumeric, stopwords, too short, or company names
        tokens = [word for word in tokens if word.isalpha() 
                        and word not in set(stopwords.words('english'))
                        and len(word) > 1
                        and word not in self.companies]

        return tokens

    def clean_test_data(self, title):
        # make doc lowercase
        title = title.lower()
        
        # split into tokens by white space
        tokens = title.split()
        
        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        
        # remove remaining tokens that are not in vocab
        tokens = [word for word in tokens if word in self.vocab]

        return tokens

    @staticmethod
    def replace_label(row):
        label_dict = {'negative' : 0, 'positive': 1}
        return label_dict[row['label']]
 
    def train_model(self, data_path):
        # load data
        data = pd.read_csv(data_path, encoding='latin-1', names=['label', 'title'])
        
        # replace word labels with numerical labels
        data = data[data['label'] != 'neutral']
        data['label'] = data.apply(self.replace_label, axis = 1)
        
        # split data into train and test sets
        train, test = train_test_split(data, test_size=0.15, stratify = data['label'])

        # clean data and convert into array of words
        train['title'] = train['title'].apply(self.clean_train_data)
        test['title'] = test['title'].apply(self.clean_test_data)

        # create vocab of all words in training set
        train['title'].apply(lambda x: self.vocab.update(x))

        # fit the tokenizer on the training data
        self.tokenizer.fit_on_texts(train['title'])

        # convert the word array into a sequence vector
        train['sequences'] = self.tokenizer.texts_to_sequences(train['title'])
        test['sequences'] = self.tokenizer.texts_to_sequences(test['title'])

        # pad the vectors to fit the correct length
        self.max_length = max([len(s) for s in train['sequences']])
        xtrain = pad_sequences(train['sequences'], maxlen=self.max_length, padding='post')
        xtest = pad_sequences(test['sequences'], maxlen=self.max_length, padding='post')
         
        # define vocabulary size
        vocab_size = len(self.tokenizer.word_index) + 1
         
        # define model
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, 150, input_length=self.max_length))
        self.model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        print(self.model.summary())

        # compile model
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'Precision', 'Recall'])
        
        # fit model
        self.model.fit(xtrain, train['label'], batch_size = 512, epochs=10)
        
        # evaluate model
        loss, acc, prec, rec = self.model.evaluate(xtest, test['label'])
        f1 = (2 * prec * rec) / (prec + rec)
        print('Test Accuracy: %f' % (acc*100))
        print('Test Precision: %f' % (prec*100))
        print('Test Recall: %f' % (rec*100))
        print('Test F1: %f' % (100*f1))

        # save the model to be used to predict
        filepath = os.path.abspath(os.path.dirname(__file__))
        self.model.save(filepath + '/my_model')

        return [acc, prec, rec, f1]