from string import punctuation
import os
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import re
from neuralNet.dataLoader import DataLoader

class SentimentTrainer:
    def __init__ (self, vocab):
        self.model = None
        self.max_length = 0
        self.tokenizer = Tokenizer()
        self.vocab = vocab
 
    @staticmethod
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
 
    # return a list of cleaned word vectors for all docs in a directory
    @staticmethod
    def process_docs(directory, vocab, is_trian):
        documents = list()
        # walk through all files in given directory
        for filename in os.listdir(directory):
            sent, num = re.match(r'([a-z]+)([0-9]+)', filename).groups()
            num = int(num)
            # process either training docs or testing docs
            if is_trian and ((sent == 'pos' and num >= 1263) or (sent == 'neg' and num >= 504)):
                continue
            if not is_trian and not ((sent == 'pos' and num >= 1263) or (sent == 'neg' and num >= 504)):
                continue
            # create the full path of the file to open
            path = directory + '/' + filename
            # load the doc
            doc = DataLoader.load_doc(path)
            # clean doc
            tokens = SentimentTrainer.clean_doc(doc, vocab)
            # add to list
            documents.append(tokens)
        return documents

    def train_model(self, pos_path, neg_path):
        # load all training docs
        positive_docs = self.process_docs(pos_path, self.vocab, True)
        negative_docs = self.process_docs(neg_path, self.vocab, True)
        
        train_docs = negative_docs + positive_docs
        
        # fit the tokenizer on the documents
        self.tokenizer.fit_on_texts(train_docs)
        
        # convert the words lists into sequence vectors
        encoded_docs = self.tokenizer.texts_to_sequences(train_docs)
        
        # pad sequences
        self.max_length = max([len(s.split()) for s in train_docs])
        Xtrain = pad_sequences(encoded_docs, maxlen=self.max_length, padding='post')
        
        # define training labels
        ytrain = array([0 for _ in range(504)] + [1 for _ in range(1263)])

        #load all testing docs
        positive_docs = self.process_docs(pos_path, self.vocab, False)
        negative_docs = self.process_docs(neg_path, self.vocab, False)
        
        test_docs = negative_docs + positive_docs
        
        # convert the words lists into sequence vectors
        encoded_docs = self.tokenizer.texts_to_sequences(test_docs)
        
        # pad sequences
        Xtest = pad_sequences(encoded_docs, maxlen=self.max_length, padding='post')
        
        # define test labels
        ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])
         
        # define vocabulary size
        vocab_size = len(self.tokenizer.word_index) + 1
         
        # define model
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, 100, input_length=self.max_length))
        self.model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        print(self.model.summary())

        # compile model
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'Precision', 'Recall'])
        
        # fit model
        self.model.fit(Xtrain, ytrain, batch_size = 512, epochs=10)
        
        # evaluate model
        loss, acc, prec, rec = self.model.evaluate(Xtest, ytest)
        f1 = (2 * prec * rec) / (prec + rec)
        print('Test Accuracy: %f' % (acc*100))
        print('Test Precision: %f' % (prec*100))
        print('Test Recall: %f' % (rec*100))
        print('Test F1: %f' % (100*f1))

        # save the model to be used to predict
        filepath = os.path.abspath(os.path.dirname(__file__))
        self.model.save(filepath + '/my_model')