from string import punctuation
from os import listdir
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
from dataLoader import DataLoader

class SentimentTrainer:
    def __init__ (self, vocab):
        self.model = None
        self.max_length = 0
        self.tokenizer = Tokenizer()
        self.vocab = vocab
 
    # turn a doc into clean tokens
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
 
    # load all docs in a directory
    @staticmethod
    def process_docs(directory, vocab, is_trian):
        documents = list()
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
            # load the doc
            doc = DataLoader.load_doc(path)
            # clean doc
            tokens = SentimentTrainer.clean_doc(doc, vocab)
            # add to list
            documents.append(tokens)
        return documents

    def train_model(self, pos_path, neg_path):
        # load all training reviews
        positive_docs = self.process_docs(pos_path, self.vocab, True)
        negative_docs = self.process_docs(neg_path, self.vocab, True)
        #neutral_docs = process_docs('../../data/finData/neu', vocab, True)
        
        train_docs = negative_docs + positive_docs# + neutral_docs
        
        # fit the tokenizer on the documents
        self.tokenizer.fit_on_texts(train_docs)
        
        # sequence encode
        encoded_docs = self.tokenizer.texts_to_sequences(train_docs)
        
        # pad sequences
        self.max_length = max([len(s.split()) for s in train_docs]) #TODO: test with arbitrary max length (possibly 50?)
        print('maxLength: ' + str(self.max_length))
        Xtrain = pad_sequences(encoded_docs, maxlen=self.max_length, padding='post')
        
        # define training labels
        ytrain = array([0 for _ in range(504)] + [1 for _ in range(1263)])# + [0.5 for _ in range(2779)])

        #load all test reviews
        positive_docs = self.process_docs('../../data/finData/pos', self.vocab, False)
        negative_docs = self.process_docs('../../data/finData/neg', self.vocab, False)
        #neutral_docs = process_docs('../../data/finData/neu', vocab, False)
        
        test_docs = negative_docs + positive_docs# + neutral_docs
        
        # sequence encode
        encoded_docs = self.tokenizer.texts_to_sequences(test_docs)
        
        # pad sequences
        Xtest = pad_sequences(encoded_docs, maxlen=self.max_length, padding='post')
        
        # define test labels
        ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])# + [0.5 for _ in range(100)])
         
        # define vocabulary size (largest integer value)
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

        # compile network
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'Precision'])
        # fit network
        self.model.fit(Xtrain, ytrain, epochs=50, batch_size=50, verbose=2)
        # evaluate
        loss, acc, prec = self.model.evaluate(Xtest, ytest, verbose=0)
        print('Test Accuracy: %f' % (acc*100))
        print('Precision: %f' % (prec*100))

        # save the model to be used to predict
        self.model.save('my_model')