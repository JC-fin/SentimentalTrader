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
        sent, num = re.match(r'([a-z]+)([0-9]+)', filename).groups()
        num = int(num)
        # skip any reviews in the test set
        if is_trian and ((sent == 'pos' and num >= 1263) or (sent == 'neg' and num >= 504) or (sent == 'neu' and num >= 2779)):
            continue
        if not is_trian and not ((sent == 'pos' and num >= 1263) or (sent == 'neg' and num >= 504) or (sent == 'neu' and num >= 2779)):
            continue
        # # skip any reviews in the test set
        # if is_trian and filename.startswith('cv9'):
        #     continue
        # if not is_trian and not filename.startswith('cv9'):
        #     continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc(doc, vocab)
        # add to list
        documents.append(tokens)
    return documents
 
# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
 
# load all training reviews
positive_docs = process_docs('../../data/finData/pos', vocab, True)
negative_docs = process_docs('../../data/finData/neg', vocab, True)
#neutral_docs = process_docs('../../data/finData/neu', vocab, True)
train_docs = negative_docs + positive_docs# + neutral_docs
 
# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)
 
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define training labels
ytrain = array([0 for _ in range(504)] + [1 for _ in range(1263)])# + [0.5 for _ in range(2779)])
 
# load all test reviews
positive_docs = process_docs('../../data/finData/pos', vocab, False)
negative_docs = process_docs('../../data/finData/neg', vocab, False)
#neutral_docs = process_docs('../../data/finData/neu', vocab, False)
test_docs = negative_docs + positive_docs# + neutral_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])# + [0.5 for _ in range(100)])
 
# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1
 
# define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Conv1D(filters=128, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))

# save the model to be used to predict
model.save('my_model')