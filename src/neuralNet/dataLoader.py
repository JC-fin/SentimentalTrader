from string import punctuation
from os import listdir
from collections import Counter
#import nltk
from nltk.corpus import stopwords
import re
#nltk.download('stopwords')

class DataLoader:
    def __init__(self):
        self.vocab = None
        self.companies = ['nkla', 'nikola', 'msft', 'microsoft', 'aapl', 'apple', 
                    'nflx', 'netflix', 'wday', 'workday', 'nvda', 'nvidia',
                     'nlok', 'norton', 'xrx', 'xerox', 'hpq', 'hp', 'amd', 
                     'amd', 'mrna', 'moderna', 'pton', 'peloton', 'hd', 
                     'home depot']
   
    @staticmethod
    def load_doc(filename):
        file = open(filename, 'r')
        text = file.read()
        file.close()
        return text
     
    def clean_doc(self, doc):
        # make doc lowercase
        doc = doc.lower()
        
        # split into tokens by white space
        tokens = doc.split()
        
        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        
        # remove remaining tokens that are not alphanumeric, stopwords, too short, or company names
        tokens = [word for word in tokens if word.isalpha()
                        and not word in set(stopwords.words('english')) 
                        and len(word) > 1 
                        and not word in self.companies]

        return tokens
     
    def add_doc_to_vocab(self, filename, vocab):
        doc = self.load_doc(filename)
        tokens = self.clean_doc(doc)
        vocab.update(tokens)
     
    # load all docs in a directory
    def process_docs(self, directory, vocab, is_trian):
        # walk through all files in given directory
        for filename in listdir(directory):
            sent, num = re.match(r'([a-z]+)([0-9]+)', filename).groups()
            num = int(num)
            # process either training docs or testing docs
            if is_trian and ((sent == 'pos' and num >= 1263) or (sent == 'neg' and num >= 504)):
                continue
            if not is_trian and not ((sent == 'pos' and num >= 1263) or (sent == 'neg' and num >= 504)):
                continue
            # create the full path of the file to open
            path = directory + '/' + filename
            # add doc to vocab
            self.add_doc_to_vocab(path, vocab)

    def load_vocab(self, neg_path, pos_path):
        self.vocab = Counter()
        
        # add training docs to vocab
        self.process_docs(neg_path, self.vocab, True)
        self.process_docs(pos_path, self.vocab, True)

        file = open('vocab.txt', 'w')
        # write all words in vocab as one word per line
        file.write('\n'.join(self.vocab))
        file.close()
