
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from gensim import models
import numpy as np
import re
import pickle
from text_utils import *

#INTERFACE FOR PREPROCESSORS AS WELL AS SOME STANDARD IMPLEMENTATIONS


#Utility function for adding padding so sequences are all the same length
#clean_text - list of documents
#maxlen - the max length of each document; longer documents will be truncated to thes length
# and shorter documents will be padded
#value - the value that is used to fill in the empty space
def pad_sequences(clean_text, maxlen, value):

    X = np.ones( (len(clean_text) , maxlen)) * value

    for i,doc in enumerate(clean_text):
        l = len(doc)
        if l > maxlen:
            X[i] = np.array(doc[0:maxlen])
        else:
            #X[i,0:len(doc)] = np.array(doc)
            X[i, maxlen - len(doc):] = np.array(doc)

    return X

#Interface for creating a preprocessor
#fit - do any necessary setup for the preprocessor (pick vocab, filter words , ect)
#transform - raw text --> numpy array

class PreProcessor(object):

    #fit the preprocessor -> fit vocabulary and embedding
    #text - list of strings
    def fit(self,text):
        pass

    #convert text to numerical representation
    #text - list of strings
    def transform(self,text):
        pass

    #save preprocessor for later in the ./data directory
    #fn - file to save to
    def save(self,fn):
        pass

    #load preprocessor from ./data directory
    #fn - file to load
    def load(self,fn):
        pass

#Standard Bag of Words representation
class BowPreProcessor(PreProcessor):

    def __init__(self,args):
        self.vectorizer = []
        self.args = args

    def fit(self, raw_text):

        print 'Fitting Vectorizer'
        if self.args.n == None:
            # create the initial vocabulary from the extracted text
            vectorizer = CountVectorizer(analyzer="word", stop_words='english')
        else:
            # create the initial vocabulary from the extracted text
            vectorizer = CountVectorizer(analyzer="word", stop_words='english', max_features=self.args.n)

        vectorizer.fit(raw_text)

        if self.args.word2vec:

            inital_vocab = vectorizer.vocabulary_

            print 'Building Embedding'
            # print inital_vocab
            # Get the embedded form of the vocabulary (some of the words in our corpus might not be in word2vec)
            w2v = models.Word2Vec.load_word2vec_format('/Users/Hadoop/Desktop/glove-to-word2vec/glove_model2.txt', binary=True)
            new_voc = down_size_word2vec(inital_vocab, w2v)

            print 'Finalizing Vectorizer'
            vectorizer2 = CountVectorizer(analyzer="word", stop_words='english', max_features=None, vocabulary=new_voc)

            self.embedding_mat = w2v_matrix(new_voc, w2v)
            self.vectorizer = vectorizer2

            print self.embedding_mat.shape

        else:

            self.embedding_mat = .1*np.random.randn(self.args.n,self.args.embedSize)
            self.vectorizer = vectorizer

    def transform(self,X,args=None):
        bow=  self.vectorizer.transform(X)
        return bow

    def save(self,fn):
        pickle.dump(self.vectorizer, open('./data/'+fn , 'wb'))

    def load(self,fn):
        self.vectorizer = pickle.load( open('./data/'+fn , 'rb'))

#Bag of Words That is combined with a word2vec word embedding
class BowW2VPreProcessor(PreProcessor):

    def __init__(self,args):
        self.args = args
        self.vectorizer = []
        self.embedding_mat = []
        self.stop_words ='english'

    def fit(self, raw_text):

        args = self.args

        print 'Fitting Vectorizer'
        if args.n == None:
            # create the initial vocabulary from the extracted text
            vectorizer = CountVectorizer(analyzer="word", stop_words=self.stop_words)
        else:
            # create the initial vocabulary from the extracted text
            vectorizer = CountVectorizer(analyzer="word", stop_words=self.stop_words, max_features=args.n)

        vectorizer.fit(raw_text)
        inital_vocab = vectorizer.vocabulary_

        print 'Building Embedding'
        # print inital_vocab
        # Get the embedded form of the vocabulary (some of the words in our corpus might not be in word2vec)
        w2v = models.Word2Vec.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
        new_voc = down_size_word2vec(inital_vocab, w2v)

        print 'Finalizing Vectorizer'
        vectorizer2 = CountVectorizer(analyzer="word", stop_words='english', max_features=None, vocabulary=new_voc)

        self.embedding_mat = w2v_matrix(new_voc, w2v)


        self.vectorizer = vectorizer2

    def transform(self, X):

        return self.vectorizer.transform(X)

    def save(self, fn):
        pickle.dump(self.vectorizer, open('./data/' + fn, 'wb'))
        #pickle.dump(self.embedding_mat, open('./data/' + 'embedding.dat', 'wb'))

    def load(self, fn):
        self.vectorizer = pickle.load(open('./data/' + fn, 'rb'))
        #self.embedding_mat = pickle.load(open('./data/' + 'embedding.dat', 'rb'))

#Convert text into a sequence of characters
class CharPreProcessor(PreProcessor):

    def fit(self,text):
        pass

    def transform(self,text,args=None):

        chars = list("""qwertyuiopasdfghjklzxcvbnm1234567890 \n\t""")
        chars = chars + [u'\x0c', u'\r']
        indexs = range(len(chars))

        char_dict = dict(zip(chars, indexs))

        clean_text = [char2index(x, char_dict) for x in text]

        # convert to numpy array for efficent storage
        X = pad_sequences(clean_text, maxlen=args.max_len, value=char_dict[' '])

        return X

    def save(self,fn):
        pass

    def load(self,fn):
        pass

#Converts text to a sequence of word indexes
#like BOW but sparse
#turns text into array of shape (num points, max seq len)
class WordIndexPreProcessor(PreProcessor):

    def __init__(self,args):
        self.vect = BowPreProcessor(args)
        self.args= args

    def fit(self, text):
        self.vect.fit(text)

    def transform(self, text):

        bow = self.vect.transform(text)

        X = []
        for i in range(len(text)):
            nz = np.nonzero(bow[i])
            X.append(nz[1].tolist())

        # convert to numpy array for efficent storage
        X = pad_sequences(X, maxlen=self.args.max_len, value=-1)
        X = X +2
        return X

    def save(self, fn):
        self.vect.save(fn)

    def load(self, fn):
        self.vect.load(fn)

class WordSequencePreProcessor(PreProcessor):

    def __init__(self,args):
        self.vect = BowPreProcessor(args)
        self.args= args

    def fit(self, text):
        self.vect.fit(text)

    def transform(self, text):

        vocab = self.vect.vectorizer.vocabulary_

        X = []
        lengths = []
        for i in range(len(text)):
            index = []
            doc = text[i]
            for word in doc.split(' '):
                if word in vocab:
                    index.append(vocab[word])
            X.append(index)
            lengths.append(len(index))

        print 'Average Seq Leng: ' , np.array(lengths).mean()

        # convert to numpy array for efficent storage
        X = pad_sequences(X, maxlen=self.args.max_len, value=-1)
        X = X +2

        return X

    def save(self, fn):
        self.vect.save(fn)
        pickle.dump(self.args.max_len,open(self.args.model+'_max_len.p','wb'))

    def load(self, fn):
        self.vect.load(fn)
        self.args.max_len = pickle.load(open(self.args.model + '_max_len.p', 'rb'))
