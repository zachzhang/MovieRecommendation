import tensorflow as tf
import sys
import numpy as np
import pandas as pd
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import tflearn
import argparse
from preprocessors import WordSequencePreProcessor
from tflearn.layers.embedding_ops import embedding

class HybridCollabFilter():
    def __init__(self, numUsers, embedding_dim, input_dim,seq_len,word_embed,embed_mat, img_feats,reg_l = .001):

        # hyper parameters
        self.batch_size = 512
        self.numUsers = numUsers
        self.epochs = 50
        self.init_var = .01
        self.l = .001
        self.h = 256
        self.h_dense = 512
        self.word_embed = word_embed
        self.n_dense_feat = 10
        self.l = reg_l

        # Movie Features
        self.movieFeatures = tf.placeholder(tf.float32, shape=(None, input_dim))
        self.imgFeatures = tf.placeholder(tf.float32, shape=(None, img_feats))
        self.lstmFeatures = tf.placeholder(tf.int32, shape=(None, seq_len))

        # input tensors for movies, usres, ratings
        self.users = tf.placeholder(tf.int32, shape=(None))
        self.rating = tf.placeholder(tf.float32, shape=(None))

        # embedding matricies for users
        self.userMat = tf.Variable(self.init_var * tf.random_normal([numUsers, embedding_dim]))
        self.userBias = tf.Variable(self.init_var * tf.random_normal([numUsers, ]))


        #LSTM for plot features
        self.E = tf.Variable(embed_mat.astype(np.float32))

        movieTensor = tf.nn.embedding_lookup(self.E, self.lstmFeatures)
        movieTensor = tf.cast(movieTensor, tf.float32)
        movieTensor = tf.unpack(movieTensor,axis=1)

        movieTensor = tflearn.lstm(movieTensor, self.h)
        movieTensor = tflearn.dropout(movieTensor, 0.8)
        movieTensor = tflearn.fully_connected(movieTensor, embedding_dim - self.n_dense_feat, activation='linear')

        # LInear Model for person fetaures
        self.W = tf.Variable(self.init_var * tf.random_normal([input_dim, self.n_dense_feat]))
        self.b = tf.Variable(self.init_var * tf.random_normal([self.n_dense_feat]))

        denseTensor = tf.matmul(self.movieFeatures,self.W) + self.b

        #Deep Model for Imag Features

        imgTensor = tflearn.fully_connected(self.imgFeatures, self.h_dense, activation='relu')
        imgTensor = tflearn.dropout(imgTensor, 0.8)
        imgTensor = tflearn.fully_connected(imgTensor, 10, activation='linear')



        featTensor = tf.concat(1,[movieTensor,denseTensor,imgTensor])

        # map each user/movie to its feature vector
        self.U = tf.nn.embedding_lookup(self.userMat, self.users)
        self.u_b = tf.nn.embedding_lookup(self.userBias, self.users)

        # predicted rating is dot product of user and movie
        self.yhat = tf.reduce_sum(tf.mul(self.U, featTensor), 1) + self.u_b

        self.cost = tf.nn.l2_loss(self.yhat - self.rating)+ \
                    tf.reduce_mean(self.l * tf.abs( self.W )) + tf.reduce_mean(self.l *tf.abs( self.b ))


        self.optimizer = tf.train.AdamOptimizer(0.005)

        tvars = tf.trainable_variables()

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 1)

        self.optimizer = self.optimizer.apply_gradients(zip(grads, tvars))

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

    def train_test_split(self, users, movies, ratings, split=.1):

        shuffle = np.random.permutation(len(users))

        partition = np.floor(len(users) * (1 - split))

        train_idx = shuffle[:partition]
        test_idx = shuffle[partition:]

        users_train = users[train_idx]
        users_test = users[test_idx]

        movies_train = movies[train_idx]
        movies_test = movies[test_idx]

        ratings_train = ratings[train_idx]
        ratings_test = ratings[test_idx]

        return users_train, movies_train, ratings_train, users_test, movies_test, ratings_test



    def train(self, users, movies, ratings, featMat,wordSeq, eval_type='AUC', val_freq=5):

        users_train, movies_train, ratings_train, users_test, movies_test, ratings_test = \
            self.train_test_split(users, movies, ratings)

        num_batches = movies_train.shape[0] // self.batch_size

        for i in range(self.epochs):

            avg_cost = 0

            for b_idx in range(num_batches):
                ratings_batch = ratings_train[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]

                users_batch = users_train[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]

                movie_ids = movies_train[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]
                movie_batch = featMat[movie_ids]
                lstm_batch = wordSeq[movie_ids]

                avg_cost += (self.session.run([self.cost, self.optimizer],
                                              {self.users: users_batch,
                                               self.lstmFeatures: lstm_batch,
                                               self.movieFeatures:movie_batch,
                                               self.rating: ratings_batch})[0]) / self.batch_size

            print ("Epoch: ", i, " Average Cost: ", avg_cost / num_batches)

            if i % val_freq == 0:
                if eval_type == 'AUC':
                    auc_mean = 0
                    uni_users = np.unique(users_test)
                    for usr in uni_users:
                        usr_idxes = users_test == usr
                        usr_idxes = np.where(usr_idxes)
                        usr_u = users_test[usr_idxes]
                        movie_u = movies_test[usr_idxes]
                        rtg_u = ratings_test[usr_idxes]
                        if len(usr_u) < 3:
                            continue
                        yhat = (self.session.run([self.yhat],
                                                 {self.users: usr_u, self.movieFeatures: movie_u,
                                                  self.rating: rtg_u})[0])
                        auc_mean += sklearn.metrics.auc(yhat, rtg_u, reorder=True) / len(uni_users)

                    print ("Testing AUC mean: ", auc_mean)

                if eval_type == 'MSE':
                    mse, r = self.evaluate(users_test, movies_test, ratings_test, featMat,wordSeq)
                    # r2 =  self.evaluate(users_test,movies_test,ratings_test,featMat,self.r_sqr)

                    print ("Testing MSE: ", mse)
                    print ("Testing R^2: ", r)

    def evaluate(self,users_test,movies_test,ratings_test,featMat,wordSeq):

        num_batches = movies_test.shape[0] // self.batch_size

        avg_mse = 0
        corr = 0

        for b_idx in range(num_batches):
            ratings_batch = ratings_test[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]

            users_batch = users_test[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]

            movie_ids = movies_test[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]
            movie_batch = featMat[movie_ids]
            lstm_batch = wordSeq[movie_ids]

            mse,yhat= (self.session.run([self.cost ,self.yhat],
                                          {self.users: users_batch, self.lstmFeatures: lstm_batch,
                                           self.movieFeatures: movie_batch,
                                           self.rating: ratings_batch}))

            avg_mse += mse/ self.batch_size

            corr += np.corrcoef(yhat,ratings_batch)[0,1]

        return avg_mse / num_batches , corr/ num_batches

    @staticmethod
    def map2idx(movieratings, mergedScrape_ML):

        users = movieratings['userId'].values
        movies = movieratings['movieId'].values

        # unique users / movies
        uni_users = movieratings['userId'].unique()
        uni_movies = mergedScrape_ML['movieId'].unique()

        # dict mapping the id to an index
        user_map = dict(zip(uni_users, range(len(uni_users))))
        movie_map = dict(zip(uni_movies, range(len(uni_movies))))

        pairs = []
        for user, movie, rating in zip(users, movies, movieratings['rating']):
            if movie in movie_map:
                pairs.append((user_map[user], movie_map[movie], rating))

        return np.array(pairs), len(uni_users), len(uni_movies)


def clean_person_string(raw_text):
    if raw_text == '':
        return ''

    cast = raw_text.split('>')
    cast = [s.split(":_")[1] for s in cast[0:-1]]
    cast = ["_".join(s.split(",")).replace(" ", "") for s in cast]

    return " ".join(cast)


def lstmFeatures(movieData):

    parser = argparse.ArgumentParser(description='Workday Text Classification Library')
    args = parser.parse_args()
    args.n = 2000
    args.word2vec = True
    args.embedSize = 50
    args.max_len = 20

    preprocessor = WordSequencePreProcessor(args)

    preprocessor.fit(movieData['plot'])

    X = preprocessor.transform(movieData['plot'])

    return X, preprocessor.vect.embedding_mat.astype(np.float)

def featureMatrix(movieData):

    #plot_vect = TfidfVectorizer(stop_words='english', max_features=2000, max_df=.9, min_df=.02)
    person_vect = CountVectorizer(max_features=400, max_df=.9, min_df=10)

    #plotFeatures = plot_vect.fit_transform(movieData['plot']).toarray()

    cast_str = movieData['cast'].apply(clean_person_string)
    director_str = movieData['director'].apply(clean_person_string)
    editor_str = movieData['editor'].apply(clean_person_string)
    writer_str = movieData['writer'].apply(clean_person_string)

    people_df = pd.DataFrame([cast_str, director_str, editor_str, writer_str])

    people_strings = people_df.apply(lambda x: ' '.join(x), axis=0)

    personFeatures = person_vect.fit_transform(people_strings).toarray()

    #movieFeatures = np.concatenate([plotFeatures, personFeatures], axis=1)

    return personFeatures


# The first iteration here will be just using plot
if __name__ == '__main__':

    scrapedMovieData = pd.read_csv('movieDataList.csv', index_col=0)
    scrapedMovieData = scrapedMovieData.fillna('')

    # Movie Lens rating data
    movieratings = pd.read_csv('ratings.csv').sample(10000)

    # List of movies in order
    movieLenseMovies = pd.read_csv('movies.csv')

    movieLenseMovies.drop('genres', axis=1, inplace=True)

    featMat = featureMatrix(scrapedMovieData)

    word_seq,embed_mat = lstmFeatures(scrapedMovieData)

    mergedScrape_ML = pd.merge(scrapedMovieData, movieLenseMovies, left_on='movie_len_title',
                               right_on='title',
                               how='left')
    mergedScrape_ML.drop_duplicates(subset='movie_len_title', inplace=True)


    #Load Image Features
    imageFeatures = pd.read_csv('imagefeatures.csv', header=None)
    imageFeatures = imageFeatures.as_matrix()

    # User and movie ids mapped to be on continuous interval
    triples, num_users, num_movie = HybridCollabFilter.map2idx(movieratings, mergedScrape_ML)

    user_idx = triples[:, 0]
    movie_idx = triples[:, 1]
    ratings = triples[:, 2]

    ratings = ratings - ratings.mean()

    movie_idx = movie_idx.astype(int)

    # (self, numUsers, embedding_dim,input_dim):
    movieModel = HybridCollabFilter(num_users, 20, featMat.shape[1],20,50,embed_mat)
    movieModel.train(user_idx, movie_idx, ratings, featMat,word_seq, eval_type="MSE")

