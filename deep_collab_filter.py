import tensorflow as tf
import sys
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

class HybridCollabFilter():

    def __init__(self, numUsers, embedding_dim,input_dim):

        # hyper parameters
        self.batch_size = 32
        self.numUsers = numUsers
        self.epochs = 100
        self.init_var =.01

        #Movie Features
        self.movieFeatures = tf.placeholder(tf.float32, shape=(None,input_dim))

        # input tensors for songs, usres, ratings
        self.users = tf.placeholder(tf.int32, shape=(None))
        self.rating = tf.placeholder(tf.float32, shape=(None))

        # embedding matricies for users
        self.userMat = tf.Variable(self.init_var*tf.random_normal([numUsers, embedding_dim]))
        self.userBias = tf.Variable(self.init_var*tf.random_normal([numUsers,]))

        #Model parameters for movies
        self.W = tf.Variable(self.init_var*tf.random_normal([input_dim, embedding_dim]))
        self.b = tf.Variable(self.init_var*tf.random_normal([embedding_dim]))

        movieTensor = tf.matmul(self.movieFeatures,self.W) + self.b

        # map each user/song to its feature vector
        self.U = tf.nn.embedding_lookup(self.userMat, self.users)
        self.u_b = tf.nn.embedding_lookup(self.userBias, self.users)

        # predicted rating is dot product of user and song
        self.yhat = tf.reduce_sum(tf.mul(self.U, movieTensor) , 1) + self.u_b

        self.cost = tf.nn.l2_loss(self.yhat - self.rating)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001).minimize(self.cost)

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())


    def train_test_split(self,users,movies,ratings,split=.1):

        shuffle  = np.random.permutation(len(users))

        partition = np.floor(len(users) * (1-split))

        train_idx = shuffle[:partition]
        test_idx = shuffle[partition:]

        users_train = users[train_idx]
        users_test = users[test_idx]

        movies_train = movies[train_idx]
        movies_test = movies[test_idx]

        ratings_train = ratings[train_idx]
        ratings_test = ratings[test_idx]

        return users_train,movies_train,ratings_train , users_test,movies_test,ratings_test


    def train(self, users, movies, ratings,val_freq=5):

        users_train, movies_train, ratings_train, users_test, movies_test, ratings_test = \
            self.train_test_split(users,movies,ratings)

        num_batches = movies_train.shape[0] // self.batch_size

        for i in range(self.epochs):

            avg_cost = 0

            for b_idx in range(num_batches):

                ratings_batch  = ratings_train[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]

                users_batch = users_train[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]
                movie_batch = movies_train[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]

                avg_cost +=  (self.session.run([self.cost, self.optimizer],
                                             {self.users: users_batch, self.movieFeatures: movie_batch,
                                              self.rating: ratings_batch})[0] ) / self.batch_size

            print "Epoch: ",i, " Average Cost: ",avg_cost / num_batches

            if i % val_freq ==0:

                oos_cost = self.session.run(self.cost, {self.users: users_test, self.movieFeatures: movies_test,
                                              self.rating: ratings_test})

                print "Testing Loss: " , oos_cost / len(users_test)

    @staticmethod
    def map2idx(movieratings):

        users = movieratings['userId'].values
        songs = movieratings['movieId'].values

        # unique users / songs
        uni_users = movieratings['userId'].unique()
        uni_songs = movieratings['movieId'].unique()

        # dict mapping the id to an index
        user_map = dict(zip(uni_users, range(len(uni_users))))
        song_map = dict(zip(uni_songs, range(len(uni_songs))))

        user_idx = np.array([user_map[user] for user in users])
        song_idx = np.array([song_map[song] for song in songs])

        return user_idx,song_idx,len(uni_users),len(uni_songs)



def clean_cast_string(raw_text):
    n=10

    if raw_text ==np.nan:
        return ''

    cast = raw_text.split('>')
    cast = [ s.split(":_")[1] for s in cast]
    cast = [ s.split(",")[1] +s.split(",")[0] for s in cast]

    return " ".join(cast[0:n])

def clean_director_string(raw_string):

    director = raw_string.split(':_')[1].split('>')[0].split(',')
    director = director[1] +director[0]
    return director

def featureMatrix(movieData):

    #Structured Data
    rawplot = movieData['plot']

    #Unstructured data
    cast_str = movieData['cast'].apply(clean_cast_string)
    #director_str = movieData['director'].apply(clean_director_string)
    #producers = movieData['producers'].apply(clean_cast_string)

    print cast_str

#The first iteration here will be just using plot
if __name__ == '__main__':

    #Data on each movie from IMDB
    movieData = pd.read_csv('/Users/Hadoop/Desktop/movieData.csv')

    featMat = featureMatrix(movieData)

    '''featMat = featureMatrix()
    #Movie Lens rating data
    movieratings = pd.read_csv('ratings.csv')

    #A matrix (num movies , num features) that has the feature represenetion of each movie
    featMat = featureMatrix()

    #User and movie ids mapped to be on continuous interval
    user_idx, movie_idx, num_users, num_movie = HybridCollabFilter.map2idx(movieratings)

    #REMOVE THIS
    movie_idx = filter( lambda x: x  < 20,movie_idx)
    print len(movie_idx)


    user_idx = user_idx
    movieFeatures =  featMat[movie_idx]
    ratings = movieratings.ix[:, 2].values

    #REMOVE THIS
    user_idx = np.random.randint(0,50,911)
    ratings = ratings[0:911]

    #(self, numUsers, embedding_dim,input_dim):
    movieModel = HybridCollabFilter(50, 10, 200)
    movieModel.train(user_idx, movieFeatures, ratings)
    '''
