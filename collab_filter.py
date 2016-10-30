import tensorflow as tf
import sys
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

class MF_RS():
    def __init__(self, numUsers, numSongs, embedding_dim):

        # hyper parameters
        self.batch_size = 32
        self.numUsers = numUsers
        self.numSongs = numSongs
        self.epochs = 100

        # embedding matricies for users and songs
        self.userMat = tf.Variable(tf.random_normal([numUsers, embedding_dim]))
        self.songMat = tf.Variable(tf.random_normal([numSongs, embedding_dim]))

        # embedding matricies for users and songs
        self.userBias = tf.Variable(tf.random_normal([numUsers,]))
        self.songBias = tf.Variable(tf.random_normal([numSongs,]))

        # input tensors for songs, usres, ratings
        self.users = tf.placeholder(tf.int32, shape=(None))
        self.songs = tf.placeholder(tf.int32, shape=(None))
        self.rating = tf.placeholder(tf.float32, shape=(None))

        # map each user/song to its feature vector
        self.U = tf.nn.embedding_lookup(self.userMat, self.users)
        self.W = tf.nn.embedding_lookup(self.songMat, self.songs)
        self.u_b = tf.nn.embedding_lookup(self.userBias, self.users)
        self.w_b = tf.nn.embedding_lookup(self.songBias, self.songs)

        # predicted rating is dot product of user and song
        self.yhat = tf.reduce_sum(tf.mul(self.U, self.W) , 1) + self.u_b + self.w_b

        self.cost = tf.nn.l2_loss(self.yhat - self.rating)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=.01).minimize(self.cost)

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())


    def train_test_split(self,users,songs,ratings,split=.1):

        X = np.array([users,songs]).transpose()

        X_train, X_test, y_train, y_test = train_test_split( X, ratings, test_size = split)

        return X_train, X_test, y_train, y_test

    def train(self, users, songs, ratings,val_freq=5):

        X_train, X_test, y_train, y_test = self.train_test_split(users,songs,ratings)

        num_batches = X_train.shape[0] // self.batch_size

        for i in range(self.epochs):

            avg_cost = 0

            for b_idx in range(num_batches):
                x_batch = X_train[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]
                ratings_batch  = y_train[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]

                users_batch = x_batch[:,0]
                songs_batch = x_batch[:,1]

                avg_cost +=  (self.session.run([self.cost, self.optimizer],
                                             {self.users: users_batch, self.songs: songs_batch,
                                              self.rating: ratings_batch})[0] ) / self.batch_size

            print "Epoch: ",i, " Average Cost: ",avg_cost / num_batches

            if i % val_freq ==0:
                users_test = X_test[:,0]
                songs_test = X_test[:,1]
                oos_cost = self.session.run(self.cost, {self.users: users_test, self.songs: songs_test,
                                              self.rating: y_test})

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


if __name__ == '__main__':
    movieratings = pd.read_csv('ratings.csv')

    user_idx, song_idx, num_users, num_songs = MF_RS.map2idx(movieratings)
    ratings = movieratings.ix[:, 2].values

    num_features = 30

    songmodel = MF_RS(num_users, num_songs, num_features)

    songmodel.train(user_idx, song_idx, ratings)