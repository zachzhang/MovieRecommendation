
# coding: utf-8

# In[46]:

import tensorflow as tf
import sys
print(sys.version)
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# In[47]:

class MF_RS():
    def __init__(self, numUsers, numSongs, embedding_dim, reg_lambda = 0.01):
        
        #hyper parameters
        self.batch_size = np.min([2000, numUsers, numSongs]);
        self.numUsers = numUsers
        self.numSongs = numSongs
        self.epochs = 50
        self.reg_lambda = reg_lambda
        
        #embedding matricies for users and songs
        self.userMat = tf.Variable(tf.random_normal([numUsers, embedding_dim]))
        self.songMat = tf.Variable(tf.random_normal([numSongs, embedding_dim]))
        self.userBias = tf.Variable(tf.random_normal([numUsers]))
        self.songBias = tf.Variable(tf.random_normal([numSongs]))
        self.overallBias = tf.Variable(tf.random_normal([1]))
        
        #input tensors for songs, usres, ratings
        self.users = tf.placeholder(tf.int32, shape =(self.batch_size))
        self.songs = tf.placeholder(tf.int32, shape =(self.batch_size))
        self.rating = tf.placeholder(tf.float32, shape = (self.batch_size))
        
        #map each user/song to its feature vector
        self.U = tf.nn.embedding_lookup(self.userMat, self.users)
        self.W = tf.nn.embedding_lookup(self.songMat, self.songs)
        #map each user/song bias to its bias vector
        self.U_bias = tf.nn.embedding_lookup(self.userBias, self.users)
        self.W_bias = tf.nn.embedding_lookup(self.songBias, self.songs)
        
        #predicted rating is dot product of user and song
        bias = self.U_bias+self.W_bias+self.overallBias
        pq = tf.reduce_sum(tf.mul(self.U, self.W), 1)
        self.yhat = pq + bias
        
        self.reg = self.reg_lambda * ( tf.reduce_sum((tf.square(self.U) + tf.square(self.W))) + 
                                 tf.reduce_sum(tf.square(self.U_bias) + tf.square(self.W_bias)))
        self.error = tf.reduce_mean(tf.nn.l2_loss(self.yhat - self.rating))
        self.cost = (self.error + self.reg)/1e4
        self.optimizer = tf.train.AdamOptimizer(learning_rate = .01).minimize(self.cost)
        
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())   
        
        
    def train(self, users, songs, ratings, verb = 0):
        
        for i in range(self.epochs):
            
            avg_cost = 0
            perm = np.random.permutation(len(ratings))
            num_batches = len(ratings) // self.batch_size
            
            for b_idx in range(num_batches):
                
                batch = perm[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]
                users_batch = users[batch]
                songs_batch = songs[batch]
                ratings_batch = ratings[batch]
                                
                avg_cost += self.session.run([self.cost, self.optimizer],
                          {self.users:users_batch, self.songs:songs_batch, self.rating:ratings_batch})[0]
            if verb > 0:
                print(avg_cost/num_batches)
    def test(self, users, songs):
        yhat = np.zeros(len(users))
        num_batches = len(users) // self.batch_size
        for b_idx in range(num_batches):
            batch = range(self.batch_size * b_idx,self.batch_size * (b_idx + 1))
            users_batch = users[batch]
            songs_batch = songs[batch]
            yhat[batch] = self.session.run([self.yhat],
                      {self.users:users_batch, self.songs:songs_batch})[0]
        batch = range(-self.batch_size,0)
        users_batch = users[batch]
        songs_batch = songs[batch]
        yhat[batch] = self.session.run([self.yhat],
                      {self.users:users_batch, self.songs:songs_batch})[0]
        return yhat
    def evaluate(self, users, songs, ratings):
        yhat = self.test(users, songs)
        return np.mean((yhat - ratings)**2)


# In[48]:

a = np.array([1, 2, 3, 4, 5])
b = np.array([1, 2, 3, 4, 5])
c = np.array([4, 3, 2, 5, 1])
#unique users / songs
uni_a = np.unique(a)
uni_b = np.unique(b)

#dict mapping the id to an index
a_map = dict(zip(uni_a,range(len(uni_a))))
b_map = dict(zip(uni_b,range(len(uni_b))))

user_idx =  np.array([ a_map[user] for user in a])
song_idx =  np.array([ b_map[song] for song in b])
model = MF_RS(len(uni_a), len(uni_b), 7)
np.random.seed(2)
model.train(user_idx, song_idx, c)


# In[49]:

movieratings = pd.read_csv('ratingsfull.csv')


# In[50]:

#movieratings.describe()


# In[51]:

def getDfSummary(input_data):
    output_data = input_data.describe(include = 'all').T
    var = pd.DataFrame(data = {'nanvals': pd.Series(), 'number_distinct': pd.Series()})
    for i in range(len(input_data.columns)):
        nanvals = input_data.ix[:,i].isnull().sum()
        number_distinct = len(input_data.ix[:,i].value_counts())
        var = var.append(pd.DataFrame([[nanvals, number_distinct]], columns = ['nanvals', 'number_distinct']))
    var.index = output_data.index.values
    output_data['nanvals'] = var['nanvals']
    output_data['number_distinct'] = var['number_distinct']
    return output_data
output_data = getDfSummary(movieratings)
print(output_data)

# In[52]:

users = movieratings.ix[:,0].values
songs = movieratings.ix[:,1].values
ratings = movieratings.ix[:,2].values

#unique users / songs
uni_users = movieratings['userId'].unique()
uni_songs = movieratings['movieId'].unique()

#dict mapping the id to an index
user_map = dict(zip(uni_users,range(len(uni_users))))
song_map = dict(zip(uni_songs,range(len(uni_songs))))

user_idx =  np.array([ user_map[user] for user in users])
song_idx =  np.array([ song_map[song] for song in songs])

print(len(uni_users),len(uni_songs))

perm = np.random.permutation(len(users))
trn_idx = perm[:(len(users)*2)//3]
val_idx = perm[(len(users)*2)//3:]
user_idx_trn, song_idx_trn, ratings_trn = user_idx[trn_idx], song_idx[trn_idx], ratings[trn_idx]
user_idx_val, song_idx_val, ratings_val = user_idx[val_idx], song_idx[val_idx], ratings[val_idx]


# In[53]:

songmodel = MF_RS(len (uni_users), len(uni_songs), 11)
print(songmodel.evaluate(user_idx_val, song_idx_val, ratings_val))
songmodel.epochs = 5
songmodel.train(user_idx_trn, song_idx_trn, ratings_trn, verb = 1)
songmodel.evaluate(user_idx_val, song_idx_val, ratings_val)


# In[55]:

edims = [10, 30]
lambdas = [10**i for i in range(-4, -2)]
errmat = np.zeros([len(edims), len(lambdas)])
for eidx, edim in enumerate(edims):
    for lidx, lamb in enumerate(lambdas):
        songmodel = MF_RS(len (uni_users), len(uni_songs), edim, reg_lambda=lamb)
        print("accuracy before training", songmodel.evaluate(user_idx_val, song_idx_val, ratings_val))
        np.random.seed(1)
        songmodel.train(user_idx_trn, song_idx_trn, ratings_trn)
        err = songmodel.evaluate(user_idx_val, song_idx_val, ratings_val)
        print("MSE after training with edim: ", edim, " and lambda: ", lamb, ": ", err)
        errmat[eidx, lidx] = err
errmat


# In[ ]:



