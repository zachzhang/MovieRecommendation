import tensorflow as tf
import sys
import numpy as np
import pandas as pd
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class HybridCollabFilter():

    def __init__(self, numUsers, numMovies, reg_l = 1, 
                 inputdim_image = None, inputdim_meta = None, 
                 edim_image = 3, edim_meta = 3,
                 edim_hidden_1 = 20, edim_hidden_2 = 0,
                 edim_custom_tf = 3, edim_user = 10):
        
        edim_hidden_output = edim_user - edim_custom_tf
        edim_movie = edim_image + edim_meta
        edim_hidden_output = edim_user - edim_custom_tf
        
        # hyper parameters
        self.batch_size = 512
        self.numUsers = numUsers
        self.numMovies = numMovies
        self.epochs = 11
        self.init_var =.01
        self.l = reg_l

        #Movie Features
        self.CustomFeatsMat = tf.Variable(tf.random_normal([numMovies, edim_custom_tf]))
        self.imageFeatures = tf.placeholder(tf.float32, shape=(None,inputdim_image))
        self.metaFeatures = tf.placeholder(tf.float32, shape=(None,inputdim_meta))
        
        # input tensors for movies, usres, ratings
        self.users = tf.placeholder(tf.int32, shape=(None))
        self.movies = tf.placeholder(tf.int32, shape=(None))
        self.rating = tf.placeholder(tf.float32, shape=(None))

        # embedding matricies for users
        self.tfCustomFeatures = tf.Variable(
                    self.init_var*tf.random_normal([numMovies, edim_custom_tf]))
        self.userMat = tf.Variable(self.init_var*tf.random_normal([numUsers, edim_user]))
        self.userBias = tf.Variable(self.init_var*tf.random_normal([numUsers,]))

        #Model parameters for images
        self.W_image = tf.Variable(self.init_var*tf.random_normal([inputdim_image, edim_image]))
        self.b_image = tf.Variable(self.init_var*tf.random_normal([edim_image]))
        #Model parameters for metadata
        self.W_meta = tf.Variable(self.init_var*tf.random_normal([inputdim_meta, edim_meta]))
        self.b_meta = tf.Variable(self.init_var*tf.random_normal([edim_meta]))
        
        
        self.customTensor = tf.nn.embedding_lookup(self.tfCustomFeatures, self.movies)
        self.imageTensor = tf.matmul(self.imageFeatures,self.W_image) + self.b_image
        self.metaTensor = tf.matmul(self.metaFeatures,self.W_meta) + self.b_meta
        #find cleaner way to calculate height:
        #self.height = self.metaTensor.get_shape()[0]
        self.movieTensor = tf.concat(1, [self.imageTensor, self.metaTensor])
        
        self.movie_size = edim_image + edim_meta
        #hidden dimension weights
        self.W_hidden_1 = tf.Variable(self.init_var * 
                                      tf.random_normal([self.movie_size, edim_hidden_1]))
        #self.W_hidden_2 = tf.Variable(self.init_var *
        #                              tf.random_normal([edim_hidden_1, edim_hidden_2]))
        self.W_hidden_output = tf.Variable(
                self.init_var * tf.random_normal([edim_hidden_1, edim_user - edim_custom_tf]))
        
        self.b_hidden_1 = tf.Variable(tf.random_normal([edim_hidden_1]))
        #self.b_hidden_2 = tf.Variable(tf.random_normal([edim_hidden_2]))
        self.b_hidden_output = tf.Variable(tf.random_normal([edim_user - edim_custom_tf]))
        
        #hidden dimension values
        self.fullyconnected1 = tf.matmul(self.movieTensor, self.W_hidden_1) + self.b_hidden_1
        self.sig_hidden_1 = tf.sigmoid(self.fullyconnected1)
        self.act_hidden_1 = tf.concat(1, [self.sig_hidden_1])
        
        #self.fullyconnected2 = tf.matmul(self.act_hidden_1, self.W_hidden_2) + self.b_hidden_2
        #self.sig_hidden_2 = tf.sigmoid(self.fullyconnected2)
        #self.act_hidden_2 = tf.concat(1, [self.sig_hidden_2])
        
        self.fullyconnectedoutput = tf.matmul(self.act_hidden_1, self.W_hidden_output) + self.b_hidden_output
        self.sig_hidden_output = tf.sigmoid(self.fullyconnectedoutput)
        self.act_hidden_output = tf.concat(1, [self.sig_hidden_output])
        
        self.movie_final_repr = tf.concat(1, [self.act_hidden_output, self.customTensor])

        
        # map each user/movie to its feature vector
        self.U = tf.nn.embedding_lookup(self.userMat, self.users)
        self.u_b = tf.nn.embedding_lookup(self.userBias, self.users)

        # predicted rating is dot product of user and movie
        self.yhat = tf.reduce_sum(tf.mul(self.U, self.movie_final_repr) , 1) + self.u_b
        
        self.cost = tf.nn.l2_loss(self.yhat - self.rating) + \
                    tf.reduce_mean(self.l * tf.abs(self.W_image) ) + \
                    tf.reduce_mean(self.l * tf.abs(self.b_image) ) + \
                    tf.reduce_mean(self.l * tf.abs(self.W_meta) ) + \
                    tf.reduce_mean(self.l * tf.abs(self.b_meta) ) + \
                    tf.reduce_mean(self.l * tf.abs(self.W_hidden_1) ) + \
                    tf.reduce_mean(self.l * tf.abs(self.W_hidden_output) )
                    #tf.reduce_mean(self.l * tf.abs(self.W_hidden_2) ) + \
        self.learning_rate = 0.1

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
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


    def train(self, users, movies, ratings,imageFeatures, metaFeatures, eval_type = 'AUC', val_freq=5):

        users_train, movies_train, ratings_train, users_test, movies_test, ratings_test = \
            self.train_test_split(users,movies,ratings)
            
        self.num_train = movies_train.shape[0]
        self.num_test = movies_test.shape[0]
        self.num_batches = self.num_train // self.batch_size

        for i in range(self.epochs):
            self.adjust_from_quick_to_stable_training()
            avg_cost = 0
            
            for b_idx in range(self.num_batches):

                ratings_batch  = ratings_train[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]
                users_batch = users_train[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]
                movie_ids = movies_train[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]
                
                if b_idx == self.num_batches - 1:
                    ratings_batch  = ratings_train[self.batch_size * b_idx:]
                    users_batch = users_train[self.batch_size * b_idx:]
                    movie_ids = movies_train[self.batch_size * b_idx:]

                movie_batch = movie_ids
                image_batch = imageFeatures[movie_ids]
                meta_batch = metaFeatures[movie_ids]

                avg_cost +=  (self.session.run([self.cost, self.optimizer],
                                   {self.users: users_batch, 
                                    self.imageFeatures: image_batch,
                                    self.metaFeatures: meta_batch,
                                    self.movies: movie_batch,
                                    self.rating: ratings_batch})[0] ) / len(movie_batch)


            print ("Epoch: ", i, " Average Cost: ",avg_cost / self.num_batches)

            if i % val_freq ==0 or i == self.epochs - 1:
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
                                                 {self.users: usr_u, self.movies: movie_u,
                                                      self.imageFeatures: image_batch,
                                                      self.metaFeatures: meta_batch,
                                                      self.rating: rtg_u})[0] )
                        auc_mean += sklearn.metrics.auc(yhat, rtg_u, reorder = True) / len(uni_users)

                    print ("Testing AUC mean: " , auc_mean)
                    err = auc_auc

                if eval_type == 'MSE':
                    self.test_batch_size = np.min((10000, self.num_test))
                    self.test_num_batches = self.num_test // self.test_batch_size
                    mse = 0
                    
                    for b_idx in range(self.test_num_batches):
                        mse += self.get_mse(imageFeatures, metaFeatures, movie_ids,
                            ratings_test, users_test, movies_test, b_idx)

                    print ("Testing MSE: ", mse)
                    err = mse
        return err 
    
    def get_mse(self, imageFeatures, metaFeatures, movie_ids, 
                ratings_test, users_test, movies_test, b_idx):
        
        ratings_batch  = ratings_test[self.test_batch_size * b_idx:self.test_batch_size * (b_idx + 1)]
        users_batch = users_test[self.test_batch_size * b_idx:self.test_batch_size * (b_idx + 1)]
        movie_ids = movies_test[self.test_batch_size * b_idx:self.test_batch_size * (b_idx + 1)]

        if b_idx == self.test_num_batches - 1:
            ratings_batch  = ratings_test[self.test_batch_size * b_idx:]
            users_batch = users_test[self.test_batch_size * b_idx:]
            movie_ids = movies_test[self.test_batch_size * b_idx:]
            
        movie_batch = movie_ids
        image_batch = imageFeatures[movie_ids]
        meta_batch = metaFeatures[movie_ids]

        mse = self.session.run(self.cost,
                               {self.users: users_batch, 
                                self.imageFeatures: image_batch,
                                self.metaFeatures: meta_batch,
                                self.movies: movie_batch,
                                self.rating: ratings_batch}) / len(users_test)
        return mse

                    
    
    def adjust_from_quick_to_stable_training(self):
        self.learning_rate = self.learning_rate * .9
        self.batch_size = int(round(np.min((self.batch_size * 1.1, self.num_train))))
        self.num_batches = self.num_train // self.batch_size


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
    cast = [ s.split(":_")[1] for s in cast[0:-1]]
    cast = ["_".join(s.split(",")).replace(" ", "") for s in cast]

    return " ".join(cast)



def featureMatrix(movieData):
    #TfidfVectorizer
    #plot_vect = CountVectorizer(stop_words='english',max_features=2000,max_df=.9,min_df=.02,ngram_range =(1,2))
    plot_vect = TfidfVectorizer(stop_words='english',max_features=2000,max_df=.9,min_df=.02)
    person_vect = TfidfVectorizer(max_features=400,max_df=.9,min_df=30)

    plotFeatures = plot_vect.fit_transform(movieData['plot']).toarray()

    cast_str = movieData['cast'].apply(clean_person_string)
    director_str = movieData['director'].apply(clean_person_string)
    editor_str = movieData['editor'].apply(clean_person_string)
    writer_str = movieData['writer'].apply(clean_person_string)

    people_df = pd.DataFrame([cast_str,director_str,editor_str,writer_str])

    people_strings = people_df.apply( lambda x:  ' '.join(x) , axis=0)

    personFeatures = person_vect.fit_transform(people_strings).toarray()

    movieFeatures = np.concatenate([plotFeatures,personFeatures],axis=1)

    #movieFeatures = plotFeatures

    return movieFeatures


#The first iteration here will be just using plot
if __name__ == '__main__':

    scrapedMovieData = pd.read_csv('movieDataList.csv', index_col=0)
    scrapedMovieData = scrapedMovieData.fillna('')
    # Movie Lens rating data
    movieratings = pd.read_csv('ratings.csv', nrows = 1000000)

    # List of movies in order
    movieLenseMovies = pd.read_csv('movies.csv')

    featMat = featureMatrix(scrapedMovieData)

    movieLenseMovies.drop('genres', axis=1, inplace=True)

    mergedScrape_ML = pd.merge(scrapedMovieData, movieLenseMovies, left_on='movie_len_title',
                               right_on='title',
                               how='left')

    mergedScrape_ML.drop_duplicates(subset='movie_len_title', inplace=True)

    #User and movie ids mapped to be on continuous interval
    triples, num_users, num_movie = HybridCollabFilter.map2idx(movieratings,mergedScrape_ML)

    user_idx = triples[:,0]
    movie_idx = triples[ :,1]
    ratings = triples[:, 2]

    movie_idx = movie_idx.astype(int)
    #movieFeatures = featMat[movie_idx.astype(int)]

    #image features
    imageFeatures = pd.read_csv('imagefeatures.csv', header=None)
    imageFeatures = imageFeatures.as_matrix()
    
    allfeatures = np.concatenate((imageFeatures, featMat), axis=1)
    edims_image = [3, 5]
    tf_custom_dim = [3, 5]
    meta_dims = [3, 5]
    errmat = np.zeros([len(meta_dims), len(tf_custom_dim), len(edims_image)])
    #(self, numUsers, embedding_dim,input_dim):
    for meta_idx, meta_dim in enumerate(meta_dims):
        for imagedim_idx, imagedim in enumerate(edims_image):
            for tfdim_idx, tfdim in enumerate(tf_custom_dim):
                movieModel = HybridCollabFilter(num_users, num_movie, edim_image = imagedim, 
                                                inputdim_image = imageFeatures.shape[1],
                                                inputdim_meta = featMat.shape[1],
                                                edim_custom_tf = tfdim)
                errmat[meta_idx, imagedim_idx, tfdim_idx] = \
                        movieModel.train(user_idx,movie_idx, ratings, 
                                         imageFeatures = imageFeatures, 
                                         metaFeatures = featMat, eval_type = "MSE")
    print(errmat)

