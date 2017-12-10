import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import copy
from numpy.linalg import solve


def train_test_split(ratings, testing_size = 5):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size = testing_size,
                                        replace = False)

        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]

    assert(np.all((train * test) == 0))
    return train, test


def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    mse = mean_squared_error(pred, actual)
    return mse



def normalize_userwise(train,test):
    train_new = copy.deepcopy(train)
    test_new = copy.deepcopy(test)
    for i in range(train_new.shape[0]):
        items = np.nonzero(train_new[i,:])[0].tolist()
        user_avg = np.sum(train_new[i,items])/len(items)
        items_test = np.nonzero(test_new[i,:])[0].tolist()
        train_new[i,items] = (train_new[i,items]-user_avg)
        test_new[i,items_test] = (test_new[i,items_test]-user_avg)
    return train_new, test_new


## ALS
class RecommendationALS():

    def __init__(self,
                 ratings,
                 n_factors = 10,
                 item_reg = 0.0,
                 user_reg = 0.0,
                 max_iter = 50,
                 random_state = None,
                 verbose = True):
        """
        Train a matrix factorization model to predict empty
        entries in a matrix. The terminology assumes a
        ratings matrix which is ~ user x item

        Params
        ======
        ratings : (ndarray)
            User x Item matrix with corresponding ratings

        n_factors : (int)
            Number of latent factors to use in matrix
            factorization model

        item_reg : (float)
            Regularization term for item latent factors

        user_reg : (float)
            Regularization term for user latent factors

        max_iter : (int)
            The maximum number of passes over the training data. Defaults to 50.

        verbose : (bool)
            Whether or not to printout training progress
        """

        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.item_reg = item_reg
        self.user_reg = user_reg
        self._v = verbose
        self.n_iter = max_iter
        self._validate_params()

    def _validate_params(self):

        """
        Validate input params.
        """

        if self.n_factors <= 0 or not isinstance(self.n_factors, int):
            raise ValueError("n_factors must be some positive integers. Got %f" % self.n_factors)

        if not isinstance(self._v, bool):
            raise ValueError("verbose must be either True or False")

        if not isinstance(self.n_factors, int):
            raise ValueError("n_factors must be an integer")

        if self.n_iter is not None and (self.n_iter <= 0 or not isinstance(self.n_iter, int)):
            raise ValueError("max_iter must be some positive integers. Got %f" % self.n_iter)

        if self.item_reg < 0.0:
            raise ValueError("item_reg must be >= 0")

        if self.user_reg < 0.0:
            raise ValueError("user_reg must be >= 0")


    def als_step(self,
                 latent_vectors,
                 fixed_vecs,
                 ratings,
                 _lambda,
                 type = 'user'):
        """
        One of the two ALS steps. Solve for the latent vectors
        specified by type.
        """

        if type == 'user':
            # fix item vector and solve for the user vector
            YTY = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(YTY.shape[0]) * _lambda

            for u in range(latent_vectors.shape[0]):
                latent_vectors[u, :] = solve((YTY + lambdaI), ratings[u, :].dot(fixed_vecs))

        elif type == 'item':
            # fix user vector and solve for the item vector
            XTX = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(XTX.shape[0]) * _lambda

            for i in range(latent_vectors.shape[0]):
                latent_vectors[i, :] = solve((XTX + lambdaI), ratings[:, i].T.dot(fixed_vecs))
        return latent_vectors


    def fit(self):
        """
        Train model for n_iter iterations.
        """
        ctr = 1
        self.user_vecs = np.random.random((self.n_users, self.n_factors))
        self.item_vecs = np.random.random((self.n_items, self.n_factors))

        while ctr <= self.n_iter:
            if ctr % 10 == 0 and self._v:
                print ('\tcurrent iteration: {}'.format(ctr))

            # als
            self.user_vecs = self.als_step(self.user_vecs,
                                           self.item_vecs,
                                           self.ratings,
                                           self.user_reg,
                                           type = 'user')

            self.item_vecs = self.als_step(self.item_vecs,
                                           self.user_vecs,
                                           self.ratings,
                                           self.item_reg,
                                           type = 'item')
            ctr += 1


        return (self.user_vecs, self.item_vecs)

    def calculate_mse(self, test):
        vecs = self.fit()
        user_vecs = vecs[0]
        item_vecs = vecs[1]

        predictions = np.zeros((user_vecs.shape[0], item_vecs.shape[0]))
        for u in range(user_vecs.shape[0]):
            for i in range(item_vecs.shape[0]):
                predictions[u, i] = user_vecs[u, :].dot(item_vecs[i, :].T)

        self.train_mse = get_mse(predictions, self.ratings)
        self.test_mse = get_mse(predictions, test)
        return (self.train_mse, self.test_mse)



# implementations
#train, test = train_test_split(ratings)
#train_normalized, test_normalized = normalize_userwise(train, test)
#RecommendationALS(train_normalized).calculate_mse(test_normalized)


## Online SGD
class OnlineRecommendationSGD():

    def __init__(self,
                 ratings,
                 n_factors = 10,
                 item_reg = 1.0,
                 user_reg = 1.0,
                 item_bias_reg = 1.0,
                 user_bias_reg = 1.0,
                 max_iter = 20,
                 learning_rate = 0.01,
                 verbose = True):

        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.item_reg = item_reg
        self.user_reg = user_reg

        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.learning_rate = learning_rate
        self.sample_row, self.sample_col = self.ratings.nonzero()
        self.n_samples = len(self.sample_row)

        self._v = verbose
        self.n_iter = max_iter


    def fit(self):
        """
        Train model
        """

        self.user_vecs = np.random.random((self.n_users, self.n_factors))
        self.item_vecs = np.random.random((self.n_items, self.n_factors))

        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])

        ctr = 1
        while ctr <= self.n_iter:
            if ctr % 10 == 0 and self._v:
                print ('\tcurrent iteration: {}'.format(ctr))

            # one sample SGD
            self.training_indices = np.arange(self.n_samples)
            np.random.shuffle(self.training_indices)

            for idx in self.training_indices:
                u = self.sample_row[idx]
                i = self.sample_col[idx]

                prediction = self.predict(u, i)


                e = (self.ratings[u,i] - prediction) # error

                # update biases
                self.user_bias[u] += self.learning_rate * \
                                (e - self.user_bias_reg * self.user_bias[u])
                self.item_bias[i] += self.learning_rate * \
                                (e - self.item_bias_reg * self.item_bias[i])

                # update latent factors
                self.user_vecs[u, :] += self.learning_rate * \
                                    (e * self.item_vecs[i, :] - \
                                     self.user_reg * self.user_vecs[u,:])
                self.item_vecs[i, :] += self.learning_rate * \
                                    (e * self.user_vecs[u, :] - \
                                     self.item_reg * self.item_vecs[i,:])
            ctr += 1


        return (self.user_vecs, self.item_vecs, self.user_bias, self.item_bias, self.global_bias)

    def predict(self, u, i):
        prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
        prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        return prediction


    def calculate_mse(self, test):
        vecs = self.fit()
        user_vecs = vecs[0]
        item_vecs = vecs[1]
        user_bias = vecs[2]
        item_bias = vecs[3]
        global_bias = vecs[4]

        predictions = np.zeros((user_vecs.shape[0], item_vecs.shape[0]))

        for u in range(user_vecs.shape[0]):
            for i in range(item_vecs.shape[0]):
                predictions[u, i] = self.predict(u, i)

        self.train_mse = get_mse(predictions, self.ratings)
        self.test_mse = get_mse(predictions, test)
        return (self.train_mse, self.test_mse)


## Mini-batch SGD
class MiniBatchRecommendationSGD():

    def __init__(self,
                 ratings,
                 n_factors = 10,
                 item_reg = 1.0,
                 user_reg = 1.0,
                 item_bias_reg = 1.0,
                 user_bias_reg = 1.0,
                 max_iter = 20,
                 learning_rate = 0.01,
                 batch_size = 5,
                 verbose = True):


        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.item_reg = item_reg
        self.user_reg = user_reg

        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.learning_rate = learning_rate
        self.sample_row, self.sample_col = self.ratings.nonzero()
        self.n_samples = len(self.sample_row)

        self._v = verbose
        self.n_iter = max_iter
        self.batch_size = batch_size


    def fit(self):
        """
        Train model
        """

        self.user_vecs = np.random.random((self.n_users, self.n_factors))
        self.item_vecs = np.random.random((self.n_items, self.n_factors))

        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])

        ctr = 1
        while ctr <= self.n_iter:
            if ctr % 10 == 0 and self._v:
                print ('\tcurrent iteration: {}'.format(ctr))

            # mini-batch SGD
            self.training_indices = np.arange(self.n_samples)
            np.random.shuffle(self.training_indices)

            for start_idx in range(0, self.n_samples - self.batch_size + 1, self.batch_size):
                idx = self.training_indices[start_idx:start_idx + self.batch_size]
                u = self.sample_row[idx]
                i = self.sample_col[idx]

                # error
                e = [self.ratings[a,b] - self.predict(a,b) for a,b in zip(u,i)]

                # update biases
                self.user_bias[u] += self.learning_rate * (e - self.user_bias_reg * self.user_bias[u])
                self.item_bias[i] += self.learning_rate * (e - self.item_bias_reg * self.item_bias[i])

                # update latent factors
                self.user_vecs[u, :] = [self.user_vecs[u, :][x] + self.learning_rate * (e[x] * self.item_vecs[i, :][x] - self.user_reg * self.user_vecs[u,:][x]) for x in range(self.batch_size)]
                self.item_vecs[i, :] = [self.item_vecs[i, :][x] + self.learning_rate * (e[x] * self.user_vecs[u, :][x] - self.item_reg * self.item_vecs[i,:][x]) for x in range(self.batch_size)]

            ctr += 1


        return (self.user_vecs, self.item_vecs, self.user_bias, self.item_bias, self.global_bias)

    def predict(self, u, i):
        prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
        prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        return prediction


    def calculate_mse(self, test):
        vecs = self.fit()
        user_vecs = vecs[0]
        item_vecs = vecs[1]
        user_bias = vecs[2]
        item_bias = vecs[3]
        global_bias = vecs[4]

        predictions = np.zeros((user_vecs.shape[0], item_vecs.shape[0]))

        for u in range(user_vecs.shape[0]):
            for i in range(item_vecs.shape[0]):
                predictions[u, i] = self.predict(u, i)

        self.train_mse = get_mse(predictions, self.ratings)
        self.test_mse = get_mse(predictions, test)
        return (self.train_mse, self.test_mse)


# added loss tolerance parameter to save computation time
class MiniBatchRecommendationSGD_tol():

    def __init__(self,
                 ratings,
                 n_factors = 10,
                 item_reg = 1.0,
                 user_reg = 1.0,
                 item_bias_reg = 1.0,
                 user_bias_reg = 1.0,
                 max_iter = 20,
                 batch_size = 50,
                 learning_rate = 0.01,
                 tolerance = 1e-3,
                 verbose = True):
        """
        Train a matrix factorization model to predict empty
        entries in a matrix. The terminology assumes a
        ratings matrix which is ~ user x item

        Params
        ======
        ratings : (ndarray)
            User x Item matrix with corresponding ratings

        n_factors : (int)
            Number of latent factors to use in matrix
            factorization model

        item_reg : (float)
            Regularization term for item latent factors

        user_reg : (float)
            Regularization term for user latent factors

        item_bias_reg : (float)
            Item bias for item latent factors, shape as number of items *1

        user_bias_reg : (float)
            User bias for user latent factors, shape as number of users *1

        max_iter: (int)
            Max number of epoch

        batch_size: (int)
            number of rows input each time to update the gradient

        learning_rate: (float)
            learning rate of latent factors and bias

        tolerance: (float)
            cut-off on iterations when percange change in loss function lower to the value

        verbose : (bool)
            Whether or not to printout training progress
        """

        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.item_reg = item_reg
        self.user_reg = user_reg

        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.learning_rate = learning_rate
        self.sample_row, self.sample_col = self.ratings.nonzero()
        self.n_samples = len(self.sample_row)
        self.batch_size = batch_size
        self._v = verbose
        self.n_iter = max_iter
        self.Loss = []
        self.tolerance = 0-tolerance

        """
        Validate input params.
        """
        if self.n_factors <= 0 or not isinstance(self.n_factors, int):
            raise ValueError("n_factors must be some positive integers. Got %f" % self.n_factors)

        if not isinstance(self._v, bool):
            raise ValueError("verbose must be either True or False")

        if not isinstance(self.n_factors, int):
            raise ValueError("n_factors must be an integer")

        if self.n_iter is not None and (self.n_iter <= 0 or not isinstance(self.n_iter, int)):
            raise ValueError("max_iter must be some positive integers. Got %f" % self.n_iter)

        if self.item_reg < 0.0:
            raise ValueError("item_reg must be >= 0")

        if self.user_reg < 0.0:
            raise ValueError("user_reg must be >= 0")

        if self.user_bias_reg < 0.0:
            raise ValueError("user_bias_reg must be >= 0")

        if self.item_bias_reg < 0.0:
            raise ValueError("item_bias_reg must be >= 0")

        if self.batch_size <= 0 or not isinstance(self.batch_size, int):
            raise ValueError("batch size must be some positive integers. Got %f" % self.batch_size)



    def fit(self):
        """
        Train model
        """
        self.ratings_zero=np.zeros(self.ratings.shape)
        self.user_vecs = np.random.random((self.n_users, self.n_factors))
        self.item_vecs = np.random.random((self.n_items, self.n_factors))

        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])

        ctr = 1
        while ctr <= self.n_iter:
            if ctr % 10 == 0 :
                print ('\tcurrent iteration: {}'.format(ctr))

            if ctr>1:
                # predict the ratings by applying the vectorized prediction forluma
                ratings_pred=self.user_bias[:,np.newaxis]+self.item_bias[np.newaxis,:]+self.global_bias+self.user_vecs.dot(self.item_vecs.T)
                ratings_pred=np.nan_to_num(ratings_pred)
                self.Loss.append(self.get_loss(ratings_pred))
                if self._v:
                    print (self.Loss[-1])

            if len(self.Loss)>1:
                # set the tolerance on the difference in Loss, cut-off the iteration when the difference is less than tolerance
                if self.Loss[-1]<self.Loss[-2] and (self.Loss[-1]-self.Loss[-2])/self.Loss[-2]>self.tolerance:
                    print ('Converge after iterations: {}'.format(ctr))
                    return (self.user_vecs, self.item_vecs, self.user_bias, self.item_bias, self.global_bias, self.Loss)

            self.training_indices = np.arange(self.n_samples)
            np.random.shuffle(self.training_indices)

            for start_idx in range(0, self.n_samples - self.batch_size + 1, self.batch_size):
                idx = self.training_indices[start_idx:start_idx + self.batch_size]
                u = self.sample_row[idx]
                i = self.sample_col[idx]

                # error
                e = [self.ratings[a,b] - self.predict(a,b) for a,b in zip(u,i)]

                # update biases
                self.user_bias[u] += self.learning_rate * (e - self.user_bias_reg * self.user_bias[u])
                self.item_bias[i] += self.learning_rate * (e - self.item_bias_reg * self.item_bias[i])

                # update latent factors
                self.user_vecs[u, :] = [self.user_vecs[u, :][x] + self.learning_rate * (e[x] * self.item_vecs[i, :][x] - self.user_reg * self.user_vecs[u,:][x]) for x in range(self.batch_size)]
                self.item_vecs[i, :] = [self.item_vecs[i, :][x] + self.learning_rate * (e[x] * self.user_vecs[u, :][x] - self.item_reg * self.item_vecs[i,:][x]) for x in range(self.batch_size)]

            ctr += 1


        return (self.user_vecs, self.item_vecs, self.user_bias, self.item_bias, self.global_bias, self.Loss)

    def predict(self, u, i):
        prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
        prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        prediction=np.nan_to_num(prediction)
        return prediction

    def get_loss(self, ratings_pred):

        ratings_predicted = ratings_pred[self.ratings.nonzero()[0].tolist(),self.ratings.nonzero()[1].tolist()].flatten()
        ratings_actual = self.ratings[self.ratings.nonzero()[0].tolist(),self.ratings.nonzero()[1].tolist()].flatten()
        new_loss = np.sum(np.power((ratings_actual-ratings_predicted), 2))+np.sum(np.power\
                                                                        (self.user_bias,2))*self.user_bias_reg+np.sum\
        (np.power(self.item_bias,2))*self.item_bias_reg+np.sum(np.power(self.user_vecs,2))*self.user_reg+np.sum\
        (np.power(self.item_vecs,2))*self.item_reg

        return new_loss



    def get_prediction_and_mse(self, test):
        vecs = self.fit()
        user_vecs = vecs[0]
        item_vecs = vecs[1]
        user_bias = vecs[2]
        item_bias = vecs[3]
        global_bias = vecs[4]
        predictions = user_bias[:,np.newaxis]+item_bias[np.newaxis,:]+global_bias+user_vecs.dot(item_vecs.T)
        mse = get_mse(predictions, test)
        return mse

# testing mse
#MiniBatchRecommendationSGD_tol(train).get_prediction_and_mse(test)
# training loss
#MiniBatchRecommendationSGD_tol(train).fit()[5]


#### time it! ###############
#from datetime import datetime
#startTime = datetime.now()
####################
### run sth here ###
####################
#timeElapsed = datetime.now()-startTime
#print('Time elpased (hh:mm:ss.ms) {}'.format(timeElapsed))
