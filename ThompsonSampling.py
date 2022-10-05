import numpy as np
from numpy.random import multivariate_normal
from numpy import argmax, matmul, identity, transpose
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pandas as pd


class ThompsonSampling:
    """
    Interactive CF Thompson Sampling algorithm from Zhao, Xiaoxue, Weinan Zhang, and Jun Wang. "Interactive
    collaborative filtering." Proceedings of the 22nd ACM international conference on Information & Knowledge
    Management. 2013.
    """
    def __init__(self, observational_data, lam, n, T):
        """
        :param
            observational_data: user, item rating matrix
            n: dimension of the latent feature vectors
        """
        self.observational_data = observational_data
        self.users = observational_data.users.unique()
        self.n_users = observational_data.shape[0]
        self.n_items = observational_data.shape[1]

        # initialize parameters
        self.b = [np.zeros(n)]*self.n_users
        self.A = [lam*identity(n)]*self.n_users
        self.T = T

        # user multinomial params
        self.item_mean = [np.zeros(n)]*self.n_items
        self.item_cov = [identity(n)]*self.n_items
        self.sigma = 1

        # save samples
        self.p = [0]*self.n_users
        self.q = [0]*self.n_users
        self.arm = [0]*self.n_users

        self.cumulative_precision = []

    def item_posterior(self):
        """
        Item feature vector is normally distributed with mean and covariance:
        v = (B'B + l_q I)^(-1)B'r
        Psi = (B'B + l_q I)^(-1) sigma^2
        B is the observational matrix for the item, each row of which is the feature vectors of the users which have
        rated the item.
        :return:
        """
        # compute required matrix once
        data = self.observational_data
        for j in range(10):
            # select all users which rated item j
            data[:, j].dropna(axis=0)

    def make_recommendation(self, user_idx):
        """
        Sample latent user and item vectors from multinomial distributions, select arm which maximizes their inner
        product (i.e., estimated reward). Assume that the conditional probability distribution of rating follows a
        Gaussian distribution: Pr(r_{u,i}|p_u^Tq_i, s^2) = N(r_{u,i}|p_u^Tq_i, s^2)
        :return: recommended arm
        """
        # get user hyper-parameters
        A = self.A[user_idx]
        b = self.b[user_idx]

        # invert A
        inv_A = inv(A)
        # estimate user mean and covariance
        user_mean = matmul(inv_A, b)
        user_cov = inv_A*self.sigma**2
        # sample latent user vector
        p = multivariate_normal(user_mean, user_cov)
        # sample latent item vector i for i in {1, 2, ... , N}
        est_reward = []
        item_vectors = []
        for i, j in zip(self.item_mean, self.item_cov):
            q = multivariate_normal(i, j)
            item_vectors.append(q)
            est_reward.append(matmul(p, q))
        # select arm
        i = argmax(est_reward)

        # save samples
        self.p[user_idx] = p
        self.q[user_idx] = item_vectors[i]
        self.arm[user_idx] = i

        # return arm
        return i

    def receive_reward(self, user_idx):
        """
        Receive reward from user following recommendation (i.e., periodically retrain), update params
        Use Replayer method from Lihong Li, Wei Chu, John Langford, Taesup Moon, and Xuanhui Wang. 2012.
        Only consider recommendation as an impression if the recommendation corresponds to what was actually
        observed in the historical logs
        :return: 1 if recommendation was good, 0 otherwise
        """
        i = self.arm[user_idx]
        # observe reward for user
        reward = self.observational_data[user_idx, i]
        q = self.q[user_idx]
        self.A = self.A + matmul(q, transpose(q))
        self.b = self.b + reward*q

        # a "good" recommendation is considered as any with a rating of no less than four
        return int(reward >= 4)

    def replayer_evaluation(self):
        """

        :return:
        """


    def train(self):
        for t in range(self.T):
            # make recommendation for each user, observe reward
            for user_idx in self.users:
                # make recommendation
                self.make_recommendation(user_idx)
                # observe reward, if the recommendation was good
                true_pos = self.receive_reward(user_idx)
                # record result
                self.cumulative_precision.append(true_pos)

    def get_cumulative_precision(self):
        """
        After training, return cumulative precision, fraction of recommendations that were good
        :return:
        """
        return np.sum(self.cumulative_precision)/self.n_users

    def plot_cumulative_precision(self):
        """
        Plot precision at each time step
        :return:
        """
        precision = np.array(self.cumulative_precision).reshape(self.n_users, self.T).mean(axis=0)
        plt.plot(np.arange(self.T), precision)
        plt.show()


if __name__ == '__main__':
    # synthesize toy rating matrix, 100 users, 10 items/arms
    R = np.random.randint(0, 2, size=1000).reshape(100, 10)
    # initialize TS object and train
    ts = ThompsonSampling(observational_data=R, lam=1, n=10, T=10)
    ts.train()
    # get average number of hits per user
    print(ts.get_cumulative_precision())
    ts.plot_cumulative_precision()


