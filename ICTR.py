from ICTRbase import ICTRbase
from numpy import argmax, ones, zeros, identity, matmul, sqrt
from numpy.linalg import inv
from numpy.random import dirichlet, multivariate_normal, multinomial, normal
from scipy.stats import invgamma, multivariate_normal
import numpy as np


class ICTR:
    """
    Sample using interactive collaborative topic regression
    """
    def __init__(self, n_users, n_items, B, K, time_buckets, policy='TS'):
        # initialize variables
        self.B = B
        self.K = K
        self.policy = policy
        self.time_buckets = time_buckets
        self.n_items = n_items + 1
        self.n_users = n_users + 1

        # average reward per bucket
        self.rewards_log = [0] * time_buckets
        self.a = [1] * time_buckets  # number of impression per bucket
        self.trace = []  # average reward trace

        # initialize hyper-parameters
        self.lambda_p = ones((self.n_users, self.K, self.B)) * 0.1
        self.eta = ones((self.K, self.n_items, self.B))*0.1
        self.mu_q = zeros((self.n_items, self.K, self.B))
        self.cov = [[identity(self.K)]*self.n_items]*self.B
        self.alpha = ones((self.n_items, self.B))
        self.beta = ones((self.n_items, self.B))*0.1
        self.z = zeros((self.n_users, self.K, self.B))  # preference assignment vector 1-of-K
        self.x = zeros((self.n_users, self.n_items, self.B))  # item assignment vector 1-of-n

        # initialize latent vectors
        self.p = zeros((self.n_users, self.K, self.B))
        self.q = zeros((self.n_items, self.K, self.B))

        # variance over reward prediction
        self.sigma_sq = ones((self.n_items, self.B))
        # currently selected user arm
        self.arm = zeros((self.n_users, self.B))

    def initialize_particles(self):
        """
        Randomly initialize B particles
        :return:
        """
        # for each user
        for user_idx in range(self.n_users):
            # draw B particles
            for i in range(self.B):
                # sample latent parameters and latent state parameters
                self.sample(user_idx, i)

    def sample(self, user_idx, i):
        """
        Generate samples using the ith particle of user
        :param user_idx: user index
        :param i: particle index
        :return:
        """
        # draw latent user vector
        self.p[user_idx, :, i] = dirichlet(self.lambda_p[user_idx, :, i])
        # sample latent user preference vector
        self.z[user_idx, :, i] = multinomial(1, self.p[user_idx, :, i])  # z is 1-of-K vector
        k = argmax(self.z[user_idx, :, i])  # get corresponding index
        # draw item mixture corresponding to preference k
        phi = dirichlet(self.eta[k, :, i])
        # draw item vector from preference k
        self.x[user_idx, :, i] = multinomial(1, phi)  # x is 1-of-K vector
        n = argmax(self.x[user_idx, :, i])  # get corresponding arm
        # cache arm selection
        self.arm[user_idx, i] = n
        # draw variance of the noise for reward prediction
        self.sigma_sq[n, i] = invgamma(self.alpha[n, i], self.beta[n, i]).rvs()
        # draw latent item vector
        self.q[n, :, i] = multivariate_normal(self.mu_q[n, :, i], self.sigma_sq[n, i] * self.cov[i][n]).rvs()
        # predicted reward
        # y = normal((self.p[user_idx] * self.q[n]).sum(), self.sigma_sq[n, i])

    def get_weights(self, user_idx, reward):
        """
        Compute particle weights for user given reward
        :param user_idx: index of user currently being served
        :param reward: observed reward
        :param n: selected arm
        :return: normalize particle weights
        """
        # compute weights
        weights = []
        for i in range(self.B):
            # get arm
            n = int(self.arm[user_idx, i])
            # append ith particles weight
            weights.append(
                (multivariate_normal.pdf([reward]*self.K, self.p[user_idx, :, i] * self.q[n, :, i],
                                         self.sigma_sq[n, i] * identity(self.K)) *
                 self.lambda_p[user_idx, :, i] / self.lambda_p[user_idx, :, i].sum() *
                 self.eta[:, n, i] / self.eta[:, n, i].sum()
                ).sum()
            )
        # normalize
        return np.array(weights) / sum(weights)

    def update(self, user_idx, reward):
        """
        Based on observed reward, resample user particles according to likelihood distribution
        :param user_idx:
        :param reward:
        :return:
        """
        # get particle weights
        weights = self.get_weights(user_idx, reward)
        # resample particles
        for i in range(self.B):
            # get arm
            n = self.arm[user_idx, i]
            # sample particle according to weight
            i = argmax(multinomial(1, weights))
            # resample
            self.sample(user_idx, i)

        # update statistics for each particle
        for i in range(self.B):
            # get arm
            n = int(self.arm[user_idx, i])
            # posterior expected latent user vector
            expected_p = reward * self.z[user_idx, :, i] + self.lambda_p[user_idx, :, i]
            # posterior expected latent item vector
            expected_q = reward + self.eta[:, n, i]
            # posterior multinomial parameter for preference vector
            theta = expected_p/expected_q.sum() * expected_p/expected_p.sum()
            # sample posterior latent preference assignment
            self.z[user_idx, :, i] = multinomial(1, theta)

            # update remaining statistics
            p = self.p[user_idx, :, i]
            # update covariance for latent item distribution
            old_cov = self.cov[i][n]
            self.cov[i][n] = inv(inv(old_cov) + matmul(p, p.T))
            # update mean for latent item distribution
            mu = self.mu_q[n, :, i]
            mu_new = matmul(old_cov, (matmul(inv(old_cov), mu) + p*reward))
            self.mu_q[n, :, i] = mu_new
            # update inverse gamma hyper-parameters
            self.alpha[n, i] += 0.5
            self.beta[n, i] += 0.5 * (matmul(matmul(old_cov, mu), mu) + reward**2 - matmul(matmul(old_cov, mu_new), mu_new))
            # update latent user preference dirichlet hyperparameter
            self.lambda_p[user_idx, :, i] += reward * self.z[user_idx, :, i]
            # update latent item mixture hyperparameter
            k = argmax(self.z[user_idx, :, i])
            self.eta[k, :, i] += reward * self.x[user_idx, :, i]

            # sample q, sigma_sq, p and Psi
            self.sample(user_idx, i)

    def eval(self, user_idx, n):
        """
        Predicted reward average over all particles
        :param user_idx: index of currently selected user
        :param n: item index
        :return: average predicted reward
        """
        reward = []
        for i in range(self.B):
            # predict reward
            r = (self.p[user_idx, :, i] * self.q[n, :, i]).sum()
            # append reward for particle i to list
            reward.append(r)
        # return average over all particles
        return np.mean(reward)

    def select_arm(self, user_idx):
        """
        Select arm based on predicted reward and policy
        :param user_idx: index of user currently being served
        :return:
        """
        reward = []
        for n in range(self.n_items):
            # get average predicted reward
            reward.append(self.eval(user_idx, n))

        # choose arm according to TS or USB
        if self.policy == 'TS':
            # select arm
            return argmax(reward)
        elif self.policy == 'UCB':
            # compute variance
            nu = (1/self.B) * (self.sigma_sq**2).sum()
            # select arm
            return argmax(reward + self.gamma*sqrt(nu))
        else:
            raise Exception("Please enter a valid policy: {'TS', 'UCB'}")

    def evaluate_policy(self, user_idx, item_idx, reward, t):
        """
        Replayer method of evaluation
        :param user_idx: user index
        :param item_idx: item index, required for replayer method
        :param reward: observed reward at time t
        :param t: time index
        :return:
        """
        # select arm
        n = self.select_arm(user_idx)
        if n == item_idx:
            # update parameter and states
            self.update(user_idx, reward)
            # update average reward log
            self.rewards_log[t] = self.rewards_log[t] + (1 / self.a[t]) * (reward - self.rewards_log[t])
            # update counter
            self.a[t] += 1
            # append to trace
            self.trace.append(self.rewards_log[t])

    def get_average_rating(self):
        """
        Get average reward and impression log
        :return: average rating at time t
        """
        return self.rewards_log, self.a

    def replay(self, ratings):
        """
        Run experiment on dataset using replayer method
        :param ratings: dataset [user_id, item_id, rating and time bucket]
        :return:
        """
        # run experiment
        i = 0
        for user_idx, item_idx, rating, t in ratings.to_numpy():
            self.evaluate_policy(user_idx=int(user_idx), item_idx=int(item_idx), reward=rating, t=int(t))
            results, impressions = self.get_average_rating()
            print(
                "Progress", np.round(i / len(ratings), 3),
                'Time Bucket:', t,
                "Impressions:", impressions[int(t)],
                "Average Rating:", results[int(t)])
            i += 1

    if __name__ == '__main__':
        import pandas as pd
        from datetime import date

        # get ratings data
        ratings = pd.read_csv('datasets/MovieLens10M_ratings.csv')
        ratings = ratings[['user_id', 'item_id', 'rating2', 'time_bucket']]

        # get model params from data
        T = ratings.time_bucket.nunique()  # get number of unique time buckets
        n_users = ratings.user_id.nunique()  # number of users
        n_items = ratings.item_id.nunique()  # number of items

        # initialize model
        from ICTR import ICTR
        B = 10  # number of particles
        K = 2  # latent parameter dimension
        model = ICTR(n_users=n_users, n_items=n_items, K=K, B=B, time_buckets=T, policy='TS')

        # run experiment
        model.replay(ratings)
