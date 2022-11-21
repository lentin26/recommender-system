import numpy as np
from numpy.random import multivariate_normal
from numpy import argmax, matmul, identity, transpose
from numpy.linalg import inv
import matplotlib.pyplot as plt


class ThompsonSampling:
    """
    Thompson Sampling based recommender system with Gaussian conjugate priors
    """

    def __init__(self, n_users, n_items, dim, sigma, time_buckets, user_offset=None, item_offset=None) -> None:
        # set latent vector dimension
        self.dim = dim
        # use training to update priors
        self.train_ratings = None

        # user-item ratings
        self.n_users = n_users #+ 1
        self.n_items = n_items #+ 1

        # prior distributions on user (p), item (q) and ratings
        self.sigma = sigma
        self.sigma_p = sigma
        self.sigma_q = sigma
        self.lambda_p = self.sigma / self.sigma_p
        self.lambda_q = self.sigma / self.sigma_q
        self.I = identity(self.dim)

        # latent user prior
        self.A = [self.lambda_p * self.I] * self.n_users
        self.b = np.zeros((self.n_users, self.dim))

        # latent item prior
        self.Psi = [self.lambda_p * self.I] * self.n_users
        self.v = np.zeros((self.n_users, self.dim))

        # rating history
        self.ratings = []

        # use data from earlier period to update latent item prior
        if self.train_ratings is not None:
            self.v, self.Psi = self.item_posterior()
        # otherwise, use same prior as latent user vectors
        else:
            self.Psi = [self.lambda_p * self.I] * self.n_items
            self.v = np.zeros((self.n_items, self.dim))

        # store offsets
        if user_offset is None:
            self.user_offset = [0] * n_users
        else:
            self.user_offset = user_offset
            assert len(user_offset) == n_users
        if item_offset is None:
            self.item_offset = [0] * n_items
        else:
            self.item_offset = item_offset
            assert len(item_offset) == n_items

        # cache
        self.sampled_item_cache = np.zeros((self.n_users, self.dim))
        self.sampled_user_cache = np.zeros((self.n_users, self.dim))
        self.recommended_arm = [0] * self.n_users
        # reward logs
        self.rewards_log = [0] * time_buckets
        # record of the number of impressions per time bucket
        self.a = [1] * time_buckets
        # trace
        self.trace = []

    def batch_update_item_prior(self):
        """
        Get posterior normal distribution for latent item vectors
        :return: posterior normal mean, covariance
        """
        # generate latent item vector
        size = self.n_items
        latent_items = multivariate_normal(self.dim * [0], self.sigma_q * self.I, size=size)

        # compute item posterior mean and covariance
        posterior_item_mean = []
        posterior_item_cov = []
        for j in range(self.ratings.shape[1]):
            # users who rated item j
            user_idx = ratings[:, j].dropna().index

            # items with user latent vectors as rows
            B = latent_items[user_idx, :]
            r = ratings[:, j].dropna().values

            # force shape to column vector
            r.shape = (r.shape[0], 1)

            # posterior mean for latent item vector, ratings affect the result
            v = matmul(matmul(inv(matmul(B.T, B) + self.lambda_q * self.I), B.T), r)
            posterior_item_mean.append(v)

            # posterior covariance for latent item vector, number of ratings reduce uncertainty
            Psi = inv(matmul(B.T, B) + self.lambda_q * self.I) * self.sigma
            posterior_item_cov.append(Psi)

            return v, Psi

    def select_arm(self, user_idx):
        """
        Select arm for user i and time t
        Return: arm
        """
        # get priors from previous period
        A = self.A[user_idx]
        b = self.b[user_idx]

        # estimate posterior mean and covariance
        mu = matmul(b, inv(A).T)
        cov = inv(A) * (self.sigma ** 2)

        # fetch offset
        theta_p = self.user_offset[user_idx]

        # sample latent user, item vector from posterior
        reward = []
        q_samples = []
        p = multivariate_normal(mu + theta_p, cov)
        for i in range(self.n_items):
            theta_q = self.item_offset[i]
            q = multivariate_normal(self.v[i] + theta_q, self.Psi[i])
            q_samples.append(q)
            reward.append((p * q).sum())

        # select arm
        arm = argmax(reward)

        # update cache
        self.recommended_arm[user_idx] = arm
        self.sampled_item_cache[user_idx] = q_samples[arm]
        self.sampled_user_cache[user_idx] = p

        return arm

    def update_prior(self, user_idx, reward):
        """
        Recieve reward and update prior
        """
        # pull cached arm and sample latent item vector for user i
        q = self.sampled_item_cache[user_idx].reshape(-1, 1)  # make column vector
        p = self.sampled_user_cache[user_idx].reshape(-1, 1)

        # update prior latent users params
        self.A[user_idx] = self.A[user_idx] + matmul(q, q.T)
        self.b[user_idx] = self.b[user_idx] + reward * q.reshape(1, -1)

        # update prior latent users params
        item_idx = self.recommended_arm[user_idx]
        self.Psi[item_idx] = self.Psi[item_idx] + matmul(p, p.T)
        self.v[item_idx] = self.v[item_idx] + reward * p.reshape(1, -1)

    def evaluate_policy(self, user_idx, item_idx, reward, t):
        """
        Recommendation for a user at time t. Replayer method: only count as impression if
        the user was actually served the same recommendation.
        Li et al. http://proceedings.mlr.press/v26/li12a/li12a.pdf
        """
        # select arm
        arm = self.select_arm(user_idx)
        # check for impression
        if arm == item_idx:
            # receive reward, update prior
            self.update_prior(user_idx, reward)
            # update average reward log
            self.rewards_log[t] = self.rewards_log[t] + (1 / self.a[t]) * (reward - self.rewards_log[t])
            # update counter
            self.a[t] += 1
            # append to trace
            self.trace.append(self.rewards_log[t])

    def get_average_ctr(self):
        """
        Return average click-through-rate
        :return:
        """
        return self.rewards_log, self.a

    def get_trace(self):
        """
        Return trace
        :return:
        """
        return self.trace


if __name__ == '__main__':
    import pandas as pd
    from datetime import date

    # get ratings data
    ratings = pd.read_csv('datasets/MovieLens10M_ratings.csv')
    ratings = ratings[['user_id', 'item_id', 'rating2', 'time_bucket']]

    # get number of unique time buckets
    T = ratings.time_bucket.nunique()
    # number of users
    n_users = ratings.user_id.nunique()
    # number of items
    n_items = ratings.item_id.nunique()
    # initialize Thompson Sampling
    TS = ThompsonSampling(n_users=n_users, n_items=n_items, dim=2, sigma=0.02, time_buckets=T)

    # run experiment
    early_stop = len(ratings)
    i = 0
    for user_idx, item_idx, rating, t in ratings.to_numpy()[:early_stop, :]:
        TS.evaluate_policy(user_idx=int(user_idx), item_idx=int(item_idx), reward=rating, t=int(t))
        results, impressions = TS.get_average_ctr()
        print("Progress", i/len(ratings), 'Period:', t, "Average CTR:", results[int(t)])
        i += 1

    # get results
    avg_rating, impressions = TS.get_average_ctr()
    trace = TS.get_trace()
    # save result,format date
    pd.DataFrame(np.array([avg_rating, impressions]).T, columns=['rating', 'impressions']) \
        .to_csv('test_results/ts_results2_{date}.csv'.format(date=date.today()))
    # save trace
    pd.DataFrame(np.array(trace), columns=['trace'])\
        .to_csv('test_results/ts_trace2_{date}.csv'.format(date=date.today()))

    # plot results
    plt.plot(np.arange(T), avg_rating, index=False)
    plt.show()
