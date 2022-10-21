from numpy import argmax, ones, zeros, identity, matmul, sqrt
from numpy.linalg import inv
from numpy.random import dirichlet, multivariate_normal, multinomial, normal
from scipy.stats import invgamma, multivariate_normal
import numpy as np


class ICTR2:
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
        self.recommended_arm = 0  # use to print recommended arm

        # initialize hyper-parameters
        self.lambda_p = ones((self.K, self.B))/self.K
        self.eta = ones((self.n_items, self.B))/self.n_items
        self.mu_q = zeros((self.n_items, self.K, self.B))
        self.cov = [[identity(self.K)]*self.n_items]*self.B
        self.Phi = zeros((self.K, self.n_items, self.B))
        self.alpha = 1
        self.beta = 1  # *0.1
        # assignments
        self.z = zeros((self.n_users, self.K, self.B))  # preference assignment vector 1-of-K
        self.x = zeros((self.n_users, self.n_items, self.B))  # item assignment vector 1-of-n

        # initialize latent vectors
        self.p = zeros((self.n_users, self.K, self.B))
        self.q = zeros((self.n_items, self.K, self.B))

        # counts
        self.user_preference_count = zeros((self.n_users, self.K))
        self.preference_item_count = zeros((self.K, self.n_items))

        # variance over reward prediction
        self.sigma_sq = ones((self.n_items, self.B))
        # currently selected user arm
        self.arm = zeros((self.n_users, self.B))

    def initialize(self):
        """
        Initialize latent user and item vectors for each particle
        :return:
        """
        print('Initializing . . . ')
        for i in range(self.B):
            # draw latent user vector
            self.p[:, :, i] = dirichlet(self.lambda_p[:, i], size=self.n_users)
            # draw latent item vector
            cov = self.sigma_sq[0, i] * self.cov[i][0]
            self.q[:, :, i] = multivariate_normal(self.mu_q[0, :, i], cov).rvs(size=self.n_items)
            # draw phi
            self.Phi[:, :, i] = dirichlet(self.eta[:, i], size=self.K)
            for user_idx in range(self.n_users):
                # sample latent user preference vector
                self.z[user_idx, :, i] = multinomial(1, self.p[user_idx, :, i])  # z is 1-of-K vector
                # draw item vector from preference k
                k = argmax(self.z[user_idx, :, i])
                self.x[user_idx, :, i] = multinomial(1, self.Phi[k, :, i])  # x is 1-of-K vector
        print('Done.')

    def sample(self, user_idx, i, j):
        """
        Generate samples using the ith particle of user
        :param user_idx: user index
        :param i: particle index
        :return:
        """
        # draw latent user vector
        self.p[user_idx, :, i] = dirichlet(self.lambda_p[ :, j])
        # sample latent user preference vector
        self.z[user_idx, :, i] = multinomial(1, self.p[user_idx, :, j])  # z is 1-of-K vector
        k = argmax(self.z[user_idx, :, j])  # get corresponding index
        # draw item mixture corresponding to preference k
        phi = dirichlet(self.eta[:, j], size=self.K)
        # draw item vector from preference k
        self.x[user_idx, :, i] = multinomial(1, phi[k, :])  # x is 1-of-K vector
        n = argmax(self.x[user_idx, :, i])  # get corresponding arm
        # cache arm selection
        self.arm[user_idx, i] = n
        # draw variance of the noise for reward prediction
        self.sigma_sq[n, i] = invgamma(self.alpha, self.beta).rvs()
        # draw latent item vector
        self.q[n, :, i] = multivariate_normal(self.mu_q[n, :, j], self.sigma_sq[n, j] * self.cov[i][n]).rvs()

    def sample_posterior(self, user_idx, n, i, j):
        # draw variance of the noise for reward prediction
        self.sigma_sq[:, i] = invgamma(self.alpha, self.beta).rvs(size=self.n_items)
        # draw latent item vector
        self.q[n, :, i] = multivariate_normal(self.mu_q[n, :, j], self.sigma_sq[n, j] * self.cov[i][n]).rvs()
        # draw latent user vector
        self.p[user_idx, :, i] = dirichlet(self.lambda_p[:, j])
        # draw item mixture corresponding to preference k
        self.Phi[:, :, i] = dirichlet(self.eta[:, j], size=self.K)

    def get_weights(self, user_idx, n, reward):
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
            # append ith particles weight
            w = 0
            for k in range(self.K):
                # evaluate data fit
                w_ = multivariate_normal.pdf(reward, self.p[user_idx, k, i] * self.q[n, k, i], self.sigma_sq[n, i])
                w_ *= self.lambda_p[k, i] / self.lambda_p[:, i].sum()
                w_ *= self.eta[n, i] / self.eta[:, i].sum()
                # sum contributions
                w += w_
            # append weight
            weights.append(w)

        # normalize
        return np.array(weights) / sum(weights)

    def update(self, user_idx, n, reward):
        """
        Based on observed reward, resample user particles according to likelihood distribution
        :param user_idx:
        :param reward: observed reward
        :param n: recommended arm
        :return:
        """
        # get particle weights
        weights = self.get_weights(user_idx, n, reward)
        # resample particles
        for i in range(self.B):
            # sample particle according to weight
            j = argmax(multinomial(1, weights))
            # resample particle
            # self.sample_posterior(user_idx, n, i, j)
            self.sample(user_idx, i, j)

        # update statistics for each particle
        for i in range(self.B):
            # update count
            k = int(argmax(self.z[user_idx, :, i]))
            self.user_preference_count[user_idx, k] = reward  # reward in {0, 1}
            self.preference_item_count[k, n] = reward  # reward in {0, 1}
            # posterior expected latent user vector
            expected_p = self.user_preference_count[user_idx, :]
            # posterior expected latent item vector
            expected_phi = self.preference_item_count[:, n]
            # posterior multinomial parameter for preference vector
            theta = expected_p/expected_p.sum() * expected_phi/expected_phi.sum()
            # sample posterior latent preference assignment
            self.z[user_idx, :, i] = multinomial(1, theta)

            # update remaining statistics
            p = self.p[user_idx, :, i]
            # update covariance for latent item distribution
            old_cov = self.cov[i][n]
            new_cov = inv(inv(old_cov) + np.outer(p, p.T))
            self.cov[i][n] = new_cov
            # update mean for latent item distribution
            mu = self.mu_q[n, :, i]
            mu_new = matmul(new_cov, (matmul(inv(old_cov), mu) + p*reward))
            self.mu_q[n, :, i] = mu_new
            # update inverse gamma hyper-parameters
            self.alpha += 0.5
            d_old = matmul(matmul(inv(old_cov), mu), mu)
            d_new = matmul(matmul(inv(new_cov), mu_new), mu_new)
            self.beta += 0.5 * (d_old + reward**2 - d_new)
            # update latent user preference dirichlet hyperparameter
            self.lambda_p[:, i] += reward * self.z[user_idx, :, i]
            # update latent item mixture hyperparameter
            # k = argmax(self.z[user_idx, :, i])
            self.eta[:, i] += reward * self.x[user_idx, :, i]

            # sample q, sigma_sq, p and Psi
            self.sample_posterior(user_idx, n, i, i)
            # self.sample(user_idx, i, i)

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
            n = argmax(reward)
            self.recommended_arm = n
            return n
        elif self.policy == 'UCB':
            # compute variance
            nu = (1/self.B) * (self.sigma_sq**2).sum()
            # select arm
            gamma = 1
            n = argmax(reward + gamma*sqrt(nu))
            self.recommend_arm = n
            return n
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
            self.update(user_idx, n, reward)
            # update average reward log for time bucket t
            self.rewards_log[t] = self.rewards_log[t] + (1 / self.a[t]) * (reward - self.rewards_log[t])
            # update impression count for time bucket t
            self.a[t] += 1
            # append trace
            self.trace.append(self.rewards_log[t])

    def replay(self, ratings):
        """
        Run experiment on dataset using replayer method
        :param ratings: dataset [user_id, item_id, rating and time bucket]
        :return:
        """
        # initialize
        self.initialize()
        # run experiment
        i = 0
        for user_idx, item_idx, rating, t in ratings.to_numpy():
            self.evaluate_policy(user_idx=int(user_idx), item_idx=int(item_idx), reward=rating, t=int(t))
            results, impressions = self.get_results()
            print(
                "Progress", np.round(i / len(ratings), 3),
                'Time Bucket:', int(t),
                "Impressions:", impressions[int(t)] - 1,
                "Selected Arm:", self.recommended_arm,
                "Average Rating:", results[int(t)])
            i += 1

    def get_results(self):
        """
        Get average reward and impression log
        :return: average rating at time t
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

    # get model params from data
    T = ratings.time_bucket.nunique()  # get number of unique time buckets
    n_users = ratings.user_id.nunique()  # number of users
    n_items = ratings.item_id.nunique()  # number of items

    # initialize model
    B = 10  # number of particles
    K = 3  # latent parameter dimension
    model = ICTR2(n_users=n_users, n_items=n_items, K=K, B=B, time_buckets=T, policy='TS')

    # run experiment
    model.replay(ratings)

    # get results
    avg_rating, impressions = model.get_results()
    trace = model.get_trace()

    # save result,format date
    pd.DataFrame(np.array([avg_rating, impressions]).T, columns=['rating', 'impressions']) \
        .to_csv('test_results/ictr_results_{date}.csv'.format(date=date.today()))
    # save trace
    pd.DataFrame(np.array(trace), columns=['trace']) \
        .to_csv('test_results/ictr_trace_{date}.csv'.format(date=date.today()))