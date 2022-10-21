from numpy import identity, zeros, outer, matmul, argmax
from numpy.linalg import norm, inv
import numpy as np
from scipy.stats import multivariate_normal

class UCB:
    """
    Upper Confidence Bound (UCB) algorithm
    """
    def __init__(self, n_lat, alpha):
        self.n_lat = n_lat
        self.alpha = alpha
        self.n_items = None
        self.n_users = None
        self.item_ratings = None
        self.rewards_log = None  # average reward per bucket
        self.a = None  # number of impression per bucket
        self.reward_trace = []  # average reward trace
        self.arm_trace = []  # arm selection trace

        # prior distributions on user (p), item (q) and ratings
        self.sigma = 0.2
        self.sigma_p = 0.2
        self.sigma_q = 0.2
        self.lambda_p = self.sigma / self.sigma_p
        self.lambda_q = self.sigma / self.sigma_q
        self.I = identity(self.n_lat)

        # latent user, item priors
        self.A = None  # user covariance
        self.b = None  # user mean
        self.Psi = None  # item covariance
        self.v = None  # item mean

        # currently selected arm
        self.recommended_arm = None

    def update(self, user_id, item_id, reward):
        """
        Update parameters
        :param user_id: ID of user currently being served
        :param item_id: item ID of selected arm
        :param reward: observed reward
        :return:
        """
        self.A[user_id] += outer(self.v[item_id], self.v[item_id])
        self.b[user_id] += reward * self.v[item_id]

        # get posterior item feature vectors
        mu = matmul(inv(self.A[user_id]), self.b[user_id])
        item_id = self.recommended_arm
        self.v[item_id] += reward * mu

    def select_arm(self, user_id):
        """
        Select arm based on predicted reward and policy
        :param user_id: index of user currently being served
        :return:
        """
        mu = matmul(inv(self.A[user_id]), self.b[user_id])
        Sigma = inv(self.A[user_id])*self.sigma**2
        reward = []
        for i in range(self.n_items):
            # get average predicted reward over particles
            reward.append(matmul(mu, self.v[i]) + self.alpha * np.sqrt(matmul(matmul(self.v[i], inv(Sigma)), self.v[i])))
        self.recommended_arm = argmax(reward)
        return argmax(reward)

    def evaluate_policy(self, user_id, item_id, reward, t):
        """
        Replayer method of evaluation
        :param user_id: user index
        :param item_id: item index, required for replayer method
        :param reward: observed reward at time t
        :param t: time index
        :return:
        """
        # select arm
        n = self.select_arm(user_id)
        if n == item_id:
            # update parameter and states
            self.update(user_id, n, reward)
            # update average reward log for time bucket t
            self.rewards_log[t] += (1 / self.a[t]) * (reward - self.rewards_log[t])
            # update impression count for time bucket t
            self.a[t] += 1
            # append trace
            self.reward_trace.append(self.rewards_log[t])
            self.arm_trace.append(n)
            self.recommended_arm = n

    def replay(self, ratings, time_buckets):
        """
        Run experiment on dataset using replayer method
        :param ratings: dataset [user_id, item_id, rating and time bucket]
        :param time_buckets: number of time buckets
        :return:
        """
        # get number of users, items
        self.n_users = ratings.user_id.nunique()  # number of users
        self.n_items = ratings.item_id.nunique()  # number of items

        self.rewards_log = [0] * time_buckets
        self.a = [1] * time_buckets  # number of impression per bucket
        self.item_ratings = [0] * self.n_items

        # latent user prior
        self.A = [self.lambda_p * self.I] * self.n_users
        self.b = zeros((self.n_users, self.n_lat))
        self.b = multivariate_normal([0]*self.n_lat, identity(self.n_lat)).rvs(size=self.n_users)

        # latent item prior
        self.Psi = [self.lambda_p * self.I] * self.n_users
        # self.v = zeros((self.n_items, self.n_lat))
        # randomly intitialize v
        self.v = multivariate_normal([0]*self.n_lat, identity(self.n_lat)).rvs(size=self.n_items)

        # run experiment
        i = 0
        for user_id, item_id, rating, t in ratings.to_numpy():
            self.evaluate_policy(user_id=int(user_id), item_id=int(item_id), reward=rating, t=int(t))
            results, impressions = self.get_results()
            print(
                "Progress", np.round(i / len(ratings), 3),
                'Time Bucket:', int(t),
                "Impressions:", impressions[int(t)] - 1,
                "User ID:", int(user_id),
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
        return self.reward_trace, self.arm_trace


if __name__ == '__main__':
    import pandas as pd
    from datetime import date

    # get ratings data
    ratings = pd.read_csv('datasets/MovieLens10M_ratings.csv')
    ratings = ratings[['user_id', 'item_id', 'rating2', 'time_bucket']]

    # get model params from data
    T = ratings.time_bucket.nunique()  # get number of unique time buckets

    # initialize model
    model = UCB(n_lat=2, alpha=0.1)

    # run experiment
    model.replay(ratings, time_buckets=T)

    # get results
    avg_rating, impressions = model.get_results()
    reward_trace, arm_trace = model.get_trace()

    # save result,format date
    pd.DataFrame(np.array([avg_rating, impressions]).T, columns=['rating', 'impressions']) \
        .to_csv('test_results/ucb_results_{date}.csv'.format(date=date.today()))
    # save trace
    pd.DataFrame(np.array([reward_trace, arm_trace]), columns=['trace']) \
        .to_csv('test_results/ucb_trace_{date}.csv'.format(date=date.today()))