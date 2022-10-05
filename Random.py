import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt


class Random:
    def __init__(self, n_users, n_items, time_buckets):
        # number of user, items
        self.n_users = n_users
        self.n_items = n_items
        # precision log
        self.cumulative_precision = []
        # reward logs
        self.rewards_log = [0] * time_buckets
        # update counter
        self.a = [1] * time_buckets

    def select_arm(self):
        """
        Make recommendations by randomly selecting among the options
        :return:
        """
        return choice(self.n_items)

    def evaluate_policy(self, item_idx, reward, t):
        """
      Trigger recommendation every period. Only count recomendation if
      the user was actually served the same recommendation. Record precision.
      Li et al. http://proceedings.mlr.press/v26/li12a/li12a.pdf
      """
        # select random arm
        arm = self.select_arm()
        # check for impression
        if arm == item_idx:
            # update average reward log
            self.rewards_log[t] = self.rewards_log[t] + (1 / self.a[t]) * (reward - self.rewards_log[t])
            # update counter
            self.a[t] += 1

    def get_average_ctr(self):
        """
        Return average click-through-rate
        :return:
        """
        return self.rewards_log, self.a

    def plot_cumulative_precision(self):
        """
        Plot precision at each time step
        :return:
        """
        precision = np.array(self.cumulative_precision).reshape(self.n_users, self.T).mean(axis=0)
        plt.plot(np.arange(self.T), precision)
        plt.show()


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
    # initialize random policy
    TS = Random(n_users=n_users, n_items=n_items, time_buckets=T)

    # run experiment
    early_stop = len(ratings)
    i = 0
    for user_idx, item_idx, rating, t in ratings.to_numpy()[:early_stop, :]:
        TS.evaluate_policy(item_idx=int(item_idx), reward=rating, t=int(t))
        results, impressions = TS.get_average_ctr()
        print("Progress", i/len(ratings), 'Period:', t, "Average CTR:", results[int(t)])
        i += 1

    # get results
    avg_rating, impressions = TS.get_average_ctr()
    # save result,format date
    pd.DataFrame(np.array([avg_rating, impressions]).T, columns=['rating', 'impressions']) \
        .to_csv('test_results/random_results_{date}.csv'.format(date=date.today()))

    # plot results
    plt.plot(np.arange(T), np.array(avg_rating))
    plt.ylim((0.4, 1))
    plt.show()
