from numpy import argmax, ones, zeros, sqrt
from numpy.random import dirichlet, multivariate_normal, multinomial
from scipy.stats import norm


class ICTRbase:
    """
    Interactive Collaborative Filtering
    Wang, Qing, et al. "Online interactive collaborative filtering using multi-armed bandit with dependent arms."
    IEEE Transactions on Knowledge and Data Engineering 31.8 (2018): 1569-1580.
    """
    def __init__(self, n_users, n_items, policy, B, K, time_buckets):
        """

        :param n_users:
        :param n_items:
        :param policy: 'TS' or 'UCB'
        :param B: Number of particles
        :param K: number of latent indices
        :param time_buckets:
        """
        # TS or USB
        self.policy = policy
        # number of latent indices z
        self.K = K
        # number of particles
        self.B = B
        # list containing particle weights
        self.weight = ones(self.B)
        # gamma
        self.gamma = 1
        # user-item ratings
        self.n_users = n_users + 1
        self.n_items = n_items + 1

        # rating history
        self.ratings = []
        # reward log
        self.rewards_log = [0] * time_buckets
        # record of the number of impressions per time bucket
        self.a = [1] * time_buckets
        # trace
        self.trace = []

    def evaluate(self, user_idx, item_idx):
        """
        Get latent vectors, predict ith reward r^i_{m, t}
        :param user_idx:
        :param item_idx:
        :return:
        """
        # iterate on each particle
        predicted_reward = []
        for i in range(self.B):
            # sample latent user vector p
            p = dirichlet(self.lamb)
            # sample latent item vector q
            q = multivariate_normal(self.mu, self.sigma*self.cov)
            # get predicted reward
            predicted_reward.append((p*q).sum())

    def select_arm_(self, user_idx, item_idx):
        """

        :param user_idx:
        :param item_idx:
        :return:
        """
        if self.policy == 'TS':
            # select arm
            return argmax(self.evaluate(user_idx, item_idx))
        elif self.policy == 'UCB':
            # calculate variance
            nu = (1/self.B) * (self.sigma**2).sum()
            # select arm
            return argmax(self.evaluate(user_idx, item_idx) + self.gamma*sqrt(nu))

    def update_sufficient_stats(self, user_idx, item_idx, reward):
        # indicator
        I = ((self.user_preference_assignment[user_idx, :]).astype(int) * reward + self.lamb) /\
            (self.user_preference_assignment[user_idx, :] == k)
        # update sufficient statistics for z_{m,t}
        s1 = (reward + self.lamb)/(reward + self.lamb).sum()
        s2 = (reward + self.eta)/(reward + self.eta).sum()
        # sample z_{m,t} from multinomial parameter
        z = multinomial(s1*s2/(s1*s2).sum())

    def compute_particle_weight(self, i, user_idx, item_idx, reward):
        """

        :param i: particle index
        :return:
        """
        # pull cached latent user and item vectors for user i
        q = self.sampled_item_cache[user_idx]
        p = self.sampled_user_cache[user_idx]
        for z in range(self.K):
            self.weight[i] += norm.pdf((p*q).sum(), self.sigma**2)

        # resample particles according to their weights
        self.weight = self.weight/self.weight.sum() # normalize
        resample_particles = multinomial(self.weight)

        # iterate through particles
        for i in range(self.B):
            # update sufficient statistics
            self.update_sufficient_stats()
            # sample latent index z

    def update_(self, user_idx, item_idx, reward):
        """

        :param user_idx: user index
        :param item_idx: item index
        :param reward: observed reward
        :return:
        """
        # iterate through particles
        for i in range(self.B):
            # compute weight for particle i
            p[i] = self.compute_particle_weight(i, reward)

    def evaluate_policy_(self, user_idx, item_idx, reward, t):
        """
        Replayer method of evaluation
        :param user_idx: user index
        :param item_idx: item index, required for replayer method
        :param reward: observed reward at time t
        :param t: time index
        :return:
        """
        # select arm
        arm = self.select_arm(user_idx)
        if arm == item_idx:
            # receive reward, update prior
            self.update_prior(user_idx, reward)
            # update average reward log
            self.rewards_log[t] = self.rewards_log[t] + (1 / self.a[t]) * (reward - self.rewards_log[t])
            # update counter
            self.a[t] += 1
            # append to trace
            self.trace.append(self.rewards_log[t])



