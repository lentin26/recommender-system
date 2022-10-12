import copy
from numpy import argmax, ones, zeros, identity, matmul, sqrt
from numpy.linalg import inv
from numpy.random import dirichlet, multivariate_normal, multinomial, choice
from scipy.stats import invgamma, norm, multivariate_normal
import numpy as np
# from tqdm import tqdm


class Particle:
    """
    A particle is a container that maintains the current status information for both user m and item x_{m,t}
    """
    def __init__(self, n_users: int, n_items: int, n_lat: int):
        # number of users, items, and latent dimensions
        self.n_users = n_users
        self.n_items = n_items
        self.n_lat = n_lat
        # dirichlet hyperparameter, preference component for users
        self.lambda_p = ones(n_lat)
        self.eta = ones((n_lat, n_items))
        self.Phi = [dirichlet(self.eta[i, :]) for i in range(n_lat)]
        # variance inverse gaussian hyper-parameters
        self.alpha = 1.
        self.beta = 1.
        # mean, variance, covariance for items
        self.mu_q = zeros((n_items, n_lat))
        self.sigma_sq = [invgamma(self.alpha, self.beta).rvs() for _ in range(n_items)]
        self.cov = [identity(self.n_lat)] * self.n_items
        # user (p) and item vectors (q)
        self.p = dirichlet(self.lambda_p, size=self.n_users)
        self.q = [
            multivariate_normal(self.mu_q[item], self.sigma_sq[item] * identity(self.n_lat)).rvs()
            for item in range(n_items)
        ]
        self.z = [multinomial(1, pvals=self.p[user_id]) for user_id in range(n_users)]

    def p_expectation(self, topic=None, reward=None):
        computed_sum = np.sum(self.lambda_p)
        user_lambda = np.copy(self.lambda_p)
        if reward is not None:
            user_lambda[topic] += reward
            computed_sum += reward
        return user_lambda / computed_sum

    def phi_expectation(self, item_id, reward=None):
        computed_sum = np.sum(self.eta, axis=1)
        item_eta = np.copy(self.eta[:, item_id])
        if reward is not None:
            item_eta += reward
            computed_sum += reward
        return item_eta / computed_sum

    def get_weight(self, user_id, item_id, reward):
        norm_val = norm(matmul(self.p[user_id], self.q[item_id]), self.sigma_sq[item_id]).pdf(reward)
        # before updating any parameters, resampling is performed based on particle weights
        return np.sum(norm_val * self.p_expectation(user_id) * self.phi_expectation(item_id))

    def get_theta(self, user_id, item_id, reward, topic):
        p_exp = self.p_expectation(topic=topic, reward=reward)
        phi_exp = self.phi_expectation(item_id, reward=reward)
        return p_exp * phi_exp

    def select_z_topic(self, user_id, item_id, reward):
        topic = argmax(multinomial(1, pvals=self.p[user_id]))
        # topic = argmax(multinomial(1, [1 / self.n_lat]) * self.n_lat)
        theta = self.get_theta(user_id, item_id, reward, topic)
        theta = theta / np.sum(theta)
        return np.argmax(multinomial(1, theta))

    def update_parameters(self, user_id, item_id, reward, topic):
        # get user latent preference vector
        p = np.copy(self.p[user_id])

        # update covariance
        old_cov = np.copy(self.cov[item_id])
        new_cov = inv(inv(old_cov) + np.outer(p, p))  # item updated covariance

        # update mean for latent item distribution
        mu = np.copy(self.mu_q[item_id])
        new_mu = matmul(new_cov, (matmul(inv(old_cov), mu) + p * reward))

        # update inverse gamma hyper-parameters
        d_old = matmul(matmul(inv(old_cov), mu), mu)
        d_new = matmul(matmul(inv(new_cov), new_mu), new_mu)

        # perform update
        self.cov[item_id] = new_cov
        self.mu_q[item_id, :] = new_mu
        self.alpha += 0.5
        self.beta += 0.5 * (d_old + reward ** 2 - d_new)
        self.lambda_p[topic] += reward
        self.eta[:, item_id] += reward

    def sample_random_variables(self, user_id, item_id, topic):
        # draw variance of the noise for reward prediction
        self.sigma_sq[item_id] = invgamma(self.alpha, self.beta).rvs()
        # draw latent item vector
        self.q[item_id] = multivariate_normal(self.mu_q[item_id], self.sigma_sq[item_id] * self.cov[item_id]).rvs()
        # draw latent user vector
        self.p[user_id] = dirichlet(self.lambda_p)
        # draw item mixture corresponding to preference k
        self.Phi[topic] = np.array([dirichlet(self.eta[topic]) for topic in range(self.n_lat)])


class ICTR(Particle):
    """
    Sample using interactive collaborative topic regression

    References
    ----------
    .. [1] Wang, Qing, et al. "Online interactive collaborative filtering using multi-armed
       bandit with dependent arms." IEEE Transactions on Knowledge and Data Engineering 31.8 (2018): 1569-1580.
    """
    def __init__(self, n_lat, n_particles, time_buckets, policy='TS'):
        # initialize variables
        self.n_items = None
        self.n_users = None
        self.particles = None
        self.n_lat = n_lat
        self.n_particles = n_particles
        self.policy = policy
        self.time_buckets = time_buckets

        # average reward per bucket
        self.rewards_log = [0] * time_buckets
        self.a = [1] * time_buckets  # number of impression per bucket
        self.trace = []  # average reward trace
        self.arm_trace = []  # trace of arm recommendations
        self.recommended_arm = 0  # use to print recommended arm

        # currently selected user arm
        # self.arm = zeros((self.n_users, self.n_particles))

    def get_weights(self, user_id, item_id, reward):
        """
        Get particle weights, proportional to likelihood function
        :param user_id:
        :param item_id:
        :param reward:
        :return:
        """
        weights = []
        # iterate through particles
        for particle in self.particles:
            weights.append(particle.get_weight(user_id, item_id, reward))

        # weights = np.exp(weights - np.max(weights)) / np.sum(np.exp(weights - np.max(weights)))
        return weights / np.sum(weights)  # replace with softmax?

    def update(self, user_id, item_id, reward):
        """
        Based on observed reward, resample user particles according to likelihood distribution
        :param user_id:
        :param reward: observed reward
        :param item_id: recommended arm
        :return:
        """
        # get particle weights
        weights = self.get_weights(user_id, item_id, reward)
        # resample particles according to weight
        ds = choice(range(self.n_particles), p=weights, size=self.n_particles)
        # update particle list
        new_particles = [copy.deepcopy(self.particles[i]) for i in ds]
        self.particles = new_particles

        # for i in range(self.n_particles):
        #     self.particles[i].p[user_id] = new_particles[i].p[user_id]
        #     self.particles[i].q[item_id] = new_particles[i].q[item_id]
        #     self.particles[i].Phi = new_particles[i].Phi
        #     self.particles[i].sigma_sq[item_id] = new_particles[i].sigma_sq[item_id]
        #     self.particles[i].lambda_p = new_particles[i].lambda_p
        #     self.particles[i].alpha = new_particles[i].alpha
        #     self.particles[i].beta = new_particles[i].beta
        #     self.particles[i].eta[:, item_id] = new_particles[i].eta[:, item_id]
        #     self.particles[i].mu_q[item_id] = new_particles[i].mu_q[item_id]
        #     self.particles[i].cov[item_id] = new_particles[i].cov[item_id]

        # update statistics for each particle
        for particle in self.particles:
            topic = particle.select_z_topic(user_id, item_id, reward)
            particle.update_parameters(user_id, item_id, reward, topic)
            particle.sample_random_variables(user_id, item_id, topic)

    def eval(self, user_id, item_id):
        """
        Predicted reward average over all particles
        :param user_id: index of currently selected user
        :param item_id: item index
        :return: average predicted reward
        """
        reward = []
        for particle in self.particles:
            # predict reward for ith particle
            r = matmul(particle.p[user_id], particle.q[item_id])
            # append reward for particle i to list
            reward.append(r)
        # return average over all particles
        return np.mean(reward)

    def select_arm(self, user_id):
        """
        Select arm based on predicted reward and policy
        :param user_id: index of user currently being served
        :return:
        """
        # get average reward over particles for each arm
        reward = []
        for item_id in range(self.n_items):
            # get average predicted reward over particles
            reward.append(self.eval(user_id, item_id))
        # choose arm according to TS or USB
        if self.policy == 'TS':
            # select arm
            self.recommended_arm = argmax(reward)
            return argmax(reward)
        elif self.policy == 'UCB':
            # compute variance
            nu = (1/self.B) * (self.sigma_sq[item_id]**2).sum()
            # select arm
            gamma = 1
            return argmax(reward + gamma*sqrt(nu))
        else:
            raise Exception("Please enter a valid policy: {'TS', 'UCB'}")

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

        # update statistics for each particle
        for particle in self.particles:
            topic = argmax(multinomial(1, pvals=particle.p[user_id]))
            particle.sample_random_variables(user_id, item_id, topic)

        if n == item_id:
            # update parameter and states
            self.update(user_id, n, reward)
            # update average reward log for time bucket t
            self.rewards_log[t] = self.rewards_log[t] + (1 / self.a[t]) * (reward - self.rewards_log[t])
            # update impression count for time bucket t
            self.a[t] += 1
            # append trace
            self.trace.append(self.rewards_log[t])
            self.arm_trace.append(n)

    def init_particles(self, n_users, n_items):
        """
        Initialize particles
        """
        # generate deep copies of particle
        self.particles = [Particle(n_users, n_items, self.n_lat) for _ in range(self.n_particles)]
        # self.particles = np.array(
        #     [Particle(n_users, n_items, self.n_lat) for _ in range(self.n_particles * self.n_users)]
        # ).reshape((self.n_particles, self.n_users))

    def replay(self, ratings):
        """
        Run experiment on dataset using replayer method
        :param ratings: dataset [user_id, item_id, rating and time bucket]
        :return:
        """
        # get number of users, items
        self.n_users = ratings.user_id.nunique()  # number of users
        self.n_items = ratings.item_id.nunique()  # number of items

        # initialize particles
        print("Initializing particles . . .")
        self.init_particles(self.n_users, self.n_items)
        print("Done.")

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
        return self.trace, self.arm_trace

    if __name__ == '__main__':
        import pandas as pd
        from datetime import date

        # get ratings data
        ratings = pd.read_csv('datasets/MovieLens10M_ratings.csv')
        ratings = ratings[['user_id', 'item_id', 'rating2', 'time_bucket']]

        # get model params from data
        T = ratings.time_bucket.nunique()  # get number of unique time buckets

        # initialize model
        from ICTR import ICTR
        B = 10  # number of particles
        K = 3  # latent parameter dimension
        model = ICTR(n_lat=K, n_particles=B, time_buckets=T, policy='TS')

        # run experiment
        model.replay(ratings)

        # get results
        avg_rating, impressions = model.get_results()
        reward_trace, arm_trace = model.get_trace()

        # save result,format date
        pd.DataFrame(np.array([avg_rating, impressions]).T, columns=['rating', 'impressions']) \
            .to_csv('test_results/ictr_results_{date}.csv'.format(date=date.today()))
        # save trace
        pd.DataFrame(np.array([reward_trace, arm_trace]), columns=['trace']) \
            .to_csv('test_results/ictr_trace_{date}.csv'.format(date=date.today()))
