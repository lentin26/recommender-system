from numpy import array, zeros, ones, identity, argmax, copy, matmul, outer, arange, delete, exp, argwhere
from numpy.random import choice
from numpy.linalg import inv
from scipy.stats import invgamma, multivariate_normal, dirichlet, multinomial
# https://github.com/irec-org/irec/blob/master/irec/recommendation/agents/value_functions/ictr.py
# https://github.com/lda-project/lda/blob/develop/lda/_lda.pyx
# https://github.com/TaskeHAMANO/OnlineLDA_ParticleFilter/blob/efe5cc88d100fa6d5c2cfbc03593721548eced83/learn.c


class ICTR3:

    def __init__(self, n_users: int, n_items: int, n_lat: int, n_particles: int, time_buckets):
        self.n_users = n_users
        self.n_items = n_items
        self.n_lat = n_lat
        self.n_particles = n_particles

        self.eta = [copy(100 * ones((n_lat, n_items))) for _ in range(n_particles)]
        self.lambda_ = [copy(100 * ones(n_lat)) for _ in range(n_particles)]
        self.mu = [copy(zeros((n_items, n_lat))) for _ in range(n_particles)]
        self.Sigma = [[copy(identity(n_lat)) for _i in range(n_users)] for _j in range(n_particles)]
        self.Phi = [copy(dirichlet(ones(n_items)).rvs(size=n_lat)) for _ in range(n_particles)]

        self.alpha = [1.] * n_particles
        self.beta = [1.] * n_particles
        self.alpha_n = [[1.] * n_items] * n_particles
        self.beta_n = [[1.] * n_items] * n_particles
        self.sigma_2 = [[1.] * n_items] * n_particles
        self.sigma_2 = [copy(invgamma(1., 1.).rvs(size=n_items)) for _ in range(n_particles)]
        self.q = [
                    [
                        copy(multivariate_normal(self.mu[i][j], self.sigma_2[i][j] * self.Sigma[i][j]).rvs())
                        for j in range(n_items)
                    ]
                    for i in range(n_particles)
                 ]
        self.p = [copy(dirichlet(ones(n_lat)).rvs(size=n_users)) for i in range(n_particles)]

        # eligible items
        self.eligible_items = [copy(arange(n_items)) for _ in range(n_users)]

        self.rewards_log = [0] * time_buckets  # average reward per bucket
        self.a = [1] * time_buckets  # number of impression per bucket
        self.trace = []  # average reward trace
        self.arm_trace = []  # trace of arm recommendations
        self.recommended_arm = None  # use to print recommended arm

    def select_arm(self, user_id):
        # choose arm
        r = []
        for n in self.eligible_items[user_id]:
            r.append(self.eval(user_id, n))
        # choose arm
        self.recommended_arm = self.eligible_items[user_id][argmax(r)]
        return self.recommended_arm

    def eval(self, user_id, item_id):
        r = []
        for i in range(self.n_particles):
            # get user, item latent vectors
            p = self.p[i][user_id]
            q = self.q[i][item_id]
            # predict ith reward
            r.append((p * q).sum())
        # return average reward
        return sum(r) / len(r)

    def get_weights(self, user_id, n, reward):
        weights = []
        for i in range(self.n_particles):
            # expected later user vector
            p_expected = self.lambda_[i] / sum(self.lambda_[i])
            # expected latent ite vector
            phi_expected = self.eta[i][:, n] / sum(self.eta[i][:, n])
            # compute probability density of observed reward vs predicted reward
            norm_val = multivariate_normal(
                (self.p[i][user_id] * self.q[i][n]).sum(), self.sigma_2[i][n]).pdf(reward)
            weights.append(sum(norm_val * p_expected * phi_expected))
        # normalize
        return array(weights) / sum(weights)

    def resample_particles(self, weights, user_id, n):
        samples = choice(self.n_particles, size=self.n_particles, p=weights)
        for i, j in enumerate(samples):
            self.eta[i][:, n] = copy(self.eta[j][:, n])
            self.lambda_[i] = copy(self.lambda_[j])
            self.mu[i][n] = self.mu[j][n]
            self.Sigma[i][n] = self.Sigma[j][n]
            self.sigma_2[i][n] = self.sigma_2[j][n]
            self.q[i][n] = self.q[j][n]
            self.p[i][user_id] = self.p[j][user_id]
            self.Phi[i][:, n] = self.Phi[j][:, n]
            self.alpha[i] = self.alpha[j]
            self.beta[i] = self.beta[j]
            self.alpha_n[i][n] = self.alpha_n[j][n]
            self.beta_n[i][n] = self.beta_n[j][n]

    def sample_topic(self, i, user_id, n, reward):
        # draw random topic
        z = argmax(multinomial(1, [1 / self.n_lat] * self.n_lat).rvs())

        self.lambda_[i][z] += reward  # update
        self.eta[i][z, n] += reward  # update

        # get expected p_m and Phi_n
        p = self.lambda_[i] / sum(self.lambda_[i])
        phi = self.eta[i][:, n] / sum(self.eta[i][:, n])

        # draw latent topic from posterior
        z = argmax(multinomial(1, p * phi).rvs())
        return z

    def update_parameters(self, i, user_id, n, z, reward):
        # get user latent preference vector
        p = copy(self.p[i][user_id])

        # update covariance
        old_cov = copy(self.Sigma[i][n])
        new_cov = inv(inv(old_cov) + outer(p, p))  # item updated covariance

        # update mean for latent item distribution
        mu = copy(self.mu[i][n])
        new_mu = matmul(new_cov, matmul(inv(old_cov), mu) + p * reward)

        # update inverse gamma hyper-parameters
        d_old = matmul(matmul(inv(old_cov), mu), mu)
        d_new = matmul(matmul(inv(new_cov), new_mu), new_mu)

        # perform update
        self.Sigma[i][n] = new_cov
        self.mu[i][n] = new_mu
        self.alpha[i] += 0.5
        self.beta[i] += 0.5 * (d_old + reward ** 2 - d_new)
        self.alpha_n[i][n] += 0.5
        self.beta_n[i][n] += 0.5 * (d_old + reward ** 2 - d_new)

    def sample_random_variables(self, i, user_id, n, z):
        # draw variance of the noise for reward prediction
        # self.sigma_2[i][n] = invgamma(self.alpha_n[i][n], 1/self.beta_n[i][n]).rvs()
        # draw latent item vector
        self.q[i][n] = multivariate_normal(self.mu[i][n], self.sigma_2[i][n] * self.Sigma[i][n]).rvs()
        # draw latent user vector
        self.p[i][user_id] = dirichlet(self.lambda_[i]).rvs()
        # draw item mixture corresponding to preference k
        self.Phi[i][z] = dirichlet(self.eta[i][z]).rvs()

    def sample(self, user_id):
        """
        Stochastically sample to encourage replayer exploration
        :param user_id:
        :return:
        """
        for i in range(self.n_particles):
            n = choice(range(self.n_items))
            # self.sigma_2[i][n] = invgamma(self.alpha[i], 1 / self.beta[i]).rvs()
            self.q[i][n] = multivariate_normal(self.mu[i][n], self.sigma_2[i][n] * self.Sigma[i][n]).rvs()
            self.p[i][user_id] = dirichlet(self.lambda_[i]).rvs()

    def update(self, user_id, n, reward):
        # get particle {m, n(t)} weights
        weights = self.get_weights(user_id, n, reward)
        # resample particles
        self.resample_particles(weights, user_id, n)
        for i in range(self.n_particles):
            # sample z from posterior
            z = self.sample_topic(i, user_id, n, reward)
            # update statistics
            self.update_parameters(i, user_id, n, z, reward)
            # sample
            self.sample_random_variables(i, user_id, n, z)

    def evaluate_policy(self, user_id, item_id, reward, t):
        """
        Replayer method of evaluation
        :param user_id: user index
        :param item_id: item index, required for replayer method
        :param reward: observed reward at time t
        :param t: time index
        :return:
        """
        # check if there are any items left to recommend
        if self.eligible_items[user_id].size > 0:
            # if yes, select arm
            n = self.select_arm(user_id)
            if n == item_id:
                # update parameter and states
                self.update(user_id, n, reward)
                # update average reward log for time bucket t
                self.rewards_log[t] = self.rewards_log[t] + (1 / self.a[t]) * (reward - self.rewards_log[t])
                # update impression count for time bucket t
                self.a[t] += 1
                # remove item from eligibility
                n_loc = argwhere(self.eligible_items[user_id] == n)
                self.eligible_items[user_id] = delete(self.eligible_items[user_id], n_loc)
                # append trace
                self.trace.append(self.rewards_log[t])
                self.arm_trace.append(n)
            else:
                self.sample(user_id)
        else:
            self.recommended_arm = None

    def replay(self, ratings):
        """
        Run experiment on dataset using replayer method
        :param ratings: dataset [user_id, item_id, rating and time bucket]
        :return:
        """
        # run experiment
        i = 0
        for user_id, item_id, rating, t in ratings.to_numpy():
            self.evaluate_policy(user_id=int(user_id), item_id=int(item_id), reward=rating, t=int(t))
            results, impressions = self.get_results()
            print(
                "Progress", round(i / len(ratings), 3),
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
    # get number of users, items
    n_users = ratings.user_id.nunique()  # number of users
    n_items = ratings.item_id.nunique()  # number of items

    # initialize model
    B = 10  # number of particles
    K = 3  # latent parameter dimension
    model = ICTR3(n_users=n_users, n_items=n_items, n_lat=K, n_particles=B, time_buckets=T)

    # run experiment
    model.replay(ratings)

    # get results
    avg_rating, impressions = model.get_results()
    reward_trace, arm_trace = model.get_trace()

    # save result,format date
    pd.DataFrame(array([avg_rating, impressions]).T, columns=['rating', 'impressions']) \
        .to_csv('test_results/ictr32_results_{date}.csv'.format(date=date.today()))
    # save trace
    pd.DataFrame(array([reward_trace, arm_trace]), columns=['trace']) \
        .to_csv('test_results/ictr32_trace_{date}.csv'.format(date=date.today()))
