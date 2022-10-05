
from numpy import arange, argmax

class Pop:
    """
    Recommends most popular items
    """
    def __init__(self, observational_data, T):
        self.observational_data = observational_data
        self.items = arange(observational_data.shape[1])
        self.n_users = observational_data.shape[0]
        self.T = T
        self.cumulative_precision = []

    def make_recommendation(self):
        """
        Recommend to each user most popular item
        :return:
        """
        return argmax(self.observational_data.mean(axis=0))

    def train(self):
        for t in range(self.T):
            for user_idx, user in enumerate(self.observational_data):
                i = self.make_recommendation()
                true_pos = int(self.observational_data[user_idx, i] == 1)
                self.cumulative_precision.append(true_pos)
