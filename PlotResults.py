from ThompsonSampling import ThompsonSampling
from Random import Random
from numpy.random import randint, multivariate_normal, dirichlet
import pandas as pd
from scipy.sparse import csr_matrix
from Sparsify import Sparsify


class PlotResults(ThompsonSampling):
    """
    Plots comparing results from various recommender systems.
    """
    def __init__(self, observational_data, lam=1, n=10, T=10):
        super().__init__(observational_data, lam, n, T)

    def plot_results(self):
        # initialize TS object and train
        ts = ThompsonSampling(observational_data=self.observational_data, lam=1, n=10, T=self.T)
        # get average number of hits per user
        ts.train()
        print("TS", ts.get_cumulative_precision())
        ts.plot_cumulative_precision()

        # get results for random
        random = Random(observational_data=self.observational_data, T=self.T)
        random.train()
        random.plot_cumulative_precision()


if __name__ == '__main__':
    # get netflix prize dataset
    print('creating the dataframe from data.csv file..')
    df = pd.read_csv('data.csv', sep=',', names=['movie', 'user', 'rating', 'date'], nrows=100000)
    df.date = pd.to_datetime(df.date)
    # add date index
    df['time_index'] = pd.cut(df.date, bins=20)
    print('Done.\n')

    # arranging the rating according to time
    print('sorting the dataframe by date..')
    df.sort_values(by='date', inplace=True)
    print('sorting done.')

    print('creating sparse matrix representation..')
    df = Sparsify().create_matrix(data=df, user_col='user', item_col='movie', rating_col='rating')
    print('Done.\n')

    results = PlotResults(observational_data=df, lam=1, n=10, T=10)
    results.plot_results()


