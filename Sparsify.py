from scipy.sparse import csr_matrix
import sparse


class Sparsify:
    """
    Class for creating sparse user-item interaction matrix
    """

    def create_matrix(self, data, user_col, item_col, rating_col):
        """
        creates the sparse user-item interaction matrix with contextual time dimension

        Parameters
        ----------
        data : DataFrame
            implicit rating data

        user_col : str
            user column name

        item_col : str
            item column name

        rating_col : str
            implicit rating column name
        """

        # create a sparse matrix of using the (rating, (rows, cols)) format
        sparse.COO(coords, data, shape=((n,) * ndims))


        rows = data[user_col].cat.codes
        cols = data[item_col].cat.codes
        rating = data[rating_col]
        ratings = csr_matrix((rating, (rows, cols)))
        ratings.eliminate_zeros()
        return ratings, data
