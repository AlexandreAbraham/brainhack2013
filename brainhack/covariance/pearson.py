import numpy as np
from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator


class PearsonCorrelation(BaseEstimator):
    """Pearson correlation estimator
    """

    def __init__(self, assume_centered=False):
        self.assume_centered = assume_centered

    def fit(self, X, y=None, connectivity=None):
        """ Compute Pearson correlation coefficient

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
          Training data, where n_samples is the number of samples and
          n_features is the number of features.

        y : not used, present for API consistence purpose.

        Returns
        -------
        self : object
            Returns self.
        """
        if connectivity is None:
            self.covariance_ = np.corrcoef(X, rowvar=0)
        else:
            # We suppose connectivity as coo but most of this code would work
            # with dense matrix
            rows, cols = connectivity.nonzero()
            values = np.zeros(rows.shape)
            for i, (r, c) in enumerate(zip(rows, cols)):
                corr = np.corrcoef(X[r], X[c])
                if not np.isnan(corr):
                    values[i] = corr
            self.covariance_ = coo_matrix((values, (rows, cols)))
        return self
