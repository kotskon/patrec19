
from sklearn.base import BaseEstimator, ClassifierMixin
from src import datautilus
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

class EuclideanClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self):
        self.X_mean_ = None


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """
        for i in range (10):
            print (i)
            feats_val = datautilus.findDigit (X, y, i)
            mean_val = feats_val.mean (axis = 0)
            print (mean_val)
            if i == 0:
                self.X_mean = mean_val
            else:
                self.X_mean = np.vstack((self.X_mean, mean_val))
        return self


    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        dists = euclidean_distances (self.X_mean, X)
        preds = np.array (dists).argmin (axis = 0)
        return preds


    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        preds = self.predict (X)
        truthVector = (preds.T.astype (float) == y).astype(int)
        pos = truthVector.sum ()
        score = pos / X.shape[0] * 100
        return score
