
from sklearn.base import BaseEstimator, ClassifierMixin
from src import datautilus
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import multivariate_normal

class EuclideanClassifier(BaseEstimator, ClassifierMixin):
    """
    Classify samples based on the distance from the mean feature value
    """
    def __init__(self):
        self.X_mean_ = None


    def fit(self, X, y):
        """
        Calculates self.X_mean_ based on the mean
        feature values in X for each class.
        """
        for i in range (10):
            feats_val = datautilus.findDigit (X, y, i)
            mean_val = feats_val.mean (axis = 0)
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
        truthVector = (preds.T.astype (float) == y)
        pos = truthVector.sum ()
        score = pos / X.shape[0]
        return score

class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifies samples based on each class' feature-conditional
    probability (Bayes Formula).
    Assumes Gaussian distribution for all features.
    """
    def __init__(self):
        self.aPriori = None
        self.X_mean_ = None
        self.X_cov = None

    def fit (self, X, y):
        """
        Calculates priors, means, covariance matrices for all
        labels.
        """
        self.aPriori = datautilus.probGen (y)
        for i in range (10):
            feats_val = datautilus.findDigit (X, y, i)
            mean_val = feats_val.mean (axis = 0)
            cov_el = X[y  == i, :].var (axis = 0)
            if i == 0:
                self.X_mean = mean_val
                self.X_cov = cov_el
            else:
                self.X_mean = np.vstack((self.X_mean, mean_val))
                self.X_cov = np.vstack((self.X_cov, cov_el))
        return self

    def predict (self, X):
        """
        Performs Bayesian inference on X.
        """
        init_preds = np.zeros ((X.shape[0], 10))
        for i in range (10):
            dig_cov  = self.X_cov[i, :]
            #Keep only the non-zero variance features, to avoid
            #NaN's when calculating Gaussian pdf.
            idx = dig_cov != 0
            dig_mean = self.X_mean[i, :]
            prob = - (X[:, idx] - dig_mean[idx]) ** 2 /     \
                   (2 * dig_cov[idx])
            #Choose logarithm notation; consecutive multiplications
            #of small numbers harm accuracy.
            prob -= 0.5 * np.log (2 * np.pi * dig_cov[idx])
            #Add the prior!
            init_preds [:, i] = prob.sum (1) +              \
                                np.log (self.aPriori [i])
        return init_preds.argmax (axis = 1)

    def score (self, X, y):
            """
            Return accuracy score on the predictions
            for X based on ground truth y
            """
            preds = self.predict (X)
            truthVector = (preds.T.astype (float) == y)
            pos = truthVector.sum ()
            score = pos / X.shape[0]
            return score

class NaiveBayesClassifierSmooth(BaseEstimator, ClassifierMixin):
    """
    Classifies samples based on each class' feature-conditional
    probability (Bayes Formula).
    Assumes Gaussian distribution for all features.
    """
    def __init__(self):
        self.aPriori = None
        self.X_mean_ = None
        self.X_cov = None

    def fit (self, X, y):
        """
        Calculates priors, means, covariance matrices for all
        labels.
        """
        self.aPriori = datautilus.probGen (y)
        for i in range (10):
            feats_val = datautilus.findDigit (X, y, i)
            mean_val = feats_val.mean (axis = 0)
            cov_el = X[y  == i, :].var (axis = 0)
            if i == 0:
                self.X_mean = mean_val
                self.X_cov = cov_el
            else:
                self.X_mean = np.vstack((self.X_mean, mean_val))
                self.X_cov = np.vstack((self.X_cov, cov_el))
        return self

    def predict (self, X):
        """
        Performs Bayesian inference on X.
        """
        init_preds = np.zeros ((X.shape[0], 10))
        smooth_fac = 1e-9 * self.X_cov.max ()
        self.X_cov += smooth_fac
        for i in range (10):
            dig_cov  = self.X_cov[i, :]
            #Keep only the non-zero variance features, to avoid
            #NaN's when calculating Gaussian pdf.
            idx = dig_cov != 0
            dig_mean = self.X_mean[i, :]
            prob = - (X[:, idx] - dig_mean[idx]) ** 2 /     \
                   (2 * dig_cov[idx])
            #Choose logarithm notation; consecutive multiplications
            #of small numbers harm accuracy.
            prob -= 0.5 * np.log (2 * np.pi * dig_cov[idx])
            #Add the prior!
            init_preds [:, i] = prob.sum (1) +              \
                                np.log (self.aPriori [i])
        return init_preds.argmax (axis = 1)

    def score (self, X, y):
            """
            Return accuracy score on the predictions
            for X based on ground truth y
            """
            preds = self.predict (X)
            truthVector = (preds.T.astype (float) == y)
            pos = truthVector.sum ()
            score = pos / X.shape[0]
            return score

class NaiveBayesClassifierOnes(BaseEstimator, ClassifierMixin):
    """
    Classifies samples based on each class' feature-conditional
    probability (Bayes Formula).
    Assumes Gaussian distribution for all features.
    """
    def __init__(self):
        self.aPriori = None
        self.X_mean_ = None
        self.X_cov = None

    def fit (self, X, y):
        """
        Calculates priors, means, covariance matrices for all
        labels.
        """
        self.aPriori = datautilus.probGen (y)
        for i in range (10):
            feats_val = datautilus.findDigit (X, y, i)
            mean_val = feats_val.mean (axis = 0)
            cov_el = X[y  == i, :].var (axis = 0)
            if i == 0:
                self.X_mean = mean_val
                self.X_cov = cov_el
            else:
                self.X_mean = np.vstack((self.X_mean, mean_val))
                self.X_cov = np.vstack((self.X_cov, cov_el))
        return self

    def predict (self, X):
        """
        Performs Bayesian inference on X.
        """
        init_preds = np.zeros ((X.shape[0], 10))
        self.X_cov = np.ones (self.X_cov.shape)
        for i in range (10):
            dig_cov  = self.X_cov[i, :]
            #Keep only the non-zero variance features, to avoid
            #NaN's when calculating Gaussian pdf.
            idx = dig_cov != 0
            dig_mean = self.X_mean[i, :]
            prob = - (X[:, idx] - dig_mean[idx]) ** 2 /     \
                   (2 * dig_cov[idx])
            #Choose logarithm notation; consecutive multiplications
            #of small numbers harm accuracy.
            prob -= 0.5 * np.log (2 * np.pi * dig_cov[idx])
            #Add the prior!
            init_preds [:, i] = prob.sum (1) +              \
                                np.log (self.aPriori [i])
        return init_preds.argmax (axis = 1)

    def score (self, X, y):
            """
            Return accuracy score on the predictions
            for X based on ground truth y
            """
            preds = self.predict (X)
            truthVector = (preds.T.astype (float) == y)
            pos = truthVector.sum ()
            score = pos / X.shape[0]
            return score
