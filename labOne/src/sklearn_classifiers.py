
from sklearn.base import BaseEstimator, ClassifierMixin
from src import datautilus
from src import pytorch_stuff as pt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import torch
from torch.utils.data import DataLoader

class EuclideanClassifier(BaseEstimator, ClassifierMixin):
    """
    Classify samples based on the distance from the mean feature value
    """
    def __init__(self):
        self.X_mean_ = None


    def fit(self, X, y):
        """
        Calculates self.X_mean based on the mean
        feature values in X for each class.
        """
        self.X_mean, useless = datautilus.analyzeDigits (X, y)
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
        self.X_var = None

    def fit (self, X, y):
        """
        Calculates priors, means, covariance matrices for all
        labels.
        """
        self.aPriori = datautilus.probGen (y)
        self.X_mean, self.X_var = datautilus.analyzeDigits (X, y)
        return self

    def predict (self, X):
        """
        Performs Bayesian inference on X.
        """
        return datautilus.bayesHelp (self.X_mean, self.X_var, self.aPriori,
                          X, 0)

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
        self.X_var = None

    def fit (self, X, y):
        """
        Calculates priors, means, covariance matrices for all
        labels.
        """
        self.aPriori = datautilus.probGen (y)
        self.X_mean, self.X_var = datautilus.analyzeDigits (X, y)
        return self

    def predict (self, X):
        """
        Performs Bayesian inference on X.
        """
        return datautilus.bayesHelp (self.X_mean, self.X_var, self.aPriori,
                          X, 1e-9)

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
        self.X_var = None

    def fit (self, X, y):
        """
        Calculates priors, means, covariance matrices for all
        labels.
        """
        self.aPriori = datautilus.probGen (y)
        self.X_mean, self.X_var = datautilus.analyzeDigits (X, y)
        return self

    def predict (self, X):
        """
        Performs Bayesian inference on X.
        """
        return datautilus.bayesHelp (self.X_mean, np.ones (self.X_var.shape), self.aPriori,
                          X, 0)

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

class NeuralNet0Hidden (BaseEstimator, ClassifierMixin):
    """
    Implements a fully connected PyTorch neural net classifier with 0 hidden
    layers.
    """

    def __init__ (self, epochs, lr):
        """
        Creates the net through invoking PyTorch class.
        """
        self.net = pt.Net0Hidden ()
        self.epochs = epochs
        self.lr = lr

    def fit (self, X, y):
        """
        Trains the net on samples X, labels y.
        X is divided in training, validation data.
        """
        Xy_train = np.hstack ((y.reshape (-1, 1), X))
        train_tensor = pt.DigitsDataset (Xy_train[0:Xy_train.shape[0] - 100, :])
        valid_tensor = pt.DigitsDataset (Xy_train[Xy_train.shape[0] - 100:, :])
        train_loader = DataLoader (train_tensor, batch_size = 25,
                                   shuffle = True)
        valid_loader = DataLoader (valid_tensor, batch_size = 50,
                                   shuffle = False)
        self.net = pt.fit (self.net, self.epochs, self.lr, train_loader,
                           valid_loader)
        return self

    def predict (self, X):
        """
        Neural inference on X.
        """
        pred_tensor = torch.from_numpy (X).type ('torch.FloatTensor')
        return self.net.forward (pred_tensor).argmax (axis = 1).numpy ()

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

class NeuralNet1Hidden (BaseEstimator, ClassifierMixin):
    """
    Implements a fully connected PyTorch neural net classifier with 0 hidden
    layers.
    """

    def __init__ (self, epochs, lr, hidden_size):
        """
        Creates the net through invoking PyTorch class.
        """
        self.net = pt.Net1Hidden (hidden_size)
        self.epochs = epochs
        self.lr = lr

    def fit (self, X, y):
        """
        Trains the net on samples X, labels y.
        X is divided in training, validation data.
        """
        Xy_train = np.hstack ((y.reshape (-1, 1), X))
        train_tensor = pt.DigitsDataset (Xy_train[0:Xy_train.shape[0] - 100, :])
        valid_tensor = pt.DigitsDataset (Xy_train[Xy_train.shape[0] - 100:, :])
        train_loader = DataLoader (train_tensor, batch_size = 25,
                                   shuffle = True)
        valid_loader = DataLoader (valid_tensor, batch_size = 50,
                                   shuffle = False)
        self.net = pt.fit (self.net, self.epochs, self.lr, train_loader,
                           valid_loader)
        return self

    def predict (self, X):
        """
        Neural inference on X.
        """
        pred_tensor = torch.from_numpy (X).type ('torch.FloatTensor')
        return self.net.forward (pred_tensor).argmax (axis = 1).numpy ()

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
