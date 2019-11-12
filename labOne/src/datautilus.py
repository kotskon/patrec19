import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

def dataReader (train_path, test_path):
    """
    Reads data.
    """
    train_concat = np.loadtxt (train_path)
    test_concat = np.loadtxt (test_path)
    """
    For convenience, the labels are turned
    from floats to int.
    """
    y_train = train_concat[:, 0].astype (int)
    X_train = train_concat[:, 1:]
    y_test = test_concat[:, 0].astype (int)
    X_test = test_concat[:, 1:]
    return X_train, X_test, y_train, y_test

def digit2Fig(feature, ax):
    """
    Returns an axes object (for sub-plotting).
    """
    transformed = np.reshape (feature, (16, 16))
    im = ax.imshow (transformed, cmap = 'gist_yarg')
    return ax, im

def pickDigit (features, patterns, index):
    """
    Returns sample correspoinding to given index.
    """
    true_idx = index-1
    ind_feats = features[true_idx, :]
    ind_id = patterns[true_idx]
    return ind_feats, ind_id

def findDigit (features, patterns, value):
    """
    Returns all samples labelled with given value.
    """
    #indexes = np.nonzero (patterns == value)[0]
    return features[patterns == value, :]

def analyzeDigit (features, patterns, value):
    """
    Computes mean and variance values for all pixels (features)
    of the digit denoted by input variable 'value'
    """
    feats_val = findDigit (features, patterns, value)
    mean_val = feats_val.mean (axis = 0)
    var_val = feats_val.var (axis = 0)
    return mean_val, var_val

def askEuclid (knowledge, feature):
    """
    Classifies a single sample based on euclidean distance.
    """
    dists = euclidean_distances (knowledge, [feature])
    return np.array (dists).argmin ()

def printSingleDigit (feature):
    """
    Plots a single sample.
    """
    fig, axs = plt.subplots ()
    axs, im = digit2Fig (feature, axs)
    fig.colorbar (im)
    axs.set_ylabel ('pixel no.')
    axs.set_xlabel ('pixel no.')
    plt.show ()

def batchEuclid (features, patterns, knowledge):
    """
    Classifies whole dataset via euclidean distance. Returns score.
    """
    dists = euclidean_distances (knowledge, features)
    preds = np.array (dists).argmin (axis = 0)
    truthVector = (preds.T.astype (float) == patterns)
    pos = truthVector.sum ()
    score = pos / features.shape[0] * 100
    return score
