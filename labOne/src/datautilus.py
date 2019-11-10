import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

def dataReader (train_path, test_path):
    train_concat = np.loadtxt (train_path)
    test_concat = np.loadtxt (test_path)
    y_train = train_concat[:, 0]
    X_train = train_concat[:, 1:]
    y_test = test_concat[:, 0]
    X_test = test_concat[:, 1:]
    """
    train_concat = pd.read_csv (train_path, header = None,
                                delim_whitespace = True)
    test_concat = pd.read_csv (test_path, header = None,
                                delim_whitespace = True)
    y_train = train_concat.iloc[:, 0]
    X_train = train_concat.iloc[:, 1:]
    y_test = test_concat.iloc[:, 0]
    X_test = test_concat.iloc[:, 1:]
    """
    return X_train, X_test, y_train, y_test

def digit2Fig(feature, ax):
    transformed = np.reshape (feature, (16, 16))
    im = ax.imshow (transformed, cmap = 'gist_yarg')
    return ax, im

def pickDigit (features, patterns, index):
    true_idx = index-1
    ind_feats = features[true_idx, :]
    ind_id = patterns[true_idx]
    return ind_feats, ind_id

def findDigit (features, patterns, value):
    indexes = np.nonzero (patterns.astype (int) == value)[0]
    return features[indexes, :]
    """
    indexes = patterns[patterns == float (value)]
    return features.iloc[indexes.index[:], :]"""

def analyzeDigit (features, patterns, value, path):
    feats_val = findDigit (features, patterns, value)
    mean_val = feats_val.mean (axis = 0)
    var_val = feats_val.var (axis = 0)
    [steps6_8_fig, axs] = plt.subplots (1, 2, squeeze = False,
                                        figsize = (20, 20))
    [axs[0, 0], im] = digit2Fig (mean_val, axs[0, 0])
    steps6_8_fig.colorbar (im, ax = axs[0, 0], fraction = .04)
    [axs[0, 1], im] = digit2Fig (var_val, axs[0, 1])
    steps6_8_fig.colorbar (im, ax = axs[0, 1], fraction = .04)
    steps6_8_fig.savefig(path + str (value) + '_stats.svg')
    plt.close ()
    return mean_val, var_val

def askEuclid (knowledge, feature):
    dists = euclidean_distances (knowledge, [feature])
    return np.array (dists).argmin ()

def printSingleDigit (feature, path, name):
    fig, axs = plt.subplots ()
    axs, im = digit2Fig (feature, axs)
    fig.colorbar (im)
    axs.set_ylabel ('pixel no.')
    axs.set_xlabel ('pixel no.')
    fig.savefig(path + name + '.svg')
    plt.close ()

def batchEuclid (features, patterns, knowledge):
    dists = euclidean_distances (knowledge, features)
    preds = np.array (dists).argmin (axis = 0)
    truthVector = (preds.T.astype (float) == patterns).astype(int)
    pos = truthVector.sum ()
    score = pos / features.shape[0] * 100
    return score
