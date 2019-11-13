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

def analyzeDigits (features, patterns):
    """
    Computes mean and variance values for all pixels (features)
    of all digits.
    """
    for i in range (10):
        feats_val = features[patterns == i]
        mean_val = feats_val.mean (axis = 0)
        var_val = feats_val.var (axis = 0)
        if i == 0:
            mean_all = mean_val
            var_all = var_val
        else:
            mean_all = np.vstack((mean_all, mean_val))
            var_all = np.vstack((var_all, var_val))
    return mean_all, var_all

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

def probGen (patterns):
    """
    Returns a vector of probabilities, based on the frequency
    of each pattern in the data.
    """
    probz = np.zeros((10,), dtype = float)
    for i in range (10):
        probz[i] = (patterns == i).sum () / patterns.size
    return probz

def bayesHelp (means, vars, priors, features, smooth_fac):
    """
    Performs naive Gaussian Bayes inference on a set of features, given
    their respective means, vars and priors.
    """
    init_preds = np.zeros ((features.shape[0], 10))
    #Based on scikit-learn's GaussianNB implementation.
    if smooth_fac != 0:
        vars += smooth_fac * vars.max ()
    for i in range (10):
        dig_cov  = vars[i, :]
        #Keep only the non-zero variance features, to avoid
        #NaN's when calculating Gaussian pdf.
        idx = dig_cov != 0
        dig_mean = means[i, :]
        prob = - (features[:, idx] - dig_mean[idx]) ** 2 /     \
               (2 * dig_cov[idx])
        #Choose logarithm notation; consecutive multiplications
        #of small numbers harm accuracy.
        prob -= 0.5 * np.log (2 * np.pi * dig_cov[idx])
        #Add the prior!
        init_preds [:, i] = prob.sum (1) +              \
                            np.log (priors[i])
    return init_preds.argmax (axis = 1)
