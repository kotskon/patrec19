import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import learning_curve

def analyzeDigits (features, patterns):
    """
    Computes mean and variance values for all pixels (features)
    of all digits.
    """
    for i in range (10):
        if i == 0:
            continue
        feats_val = features[patterns == i]
        mean_val = feats_val.mean (axis = 0)
        var_val = feats_val.var (axis = 0)
        if i == 1:
            mean_all = mean_val
            var_all = var_val
        else:
            mean_all = np.vstack((mean_all, mean_val))
            var_all = np.vstack((var_all, var_val))
    return mean_all, var_all

def probGen (patterns):
    """
    Returns a vector of probabilities, based on the frequency
    of each pattern in the data.
    """
    probz = np.zeros((9,), dtype = float)
    for i in range (10):
        if i == 0:
            continue
        probz[i - 1] = (patterns == i).sum () / patterns.size
    return probz

def bayesHelp (means, vars, priors, features, smooth_fac):
    """
    Performs naive Gaussian Bayes inference on a set of features, given
    their respective means, vars and priors.
    """
    init_preds = np.zeros ((features.shape[0], 9))
    #Based on scikit-learn's GaussianNB implementation.
    if smooth_fac != 0:
        vars += smooth_fac * vars.max ()
    for i in range (10):
        if i == 0:
            continue
        dig_cov  = vars[i - 1, :]
        #Keep only the non-zero variance features, to avoid
        #NaN's when calculating Gaussian pdf.
        idx = dig_cov != 0
        dig_mean = means[i - 1, :]
        prob = - (features[:, idx] - dig_mean[idx]) ** 2 /     \
               (2 * dig_cov[idx])
        #Choose logarithm notation; consecutive multiplications
        #of small numbers harm accuracy.
        prob -= 0.5 * np.log (2 * np.pi * dig_cov[idx])
        #Add the prior!
        init_preds [:, i - 1] = prob.sum (1) +              \
                            np.log (priors[i - 1])
    return init_preds.argmax (axis = 1) + 1
