import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def dataReader (train_path, test_path):
    train_concat = pd.read_csv (train_path, header = None,
                                delim_whitespace = True)
    test_concat = pd.read_csv (test_path, header = None,
                                delim_whitespace = True)
    y_train = train_concat.iloc[:, 0]
    X_train = train_concat.iloc[:, 1:]
    y_test = test_concat.iloc[:, 0]
    X_test = test_concat.iloc[:, 1:]
    return X_train, X_test, y_train, y_test

def digit2Fig(feature, ax):
    transformed = np.reshape (feature.values, (16, 16))
    im = ax.imshow (transformed, cmap = 'gist_yarg')
    return ax, im

def pickDigit (features, patterns, index):
    true_idx = index-1
    ind_feats = features.iloc[true_idx, :]
    ind_id = patterns.iloc[true_idx]
    return ind_feats, ind_id

def findDigit (features, patterns, value):
    indexes = patterns[patterns == float (value)]
    return features.iloc[indexes.index[:], :]

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
    dists = np.zeros (shape = (10, 1))
    for i in range (knowledge.shape[0]):
        dists[i] = np.linalg.norm (knowledge.iloc[i, :].values -
                   feature.values)
        #print ('Distance from ', i, ': ', dists[i])
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
    featnum = features.shape[0]
    correct = 0
    pairs = pd.DataFrame ()
    for i in range (featnum):
        pred = askEuclid (knowledge, features.iloc[i, :])
        pairs = pairs.append (pd.Series([int (patterns[i]), pred]),
                              ignore_index = True)
        if int (patterns[i]) == pred:
            correct += 1
    return correct / featnum * 100, pairs
