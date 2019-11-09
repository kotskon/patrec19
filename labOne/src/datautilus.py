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
    return [X_train, X_test, y_train, y_test]

def digit2Fig(feature, ax):
    transformed = np.reshape (feature.values, (16, 16))
    im = ax.imshow (transformed, cmap = 'gist_yarg')
    return [ax, im]

def pickDigit (features, patterns, index):
    true_idx = index-1
    ind_feats = features.iloc[true_idx, :]
    ind_id = patterns.iloc[true_idx]
    return [ind_feats, ind_id]

def findDigit (features, patterns, value):
    indexes = patterns[patterns == float (value)]
    return features.iloc[indexes.index[:], :]

def pixelSummoner (digit_feats, loc):
    x = loc[0] - 1
    y = loc[1] - 1
    flat_pos = 16 * x + y
    return digit_feats.iloc[:, flat_pos]
