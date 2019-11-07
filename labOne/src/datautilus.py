import pandas as pd
import numpy as np
import PyQt5
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

def printDigit (digit):
    transformed = np.reshape (digit.values, (16, 16))
    plt.imshow (transformed)
    plt.show ()
