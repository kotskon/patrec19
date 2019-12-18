import numpy as np
from pomegranate import *

def transInit (n_states):
    """
    Returns a left-to-right transition matrix.
    """
    mat = np.double (np.random.rand (n_states, n_states))       # Initialization
    for i in range (n_states):
        sum = 0                                                 # Scan each row
        for j in range (n_states):
            if j < i:
                mat[i, j] = 0                                   # Left-to-right!
            sum += mat[i, j]
            if sum >= 1:                                        # Overflow?
                sum -= mat[i, j]
                if j == n_states - 1:
                    mat[i, j] = 1.0 - sum
                else:
                    mat[i, j] = np.random.uniform (0, 1.0 - sum)
                sum += mat[i, j]
            else:
                if j == n_states - 1:                           # Underflow?
                    mat[i, j] = 1.0 - sum + mat[i, j]
    return mat

def magicMarkov (X, n_states, n_mixtures, gmm = True):
    """
    Initializes and trains a Hidden Markov Model on X data. By default,
    a mixture of Gaussians is used as the emission distribution.
    """
    for i in range (len (X)):
        for j in range (len (X[i])):
            if i == 0 and j == 0:
                X_friendly = X[i][j]
            else:
                X_friendly = np.vstack ((X_friendly, X[i][j]))
    dists = [] # list of probability distributions for the HMM states
    if n_mixtures == 1:
        gmm = False
    for i in range(n_states):
        if gmm:
            a = GeneralMixtureModel.from_samples (MultivariateGaussianDistribution, \
                                                  n_mixtures, np.float_ (X_friendly))
        else:
            a = MultivariateGaussianDistribution.from_samples (np.float_ (X_friendly))
        dists.append(a)
    trans_mat = transInit (n_states) # your transition matrix
    starts = np.zeros (n_states) # your starting probability matrix
    starts[0] = 1
    ends = np.zeros (n_states) # your starting probability matrix
    ends[n_states - 1] = 1
    data = X.tolist ()    # your data: must be a Python list that contains: 2D lists with the
                # sequences (so its dimension would be num_sequences x seq_length x
                # feature_dimension). But be careful, it is not a numpy array, it is
                # a Python list (so each sequence can have different length)
    # Define the GMM-HMM
    model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends,           \
                                          state_names=['s{}'.format(i) for i in     \
                                          range(n_states)])
    model.bake ()
    # Fit the model
    model.fit(data, max_iterations=50)
    return model

def markovsAdvice (model, sample):
    # Predict a sequence
    # a sample sequence
    logp, _ = model.viterbi(sample) # Run viterbi algorithm and return
                                    # log-probability
    return logp

def ensembleSummon (models, states, mixtures):
    """
    Returns the ensemble for a particular configuration.
    """
    start = 10 * (mixtures - 1) + 50 * (states - 2)
    end = start + 10
    return models[start:end]

def markovsCouncil (models, seq):
    """
    Returns the ensemble's prediction.
    """
    preds = np.zeros (10)
    for num in range (10):
        preds[num] = markovsAdvice (models[num], seq)
    return preds.argmax ()

def markovsScore (models, data, labels):
    """
    Returns the accuracy of an ensemble on some particular data.
    """
    all = labels.size
    corrects = 0
    for i in range (len (data)):
        if markovsCouncil (models, data[i]) == labels[i]:
            corrects += 1
    return corrects / all * 100

def markovsConfusion (models, data, labels):
    """
    Returns the confusion matrix.
    """
    conf = np.zeros ((10, 10))
    for i in range (len (data)):
        pred = markovsCouncil (models, data[i])
        conf[labels[i], pred] += 1
    return conf / conf.max ()
