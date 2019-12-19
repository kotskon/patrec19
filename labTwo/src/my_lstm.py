import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn

class FrameLevelDataset(Dataset):
    def __init__(self, feats, labels):
        """
            feats: Python list of numpy arrays that contain the sequence features.
                   Each element of this list is a numpy array of shape seq_length x feature_dimension
            labels: Python list that contains the label for each sequence (each label must be an integer)
        """
        self.lengths = np.array ([len (seq) for seq in feats])
        self.feats = self.zero_pad_and_stack(np.array (feats))
        if isinstance(labels, (list, tuple)):
            self.labels = torch.tensor (labels)

    def zero_pad_and_stack(self, x):
        """
            This function performs zero padding on a list of features and forms them into a numpy 3D array
            returns
                padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        maxLen = self.lengths.max ()
        y = np.zeros ((len (x), maxLen, x[0].shape[-1]))
        for i in range (len (x)):
            if x[i].shape[0] < maxLen:
                diff = maxLen - x[i].shape[0]
                x[i] = np.vstack ((x[i], np.zeros ((diff, x[0].shape[-1]))))
            y[i, :, :] = x[i]
        return torch.from_numpy (y).type ('torch.FloatTensor')

    def __getitem__(self, item):
        return self.feats[item], self.labels[item], self.lengths[item]

    def __len__(self):
        return len(self.feats)

class BasicLSTM(nn.Module):

    def __init__(self, input_dim, rnn_size, output_dim, num_layers,         \
                 bidirectional = False, dropout = 0):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size
        self.lstm = nn.LSTM (input_dim, rnn_size, num_layers,               \
                             batch_first = True, dropout = dropout,         \
                             bidirectional = self.bidirectional)
        self.hidden_cell = torch.zeros (num_layers, 1, self.feature_size)
        self.lin = nn.Linear (rnn_size, output_dim)

    def forward(self, x, lengths):
        """
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index

            lengths: N x 1
         """

        # --------------- Insert your code here ---------------- #

        # You must have all of the outputs of the LSTM, but you need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network

        return last_outputs
