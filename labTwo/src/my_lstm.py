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
        self.lin = nn.Linear (rnn_size, output_dim)
        self.hidden = rnn_size
        self.layers = num_layers

    def forward(self, x, lengths):
        h_0 = torch.zeros (self.layers * (1 + int (self.bidirectional)),    \
                           len (x), self.hidden).requires_grad_ ()
        c_0 = h_0
        out, (h_n, c_n) = self.lstm (x, (h_0.detach_ (), c_0.detach_ ()))
        last_outputs = self.lin (self.last_by_index (out, lengths))
        return last_outputs

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()
