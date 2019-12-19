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
            self.labels = np.array(labels).astype('int64')

    def zero_pad_and_stack(self, x):
        """
            This function performs zero padding on a list of features and forms them into a numpy 3D array
            returns
                padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        maxLen = self.lengths.max ()
        pad = np.zeros ((x.shape[-2], x.shape[-1]))
        for seq in x:
            if len (seq) < maxLen:
                diff = maxLen - seq.shape[0]
                seq = np.vstack ((seq, np.zeros ((diff,x.shape[-2],         \
                                  x.shape[-1]))))
        return padded

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

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()
