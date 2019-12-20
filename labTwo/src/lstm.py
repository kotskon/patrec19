import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

sns.set ()

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
    """
    Implements LSTM net, based on the inputs given.
        rnn_size: hidden layer width
        output_dim: number of classes
        num_layers, bidirectional: self-explanatory.
        dropout: probability of dropout
    """
    def __init__(self, input_dim, rnn_size, output_dim, num_layers,         \
                 bidirectional = False, dropout = 0):
        """
        Initializes network.
        """
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        # feature_size refers to the input vectors for the linear layers.
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size
        self.lstm = nn.LSTM (input_dim, rnn_size, num_layers,               \
                             batch_first = True, dropout = dropout,         \
                             bidirectional = self.bidirectional)
        # lin implements the layer just before the output.
        self.lin = nn.Linear (self.feature_size, output_dim)
        self.layers = num_layers
        # hidden marks the number of nodes in each LSTM's hidden layer.
        self.hidden = rnn_size

    def forward(self, x, lengths):
        """
        Returns LSTM final inferences for all sequences in batch x.
        """
        # h_0 initializes the hidden state. Dimensions according to the
        # documentation. Grad is set so that training can take place.
        h_0 = torch.zeros (self.layers * (1 + int (self.bidirectional)),    \
                           len (x), self.hidden).requires_grad_ ()
        # c_0 initializes the cell state.
        c_0 = h_0
        # both hidden and cell states are detached from the grad graph, so that
        # each batch is trained separately.
        out, (h_n, c_n) = self.lstm (x, (h_0.detach_ (), c_0.detach_ ()))
        # existing functions are used to isolate the outputs at the actual
        # endpoints of each sequence (sequence lengths are various)
        last_outputs = self.lin (self.last_timestep (out, lengths,          \
                                 self.bidirectional))
        return last_outputs

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account
            the zero padding
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
        direction_size = outputs.size(-1) // 2
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()

def fit (net, epochs, lr, loader, v_loader, L2 = 0, valTime = 5,            \
         showVal = False, earlyStopping = False):
    #Define loss function.
    loss_func = F.cross_entropy
    #Use Stochastic Gradient Descent of the optim package for parameter updates.
    opt = torch.optim.SGD (net.parameters (), lr = lr, weight_decay = L2)
    #A flag  for validation loss.
    old_valid_loss = -62
    old_net = net
    train_losses = []
    val_losses = []
    for epoch in range (epochs):
        epoch_loss = 0
        #It is wise, according to the documentation, to put the network in
        #training mode before each batch is loaded.
        net.train ()
        for xb, yb, lb in loader:
            out = net.forward (xb, lb)
            loss = loss_func (out, yb.type ('torch.LongTensor'))
            epoch_loss += loss
            #Backpropagate the loss.
            loss.backward ()
            #Update weights, bias.
            opt.step ()
            #A new batch is ready to be loaded. Clean gradient memory!
            opt.zero_grad ()
        print ('Train loss at epoch', epoch, ':', float (epoch_loss))
        train_losses.append (float (loss))
        val_losses.append (0)
        #At the end of each epoch, the network is put in evaluation mode.
        net.eval ()
        # Will infer on the validation set (and maybe check for early       \
        # stopping) every earlyCheck epochs.
        if showVal and epoch % valTime == 0:
            #No reason to keep the gradient for the validation set.
            with torch.no_grad ():
                valid_loss = 0
                for xb, yb, lb in v_loader:
                    out = net.forward (xb, lb)
                    valid_loss += loss_func (out, yb.type ('torch.LongTensor'))
                print ('Validation loss at epoch', epoch, ':',              \
                       float (valid_loss))
                val_losses [-1] = (float (valid_loss))
                #Early stopping!
                if old_valid_loss != -62 and valid_loss > old_valid_loss    \
                                         and earlyStopping:
                    # if the criterion is broken, pause training and return the
                    # previous logged net
                    net = old_net
                    print ('Training finished due to early stopping. Actual \
                           number of epochs:', epoch)
                    break
                old_valid_loss = valid_loss
                # If the validation has passed, keep the net logged!
                old_net = net
    train_losses = np.array (train_losses)
    val_losses = np.array (val_losses)
    train_losses = train_losses / train_losses.max ()
    val_losses = val_losses / val_losses.max ()
    return net, train_losses, val_losses

def finalStuff (net, val_loader, test_loader):
    val_conf = np.zeros ((10, 10))
    for xb, yb, lb in val_loader:
        out = net[0].forward (xb, lb)
        preds = F.softmax (out, dim = -1).argmax (dim = -1)
        acc_val = int ((preds == yb).sum ()) / len (yb) * 100
    print ('Accuracy on validation set:', acc_val, '%')
    for i in range (len (preds)):
        val_conf[int (yb[i]), int (preds[i])] += 1
    test_conf = np.zeros ((10, 10))
    for xb, yb, lb in test_loader:
        out = net[0].forward (xb, lb)
        preds = F.softmax (out, dim = -1).argmax (dim = -1)
        acc_test = int ((preds == yb).sum ()) / len (yb) * 100
    for i in range (len (preds)):
        test_conf[int (yb[i]), int (preds[i])] += 1
    print ('Accuracy on test set:', acc_test, '%')
    fig = plt.figure (figsize = (11 ,5))
    ax1 = fig.add_subplot (1, 2, 1)
    ax1 = sns.heatmap (val_conf / val_conf.max ())
    ax1.set_title ('Validation set')
    ax2 = fig.add_subplot (1, 2, 2)
    ax2 = sns.heatmap (test_conf / test_conf.max ())
    ax2.set_title ('Test set')
    fig.tight_layout ()
    fig.suptitle ('Confusion Matrices')
    plt.show ()
    fig2 = plt.figure ()
    plt.plot (np.arange (len (net[1])), net[1], label = 'training', alpha = 0.6)
    plt.plot (np.arange (len (net[2])), net[2], label = 'validation', alpha = 0.6)
    plt.xlabel ('epoch')
    plt.ylabel ('cross entropy')
    plt.legend (loc = 'best')
