import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class DigitsDataset (Dataset):
    def __init__ (self, array):
        self.dad = array
        #FloatTensor is needed by the nn.Linear module.
        self.data = torch.from_numpy (self.dad).type ('torch.FloatTensor')

    def __len__ (self):
        return self.dad.shape[0]

    def __getitem__ (self, idx):

        return self.data.view (self.dad.shape[0], 1, -1)[idx][0][:10],        \
               self.data.view (self.dad.shape[0], 1, -1)[idx][0][10:]

class DummyRNN (torch.nn.Module):
    """
    Implements a basic RNN architecture. Can process data in batches, as well
    as individually.
    """
    def __init__ (self, input_size=1, hidden_layer_size=10, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.rnn = torch.nn.RNN (input_size, hidden_layer_size)
        self.linear = torch.nn.Linear (hidden_layer_size, output_size)
        self.hidden_cell = torch.zeros (1, 1, self.hidden_layer_size)

    def forward (self, input_seq):
        seqNum = input_seq.view (-1, 1, 10).shape[0]
        seqLength = input_seq.view (-1, 1, 10).shape[-1]
        pred = torch.zeros (seqNum, 1, seqLength)
        for s in range (seqNum):
            for i in range (seqLength):
                rnn_out, self.hidden_cell = self.rnn (input_seq.view        \
                                            (seqNum, 1, seqLength)[s][0][i] \
                                            .view (1, 1, -1), self.hidden_cell)
                pred[s][0][i] = self.linear (self.hidden_cell)
        return pred

def fit (net, epochs, lr, loader, v_loader):
    #Define loss function.
    loss_func = F.mse_loss
    #Use Stochastic Gradient Descent of the optim package for parameter updates.
    opt = optim.SGD (net.parameters (), lr = lr)
    #A flag  for validation loss.
    old_valid_loss = -62
    for epoch in range (epochs):
        #It is wise, according to the documentation, to put the network in
        #training mode before each batch is loaded.
        net.train ()
        for xb, yb in loader:
            out = net.forward (xb)
            loss = loss_func (out, yb.view (-1, 1, 10))
            #Backpropagate the loss.
            loss.backward ()
            #Update weights, bias.
            opt.step ()
            #A new batch is ready to be loaded. Clean gradient memory!
            opt.zero_grad ()
            #Reset hidden gradient between batches.
            net.hidden_cell.detach_ ()
        #At the end of each epoch, the network is put in evaluation mode.
        net.eval ()
        #No reason to keep the gradient for the validation set.
        with torch.no_grad ():
            valid_loss = 0
            for xb, yb in v_loader:
                valid_loss += loss_func (net.forward (xb), \
                                         yb.view (-1, 1, 10))
            print ('Epoch', epoch, 'finished with val loss:', valid_loss)
    print ('Training finished.')
    return net
