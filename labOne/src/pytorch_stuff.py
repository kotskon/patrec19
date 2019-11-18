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
        return self.data[idx, 1:], self.data[idx, 0]

class Net0Hidden (nn.Module):
    def __init__ (self):
        super (Net0Hidden, self).__init__ ()
        self.lin = nn.Linear (256, 10)

    def forward (self, batch):
        #The activation function (log softmax) is encoded in the loss function
        #used for fitting (cross_entropy)
        return self.lin (batch)

class Net1Hidden (nn.Module):
    #A single hidden layer is introduced. Its size can be set upon initializing.
    def __init__ (self, hidden_size):
        super (Net1Hidden, self).__init__ ()
        self.lin1 = nn.Linear (256, hidden_size)
        #The output is kept at the same size of 10 (so that log softmax still
        #has meaning).
        self.lin2 = nn.Linear (hidden_size, 10)

    def forward (self, batch):
        #ReLU is used as activation function for the input layer.
        mid = F.relu (self.lin1 (batch))
        return self.lin2 (mid)

def fit (net, epochs, lr, loader, v_loader):
    #Define loss function.
    loss_func = F.cross_entropy
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
            loss = loss_func (out, yb.type ('torch.LongTensor'))
            #Backpropagate the loss.
            loss.backward ()
            #Update weights, bias.
            opt.step ()
            #A new batch is ready to be loaded. Clean gradient memory!
            opt.zero_grad ()
        #At the end of each epoch, the network is put in evaluation mode.
        net.eval ()
        #No reason to keep the gradient for the validation set.
        with torch.no_grad ():
            valid_loss = 0
            for xb, yb in v_loader:
                valid_loss += loss_func (net.forward (xb), \
                                         yb.type ('torch.LongTensor'))
            #Early stopping!
            if old_valid_loss != -62 and valid_loss > old_valid_loss:
                break
            old_valid_loss = valid_loss
    print ('Training finished due to early stopping. Actual number of epochs: ',
           epoch)
    return net

def netScore (net, X):
    features = X[:][0]
    patterns = X[:][1]
    preds = net.forward (features).argmax (axis = 1).type ('torch.FloatTensor')
    patterns = patterns.reshape (1, -1).numpy ()
    preds = preds.reshape (1, -1).numpy ()
    return (patterns == preds).sum () / preds.size

def training_wrapper (net, epochs, lr, loader, v_loader, test_tensor):
    print ('Score before training: ', netScore (net, test_tensor))
    net = fit (net, epochs, lr, loader, v_loader)
    print ('Score after training: ', netScore (net, test_tensor))
    return net
