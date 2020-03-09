# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:19:21 2020

@author: hongj
"""

import torch.nn as nn
import torch.nn.functional as F
            

class LSTM_layer(nn.Module):
    
    def __init__(self, args):

        super(LSTM_layer, self).__init__()
        
        self.input_shape= args.input_shape
        
        self.n_layers= args.n_layers
        self.n_hidden= args.n_hidden
        self.drop_prob= args.drop_prob
        
        self.lstm= nn.LSTM(
                self.input_shape, 
                self.n_hidden, 
                self.n_layers, 
                dropout= self.drop_prob, 
                batch_first= True
                )
#        self.fc1= nn.Linear(self.n_hidden, 3)
#        self.fc2= nn.Linear(3, 1)


    def forward(self, input, hidden):

        output, hidden= self.lstm(input, hidden)
        output= output.contiguous().view(-1, self.n_hidden)
#        output= F.relu(self.fc1(output))
#        output= self.fc2(output)
        
        return output, hidden


    def init_hidden(self, batch_size):
        
        weight= next(self.parameters()).data
        initial_hidden= (
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_()
                )

        return initial_hidden



class fc_layer(nn.Module):
    
    def __init__(self):
        
        super(fc_layer, self).__init__()
        
        self.fc1= nn.Linear(self.n_hidden, 3)
        self.bn= nn.BatchNorm1d(3)
        self.fc2= nn.Linear(3, 1)

        
    def forward(self, input):
        
        output= F.relu(self.f1c(input))
        output= self.bn(output)
        output= self.fc2(output)
        
        return output