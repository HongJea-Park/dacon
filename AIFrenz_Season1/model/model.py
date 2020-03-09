# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:19:21 2020

@author: hongj
"""

import torch.nn as nn
import numpy as np
            

class SequentialModel(nn.Module):
    
    def __init__(self, args):

        super(SequentialModel, self).__init__()
        
        self.input_shape= args.input_shape
        
        self.n_layers= args.n_layers
        self.drop_prob= args.drop_prob
        self.n_hidden1= args.n_hidden
        self.n_hidden2= int(np.ceil(self.n_hidden1/ 2))
        
        self.lstm= nn.LSTM(
                self.input_shape, 
                self.n_hidden1, 
                self.n_layers, 
                dropout= self.drop_prob, 
                batch_first= True
                )
        
        self.linear= nn.Sequential(
                nn.Linear(self.n_hidden1, self.n_hidden2),
                nn.BatchNorm1d(self.n_hidden2),
                nn.ReLU(),
                nn.Linear(self.n_hidden2, 1)
                )


    def forward(self, input, hidden):

        output, hidden= self.lstm(input, hidden)
        output= output.contiguous().view(-1, self.n_hidden1)
        output= self.linear(output)
        
        return output, hidden


    def init_hidden(self, batch_size):
        
        weight= next(self.parameters()).data
        initial_hidden= (
                weight.new(self.n_layers, batch_size, self.n_hidden1).zero_(),
                weight.new(self.n_layers, batch_size, self.n_hidden1).zero_()
                )

        return initial_hidden


