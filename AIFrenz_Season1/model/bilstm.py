# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:19:21 2020

@author: hongj
"""

import torch
import torch.nn as nn
            

class BiLSTM(nn.Module):
    
    def __init__(self):

        super(BiLSTM, self).__init__()
        
        self.n_layers = 2
        self.drop_prob = .2
        self.input_size = 30
        self.n_hidden1 = 128
                
        self.lstm = nn.LSTM(
                self.input_size,
                self.n_hidden1, 
                self.n_layers, 
                dropout= self.drop_prob, 
                batch_first= True,
                bidirectional= True
                )
        self.layernorm = nn.LayerNorm(self.n_hidden1*2)
        
    def forward(self, input):
        
        output, hidden = self.lstm(input)
        output = self.layernorm(output)
        
        return output, hidden


class Linear(nn.Module):
    
    def __init__(self):
        
        super(Linear, self).__init__()
        
        self.drop_prob = .1
        self.input_size = 13
        self.n_hidden1 = 128
        
        self.linear = nn.Sequential(
                nn.Dropout(self.drop_prob),
                nn.Linear(self.input_size, self.n_hidden1),
                nn.ReLU(),
                nn.BatchNorm1d(self.n_hidden1))

    def forward(self, input):
        
        input = input.view(-1, self.input_size)
        output = self.linear(input)
        
        return output
