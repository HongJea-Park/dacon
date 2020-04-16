# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:27:47 2020

@author: hongj
"""

import torch
import torch.nn as nn

from model.attention import Encoder, Attention
from model.bilstm import BiLSTM, Linear


class BiLSTM_Regressor(nn.Module):
    
    def __init__(self):
        
        super(BiLSTM_Regressor, self).__init__()
        
        self.drop_prob = .2
        self.bilstm = BiLSTM()
        self.linear = Linear()
        self.bilstm_n_hidden = self.bilstm.n_hidden1*2
        self.linear_n_hidden = self.linear.n_hidden1
        self.n_hidden1 = self.bilstm_n_hidden + self.linear_n_hidden
        self.n_hidden2 = 128
        self.n_hidden3 = 64
        self.linear_input_size = self.linear.input_size

        self.concat_layer = nn.Sequential(
                nn.Dropout(self.drop_prob),
                nn.Linear(self.n_hidden1, self.n_hidden2),
                nn.ReLU(),
                nn.LayerNorm(self.n_hidden2))        
        
        self.transfer_layer = nn.Sequential(
                nn.Linear(self.n_hidden2, self.n_hidden3),
                nn.ReLU(),
                nn.Linear(self.n_hidden3, 1))

     
    def forward(self, input):
        
        bilstm_output, hidden = self.bilstm(input)
        bilstm_output = bilstm_output.view(-1, self.bilstm_n_hidden)
        
        linear_output = self.linear(input[:, :, :self.linear_input_size])

        output = torch.cat((bilstm_output, linear_output), dim= 1)
        output = output.view(-1, self.n_hidden1)
        output = self.concat_layer(output)
        output = self.transfer_layer(output)
        
        return output, hidden
    
    

class Attention_Regressor(nn.Module):
    
    def __init__(self, input_size, encoder_hidden_size= 256, num_layers= 2, 
                 bidirectional= True, dropout= 0.2):
        
        super(Attention_Regressor, self).__init__()
        
        self.input_size = input_size
        self.encoder_hidden_size = encoder_hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        if self.bidirectional:
            self.attention_size = self.encoder_hidden_size*2
        else:
            self.attention_size = self.encoder_hidden_size

        self.encoder = Encoder(
                input_size= self.input_size, 
                encoder_hidden_size= self.encoder_hidden_size, 
                num_layers= self.num_layers, 
                bidirectional= self.bidirectional,
                dropout= self.dropout)
        self.attention = Attention(self.attention_size)
        self.transfer_layer = nn.Linear(self.attention_size, 1)
        
        # size = 0
        # for p in self.parameters():
        #     size += p.nelement()
        # print('Total param size: {}'.format(size))

    def forward(self, input):
        
        outputs, hidden = self.encoder(input)
        
        # if isinstance(hidden, tuple): # for LSTM
        #     hidden = hidden[0] # take the hidden state
        
        if self.encoder.bidirectional: # need to concat the last 2 hidden layers
            hidden = torch.cat([hidden[-1], hidden[-2]], dim= 1)
        else:
            hidden = hidden[-1]
        
        weight, attention_value = self.attention(hidden, outputs, outputs)
        
        # option1: use only attention value
        
        output = self.transfer_layer(attention_value)
        
        # option2: use attention value and hidden value
        # remind input size of transfer layer should be attention_size*2
        
        # vector = torch.cat([hidden, attention_value], dim= 1)
        # output = self.transfer_layer(vector)
        
        return output, weight
