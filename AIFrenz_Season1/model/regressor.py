# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:19:21 2020

@author: hongj
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
            

class Feature_Sequential(nn.Module):
    
    
    def __init__(self):

        super(Feature_Sequential, self).__init__()
        
        self.n_layers= 2
        self.drop_prob= .25
        self.n_hidden1= 256
        
#        self.linear= nn.Sequential(
#                nn.Linear(32, self.n_hidden1),
#                nn.ReLU(),
#                nn.LayerNorm(self.n_hidden1),
#                nn.Dropout(self.drop_prob))
        
        self.lstm= nn.LSTM(
                36,
                self.n_hidden1, 
                self.n_layers, 
                dropout= self.drop_prob, 
                batch_first= True,
                bidirectional= True
                )

    def forward(self, input):
        
#        output= self.linear(input)
        output, hidden= self.lstm(input)
        output= F.relu(output)
        
        return output, hidden


class Feature_Linear(nn.Module):
    
    
    def __init__(self):
        
        super(Feature_Linear, self).__init__()
        
        self.drop_prob= .25
        self.n_hidden1= 256
        
        self.linear= nn.Sequential(
                nn.Linear(36, self.n_hidden1),
                nn.ReLU(),
                nn.LayerNorm(self.n_hidden1),
                nn.Dropout(self.drop_prob),
                nn.Linear(self.n_hidden1, self.n_hidden1),
                nn.ReLU(),
                nn.LayerNorm(self.n_hidden1),
                nn.Dropout(self.drop_prob))

    def forward(self, input):
        
        output= self.linear(input)
        
        return output
    

class Ensemble(nn.Module):
    
    
    def __init__(self):
        
        super(Ensemble, self).__init__()
        
        self.drop_prob= .25
        self.Feature_Linear= Feature_Linear()
        self.Feature_Sequential= Feature_Sequential()
        self.n_hidden1= self.Feature_Sequential.n_hidden1* 2+ self.Feature_Linear.n_hidden1
        self.n_hidden2= self.n_hidden1// 4

        self.linear= nn.Sequential(
                nn.Linear(self.n_hidden1, self.n_hidden2),
                nn.ReLU(),
                nn.LayerNorm(self.n_hidden2),
                nn.Dropout(self.drop_prob),
                nn.Linear(self.n_hidden2, 1)
                )
     
        
    def forward(self, input):
        
        linear_output= self.Feature_Linear(input)
        sequential_output, hidden= self.Feature_Sequential(input)
        output= torch.cat((linear_output, sequential_output), dim= 2)
        output= output.contiguous().view(-1, self.n_hidden1)
        output= self.linear(output)
        
        return output, hidden
    
    