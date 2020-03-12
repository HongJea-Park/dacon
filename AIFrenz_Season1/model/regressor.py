# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:19:21 2020

@author: hongj
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
            

class Feature_Sequential(nn.Module):
    
    
    def __init__(self, args):

        super(Feature_Sequential, self).__init__()
        
        self.n_layers= 3
        self.drop_prob= args.drop_prob
        self.n_hidden1= 128
        
        self.lstm= nn.LSTM(
                40, 
                self.n_hidden1, 
                self.n_layers, 
                dropout= self.drop_prob, 
                batch_first= True
                )
        

    def forward(self, input):
        
        output, hidden= self.lstm(input)
        output= F.relu(output)
        
        return output, hidden


class Feature_Linear(nn.Module):
    
    
    def __init__(self, args):
        
        super(Feature_Linear, self).__init__()
        
        self.drop_prob= args.drop_prob
        self.n_hidden1= 512
        self.n_hidden2= 128
        
        self.linear= nn.Sequential(
                nn.Dropout(self.drop_prob),
                nn.Linear(40, self.n_hidden1),
                nn.ReLU(),
                nn.LayerNorm(self.n_hidden1),
                nn.Dropout(self.drop_prob),
                nn.Linear(self.n_hidden1, self.n_hidden2),
                nn.ReLU(),
                nn.LayerNorm(self.n_hidden2))


    def forward(self, input):
        
        output= self.linear(input)
        
        return output
    

class Ensemble(nn.Module):
    
    
    def __init__(self, args):
        
        super(Ensemble, self).__init__()
        
        self.args= args
        self.drop_prob= args.drop_prob
        self.n_hidden1= 128
        self.n_hidden2= 32
        self.Feature_Linear= Feature_Linear(args)
        self.Feature_Sequential= Feature_Sequential(args)
        
        self.linear= nn.Sequential(
                nn.Dropout(self.drop_prob),
                nn.Linear(self.n_hidden1* 2, self.n_hidden2),
                nn.ReLU(),
                nn.LayerNorm(self.n_hidden2),
                nn.Linear(self.n_hidden2, 1)
                )
        
    def forward(self, input):
        
        lstm_output, hidden= self.Feature_Sequential(input)
        linear_output= self.Feature_Linear(input)
        
        output= torch.cat((lstm_output, linear_output), dim= 2)
        output= output.contiguous().view(-1, self.n_hidden1* 2)
        output= self.linear(output)
        
        return output, hidden
    
    