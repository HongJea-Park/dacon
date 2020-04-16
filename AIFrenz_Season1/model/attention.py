# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 00:52:28 2020

@author: hongj
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Encoder(nn.Module):
    
    def __init__(
            self, input_size, encoder_hidden_size, num_layers, 
            bidirectional= True, dropout= 0.2):
        
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = encoder_hidden_size
        self.num_layers= num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.gru = nn.GRU(
                input_size= self.input_size, 
                hidden_size= self.hidden_size, 
                num_layers= self.num_layers,
                bidirectional= self.bidirectional,
                batch_first= True,
                dropout= self.dropout)
        
        self.layernorm = nn.LayerNorm(self.hidden_size)

    def forward(self, input):
        
        output, hidden = self.gru(input)
        output_size = output.size()
        output = output.reshape(-1, self.hidden_size)
        output = self.layernorm(output)
        output = output.reshape(output_size)
        
        return output, hidden


class Attention(nn.Module):
    
    def __init__(self, query_dim):
        
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, keys, values):
    
        # inputs-
        #     query: [Batch*Q(attn_size)]
        #     keys: [Batch*chunk_size*K(attn_size)]
        #     values: [Batch*chunk_size*V(attn_size)]
        # outputs- 
        #     weight:[chunk_size*Batch], 
        #     attention_value:[Batch*V(attn_size)]
        
        query = query.unsqueeze(1) # [Batch*1*Q(attn_size)]
        keys = keys.permute(0, 2, 1) # [Batch*K(attn_size)*chunk_size]
        score = torch.bmm(query, keys) # [Batch*1*chunk_size]
        weight = F.softmax(score.mul_(self.scale), dim=2) # [Batch*1*Q(attn_size)]
        
        attention_value = torch.bmm(weight, values).squeeze(1) #[Batch*V(attn_size)]
        return weight, attention_value

