# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 01:33:13 2020

@author: hongj
"""

from dataset import datasets
from model.model import SequentialModel
from utils.custom_loss import mse_AIFrenz_torch
import argparse
import torch
import numpy as np
from utils.argtype import boolean
from glob import glob
import re
import pandas as pd


def model_selection(dataset):

    w_regex= re.compile('w+[a-z0-9]+-+[0-9]')
    l_regex= re.compile('l+[0-9]+')
    h_regex= re.compile('h+[0-9]+')
    c_regex= re.compile('_c+[a-zA-Z]+')
    n_regex= re.compile('[0-9]+')
    
    input, target= dataset[0]
    input= input[None, :, :]
    target= target.view(-1, 1)
        
    ft_model_list= glob('../checkpoint/AIFrenz_Season1/*_model_fine_tune.pth')
    
    loss_list= []
    
    for model_name in ft_model_list:
        
        args= argparse.ArgumentParser().parse_args()
        
        args.weight_decay= w_regex.findall(model_name)[0][1:]
        args.n_layers= int(n_regex.findall(l_regex.findall(model_name)[0])[0])
        args.n_hidden= int(n_regex.findall(h_regex.findall(model_name)[0])[0])
        args.c_loss= boolean(c_regex.findall(model_name)[0][2:])
        args.drop_prob= .3
        args.input_shape= input.size(2)
        
        model= SequentialModel(args)        
        model_state= torch.load(model_name)        
        model.load_state_dict(model_state)
        
        model.eval()
        
        h= model.init_hidden(input.size(0))
        model_output, init= model(input, h)
        
        loss= mse_AIFrenz_torch(model_output, target).item()
        
        print('weight decay: %s \t n_layers: %s \t n_hidden: %s \t c_loss: %s \t loss: %.6f'%(
                args.weight_decay, 
                args.n_layers, 
                args.n_hidden,
                args.c_loss,
                loss
                ))
        loss_list.append(loss)
        
        if loss== np.min(loss_list):
            best_model= model
            best_init= init

    return best_model, best_init


if __name__== '__main__':
    
    dataset= datasets.trn_dataset(chunk_size= 402, step_size= 1, fine_tune= True)
    mean, std, X_cols= dataset.mean, dataset.std, dataset.X_cols
    
    model, init= model_selection(dataset)

    test_dataset= pd.read_csv('./data/test.csv')[X_cols]
    id_= test_dataset['id']
    test_input= dataset.normalization(test_dataset)
    
    input= torch.from_numpy(test_input).float()
    input= input[None, :, :]
    
    model_output, _= model(input, init)
    model_output= model_output.detach().numpy().reshape(-1)
    
    result_df= {'id': id_,
                'Y18': model_output}
    
    result_df= pd.DataFrame(result_df, columns= ['id', 'Y18'])
    result_df.to_csv('./data/submission.csv', index= False)
