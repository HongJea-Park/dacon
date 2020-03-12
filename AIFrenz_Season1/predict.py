# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 01:33:13 2020

@author: hongj
"""

from dataset import datasets
from model.model import ensemble
from utils.custom_loss import mse_AIFrenz_torch, mse_AIFrenz_torch_std
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from utils.argtype import boolean
from glob import glob
import re
import pandas as pd


def avg_predict(model, dataset, device):
    
    chunk_size= dataset.chunk_size
    step_size= dataset.step_size
    num_y= dataset.input.size(0)
    
    pred= torch.zeros(num_y).to(device)
    count= torch.zeros(num_y).to(device)
    
    loader= DataLoader(dataset, batch_size= 1)
    
    for batch, (input, _) in enumerate(loader):
        
        input= input.to(device)
        
        output, _= model(input)
        
        indices= np.arange(batch* step_size, batch* step_size+ chunk_size)
        pred[indices]+= output.view(-1)
        count[indices]+= 1
        
    return pred/ count


def predict(model, df, chunk_size, device):
    
    input_len= len(df)
    
    pred= torch.zeros(input_len).to(device)
    
#    input= torch.from_numpy(df.values[:1]).float().to(device)
#    output, hidden= model(input[None, :, :])
#    pred[0]= output.item()
    
    for i in range(1, input_len+ 1):
        
        if i< chunk_size:
            input= torch.from_numpy(df.values[:i]).float().to(device)
        else:
            input= torch.from_numpy(df.values[(i- chunk_size): i]).float().to(device)
        
        output, _= model(input[None, :, :])
        
        pred[i- 1]= output.view(-1)[-1]
        
    return pred
        

def model_selection(dataset):

    w_regex= re.compile('w+[a-z0-9]+-+[0-9]')
    l_regex= re.compile('l+[0-9]+')
    h_regex= re.compile('h+[0-9]+')
    c_regex= re.compile('_c+[a-zA-Z]+')
    n_regex= re.compile('[0-9]+')
    
    input, target= torch.from_numpy(dataset.input).float(), torch.from_numpy(dataset.label).float()
    input= input[None, :, :]
    target= target.view(-1, 1)
        
    ft_model_list= glob('../checkpoint/AIFrenz_Season1/*_model_fine_tune.pth')
    
    loss_list= []
    model_list= []
    args_list= []
    
    for model_name in ft_model_list:
        
        args= argparse.ArgumentParser().parse_args()
        
        args.weight_decay= w_regex.findall(model_name)[0][1:]
        args.n_layers= int(n_regex.findall(l_regex.findall(model_name)[0])[0])
        args.n_hidden= int(n_regex.findall(h_regex.findall(model_name)[0])[0])
        args.c_loss= boolean(c_regex.findall(model_name)[0][2:])
        args.drop_prob= .3
        args.input_shape= input.size(2)
        
        model= ensemble(args)        
        model_state= torch.load(model_name)        
        model.load_state_dict(model_state)
        
        model.eval()
        
        h= model.init_hidden(input.size(0))
        model_output, init= model(input, h)
        
        loss= mse_AIFrenz_torch(model_output, target).item()
        loss_std= mse_AIFrenz_torch_std(model_output, target)
        
        print('weight decay: %s \t n_layers: %s \t n_hidden: %s \t c_loss: %s \t loss: %.6f \t std: %.6f'%(
                args.weight_decay, 
                args.n_layers, 
                args.n_hidden,
                args.c_loss,
                loss,
                loss_std
                ))
        
        loss_list.append(loss)
        model_list.append((model, init))
        args_list.append(args)
        
#        if loss== np.min(loss_list):
#            best_model= model
#            best_init= init

    return model_list, loss_list, args_list


if __name__== '__main__':
    
    dataset= datasets.trn_dataset(chunk_size= 432, step_size= 1, fine_tune= True)
    X_cols= dataset.X_cols
    
    model_list, loss_list, args_list= model_selection(dataset)
    idx= np.argmin(loss_list)
    args= args_list[idx]
    
    model, init= model_list[idx]

    test_dataset= pd.read_csv('./data/test.csv')
    id_= test_dataset['id']
    test_input= dataset.normalization(test_dataset[X_cols])
    
    input= torch.from_numpy(test_input).float()
    input= input[None, :, :]
    
#    init= model.init_hidden(input.size(0))
    model_output, _= model(input, init)
    model_output= model_output.detach().numpy().reshape(-1)
    
    result_df= {'id': id_,
                'Y18': model_output}
    
    result_df= pd.DataFrame(result_df, columns= ['id', 'Y18'])
    result_df.to_csv('./data/submission.csv', index= False)
