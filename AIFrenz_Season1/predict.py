# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 01:33:13 2020

@author: hongj
"""

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from dataset import dataframe
from dataset.datasets import CustomSequenceDataset
from utils.torch_device import torch_device


def test_predict(
        model, chunk_size, save_option= True, data_dir= './data', 
        filename= 'submission', attention= False):
    
    device = torch_device(model)

    model.to(device)
    model.eval()
        
    DF = dataframe.Dataframe(data_dir)

    pre_df = DF.get_y18_df().iloc[-chunk_size+1:]
    test_df = DF.get_test_df()
    
    df = pd.concat([pre_df, test_df], axis= 0)
    
    dataset = CustomSequenceDataset(
            chunk_size= chunk_size, 
            df= df, 
            Y= 'Y18', 
            step_size= 1)
    
    loader = DataLoader(dataset, batch_size= 256, shuffle= False)
    
    if attention:
    
        y_pred = np.zeros((len(dataset), 1))
        idx = 0
            
        for batch, (input, _) in enumerate(loader):
            
            batch_size= input.size(0)
            input = input.to(device)
            pred, _ = model(input)
            y_pred[idx:idx+batch_size] = pred.cpu().data.detach().numpy()
            idx += batch_size
            
    else:
        
        input = dataset.X[chunk_size-1:, :].unsqueeze(0)
        input = input.to(device)
        pred, _ = model(input)
        y_pred = pred.cpu().data.detach().numpy().squeeze()
        
    y_pred = y_pred.reshape(-1, 1)
    submission= pd.DataFrame(y_pred, index= test_df.index, columns= ['Y18'])
    
    if save_option:
        submission.to_csv('%s/%s.csv'%(data_dir, filename))
        
    else:
        return submission
    
    
def trainset_predict(
        model, data_dir, Y, chunk_size, attention= False, window_size= 1):

    device = torch_device(model)

    model.to(device)
    model.eval()

    DF = dataframe.Dataframe(data_dir)
    
    if Y == 'Y18':
        
        df = DF.get_y18_df()
        
        if attention:
            pre_df = DF.get_pretrain_df().iloc[-chunk_size+1:][DF.feature_cols]            
            df = pd.concat([pre_df, df], axis= 0)
        
        y_idx = df.dropna().index
        y_true = df.dropna()[Y].values
        y_pred = np.zeros((df.shape[0]-chunk_size+1))
        idx = 0
    
    else:
        
        df = DF.get_pretrain_df()
        df[Y] = df[Y].rolling(window= window_size, min_periods= 1).mean()

        y_idx = df.index
        y_true = df[Y].values
        y_pred = np.zeros((y_true.shape[0]))
        
        idx = chunk_size-1
        y_pred[:idx] = y_true[:idx]
        
    dataset = CustomSequenceDataset(
            chunk_size= chunk_size, df= df, Y= Y, step_size= 1)
    loader = DataLoader(dataset, batch_size= chunk_size, shuffle= False)
        
    if attention:
        
        for batch, (input, target) in enumerate(loader):
            
            batch_size= input.size(0)
            input = input.to(device)
            pred, _ = model(input)
            y_pred[idx:idx+batch_size] = \
                    pred.cpu().data.detach().numpy().squeeze()
            idx += batch_size
            
        return y_true, y_pred, y_idx
    
    else:
     
        input = dataset.X.unsqueeze(0)
        input = input.to(device)
        pred, _ = model(input)
        y_pred = pred.cpu().data.detach().numpy().squeeze()
        
        return y_true, y_pred, y_idx
