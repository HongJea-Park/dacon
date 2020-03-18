# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 01:33:13 2020

@author: hongj
"""

from dataset import dataframe
from model.regressor import Ensemble
from utils.checkpoint import Checkpoint
from utils import argtype
import argparse
import torch
import pandas as pd


if __name__== '__main__':
    
    parser= argparse.ArgumentParser()
    parser.add_argument('--drop_prob', type= float, default= .2, help= 'probability of dropout')
    parser.add_argument('--model_name', type= str, default= 'main_model', help= 'model name to save')
    parser.add_argument('--fine_tune', type= argtype.boolean, default= True, help= 'whether fine tuning or not')
    
    args= parser.parse_args()
    
    checkpoint= Checkpoint(args)
    model= Ensemble().to('cpu')
    checkpoint.load_model(model, fine_tuned= True)
    model.eval()
    
    test_df= dataframe.get_test_df()
    test_id= test_df.index
    
    X_cols= dataframe.sort_Xcols()
    
    input= torch.from_numpy(test_df[X_cols].values).float().to('cpu')[None, :, :]
    y_pred= model(input)[0].detach().numpy().reshape(-1)
    
    submission= pd.DataFrame(y_pred, index= test_id, columns= ['Y18'])
    submission.to_csv('./data/submission.csv')
