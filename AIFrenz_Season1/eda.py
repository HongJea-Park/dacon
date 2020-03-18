# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 23:04:43 2020

@author: hongj
"""


from dataset import dataframe
from model.regressor import Ensemble
import argparse
from utils.checkpoint import Checkpoint
from utils import argtype
from training import custom_loss
from visualization.predict_curve import Predict_Curve
import torch
import pandas as pd


def eda(y, shift):
    
    parser= argparse.ArgumentParser()
    parser.add_argument('--drop_prob', type= float, default= .2, help= 'probability of dropout')
    parser.add_argument('--model_name', type= str, default= 'target_eda', help= 'model name to save')
    parser.add_argument('--fine_tune', type= argtype.boolean, default= False, help= 'whether fine tuning or not')
    
    args= parser.parse_args()
    Y_cols= 'Y%s'%str(y).zfill(2)
    args.model_name= '%s/%s'%(args.model_name, Y_cols)
    
    P_curve= Predict_Curve(args.model_name)
    
    checkpoint= Checkpoint(args)
    model= Ensemble().to('cpu')
    model.eval()
    checkpoint.load_model(model)
    
    ft_df= dataframe.get_fine_df()
    ft_true= ft_df['Y18']
    ft_id= ft_df.index
    
    input= torch.from_numpy(ft_df[dataframe.sort_Xcols()].values).float()[None, :, :]
    ft_pred= pd.Series(model(input)[0].detach().numpy().reshape(-1), index= ft_id)
        
    ft_shift_pred= ft_pred.copy()
    ft_shift_pred.index+= shift
    ft_shift_pred= ft_shift_pred.reindex(ft_id).dropna()
    ft_shift_id= ft_shift_pred.index
    
    loss_mean= custom_loss.mse_AIFrenz(ft_shift_pred, ft_true.reindex(ft_shift_id).dropna())
    loss_std= custom_loss.mse_AIFrenz_std(ft_shift_pred, ft_true.reindex(ft_shift_id).dropna())
    
    *args,= ('Y18', '%s_pred (%s shift)'%(Y_cols, shift), loss_mean, loss_std)
    
    P_curve.compare((ft_true, ft_id), (ft_shift_pred, ft_shift_id), False, *args)
        
        
eda(2, -15)
