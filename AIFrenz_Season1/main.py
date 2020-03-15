# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:56:30 2020

@author: hongj
"""

from dataset.datasets import Train_Dataset, split_dataset
from dataset import dataframe
from model.regressor import Ensemble
from training.training import train, valid
from training import custom_loss
from utils.early_stopping import Early_stopping
from utils import argtype
from utils.checkpoint import Checkpoint
from visualization.predict_curve import Predict_Curve
from visualization.losscurve import MSE_loss
import argparse
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
import pandas as pd


def main():
    
    parser= argparse.ArgumentParser()
    
    parser.add_argument('--device', type= str, default= 'gpu', help= 'For cpu: \'cpu\', for gpu: \'gpu\'')
    parser.add_argument('--chunk_size', type= int, default= 144, help= 'chunk size(sequence length)')
    parser.add_argument('--step_size', type= int, default= 1, help= 'sequence split step')
    parser.add_argument('--drop_prob', type= float, default= .2, help= 'probability of dropout')
    parser.add_argument('--lr', type= float, default= 1e-3, help= 'learning rate')
    parser.add_argument('--weight_decay', type= argtype.check_float, default= '1e-2', help= 'weight_decay')
    parser.add_argument('--epoch', type= int, default= 200, help= 'epoch')
    parser.add_argument('--batch_size', type= int, default= 256, help= 'size of batches for training')
    parser.add_argument('--val_ratio', type= float, default= .5, help= 'validation set ratio')
    parser.add_argument('--model_name', type= str, default= 'main_model', help= 'model name to save')
    parser.add_argument('--fine_tune', type= argtype.boolean, default= False, help= 'whether fine tuning or not')
    parser.add_argument('--oversample_times', type= int, default= 200, help= 'the times oversampling times for fine tuning')
    parser.add_argument('--early_stop', type= argtype.boolean, default= False, help= 'whether apply early stopping or not')
    parser.add_argument('--patience', type= int, default= 10, help= 'patience for early stopping')
    parser.add_argument('--per_batch', type= argtype.boolean, default= True, help= 'whether load checkpoint or not')
    parser.add_argument('--c_loss', type= argtype.boolean, default= True, help= 'whether using custom loss or not')
    parser.add_argument('--resume', type= argtype.boolean, default= False, help= 'whether load checkpoint or not')
#    parser.add_argument('--Y_cols', type= argtype.str_to_list, default= '', help= 'input the y class type(by spliting ",")')

    args= parser.parse_args()

    chunk_size= args.chunk_size
    step_size= args.step_size
    lr= args.lr
    weight_decay= args.weight_decay
    batch_size= args.batch_size
    val_ratio= args.val_ratio
    fine_tune= args.fine_tune
    times= args.oversample_times
    early_stop= args.early_stop
    resume= args.resume
    c_loss= args.c_loss
#    Y_cols= args.Y_cols
    patience= args.patience
    fine_tune= args.fine_tune

    if args.device== 'gpu': 
        args.device= 'cuda'
    device= torch.device(args.device)
    
    model= Ensemble(args).to(device)
    checkpoint= Checkpoint(args)
    
    if fine_tune:        
        
        ft_os_df= dataframe.get_fine_df_oversampling(times)
        ft_df= dataframe.get_fine_df()
        
        train_dataset= Train_Dataset(chunk_size= chunk_size, df= ft_os_df, Y_cols= ['Y18'], step_size= step_size)
        valid_dataset= Train_Dataset(chunk_size= 1, df= ft_df, Y_cols= ['Y18'], step_size= 1)
        
        train_loader= DataLoader(train_dataset, batch_size= batch_size, shuffle= True, pin_memory= True)
        valid_loader= DataLoader(valid_dataset, batch_size= batch_size, shuffle= True, pin_memory= True)
        
        checkpoint.load_model(model)
        P_curve= Predict_Curve('%s_ft'%args.model_name)
        
    else:
        
        pretrain_dataset= {'Y00': -16,
                           'Y01': -10,
                           'Y02': -10,
                           'Y03': -20,
                           'Y04': -20,
                           'Y09': -4,
                           'Y12': 4,
                           'Y13': -9,
                           'Y16': 0}
        
        X_df= pd.DataFrame()
        Y_df= pd.DataFrame()
        
        X_cols= dataframe.sort_Xcols()
        
        for Y, shift in pretrain_dataset.items():
            df= dataframe.get_pretrain_df(shift)
            X, Y= df[X_cols], df[Y]
            X_df= pd.concat([X_df, X], axis= 0).reset_index(drop= True)
            Y_df= pd.concat([Y_df, Y], axis= 0).reset_index(drop= True)
            
        Y_df.columns= ['Y']
        pre_df= pd.concat([X_df, Y_df], axis= 1)
            
        dataset= Train_Dataset(chunk_size= chunk_size, df= pre_df, Y_cols= ['Y'], step_size= step_size)
        P_curve= Predict_Curve(args.model_name)
        
        train_loader, valid_loader= split_dataset(dataset, batch_size, val_ratio, True)
 
    if resume:            
        epoch, best_valid_loss= checkpoint.load_log(return_best= True)
        optimizer= Adam(model.parameters(), lr= lr, weight_decay= float(weight_decay))
        checkpoint.load_checkpoint(model, optimizer)
    else:
        optimizer= Adam(model.parameters(), lr= lr, weight_decay= float(weight_decay))
        epoch, best_valid_loss= 1, np.inf

    if c_loss:
        criterion= custom_loss.mse_AIFrenz_torch
    else:
        criterion= nn.MSELoss()
    
    if early_stop:
        early_stopping= Early_stopping(patience= patience)
    else:
        early_stopping= Early_stopping(patience= np.inf)
        
    for e in range(epoch, args.epoch+ 1):
        
        training_time= time.time()
        
        train_loss_list_per_batch, batch_list, train_loss_per_epoch= train(
                model,
                train_loader, 
                device, 
                criterion, 
                optimizer, 
                args.epoch, 
                e,
                fine_tune= fine_tune,
                print_option= True)
        
        valid_loss= valid(
                model, 
                valid_loader, 
                device, 
                criterion)

        print('\r Epoch: %3d/%3d train time: %5.2f avg train loss: %.6f valid_loss: %.6f \n'%(
                e,
                args.epoch,
                time.time()- training_time,
                train_loss_per_epoch,
                valid_loss))
        
        checkpoint.save_log(
                batch_list, 
                e, 
                train_loss_list_per_batch, 
                train_loss_per_epoch, 
                valid_loss)
        
        best_valid_loss, check= early_stopping.check_best(train_loss_per_epoch+ valid_loss, best_valid_loss)
        checkpoint.save_checkpoint(model, optimizer, check)
        
        MSE_loss(checkpoint, args.per_batch)
        
        if early_stopping.check_stop():
            
            break
                
        P_curve.Y18_predict_curve(model, device, 0, True)
        
    checkpoint.load_model(model)
    P_curve.Y18_predict_curve(model, device, 0, True)


        
        
if __name__== '__main__':
    
    main()
