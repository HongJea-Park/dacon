# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 23:30:58 2020

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
import time
import numpy as np


def main():
    
    parser= argparse.ArgumentParser()
    
    parser.add_argument('--device', type= str, default= 'gpu', help= 'For cpu: \'cpu\', for gpu: \'gpu\'')
    parser.add_argument('--chunk_size', type= int, default= 144, help= 'chunk size(sequence length)')
    parser.add_argument('--step_size', type= int, default= 3, help= 'sequence split step')
    parser.add_argument('--lr', type= float, default= 1e-3, help= 'learning rate')
    parser.add_argument('--weight_decay', type= argtype.check_float, default= '1e-3', help= 'weight_decay')
    parser.add_argument('--epoch', type= int, default= 200, help= 'epoch')
    parser.add_argument('--batch_size', type= int, default= 128, help= 'size of batches for training')
    parser.add_argument('--val_ratio', type= float, default= .3, help= 'validation set ratio')
    parser.add_argument('--model_name', type= str, default= 'target_eda', help= 'model name to save')
    parser.add_argument('--fine_tune', type= argtype.boolean, default= False, help= 'whether fine tuning or not')
    parser.add_argument('--early_stop', type= argtype.boolean, default= False, help= 'whether apply early stopping or not')
    parser.add_argument('--patience', type= int, default= 10, help= 'patience for early stopping')
    parser.add_argument('--per_batch', type= argtype.boolean, default= True, help= 'whether load checkpoint or not')
    parser.add_argument('--c_loss', type= argtype.boolean, default= True, help= 'whether using custom loss or not')
    parser.add_argument('--resume', type= argtype.boolean, default= False, help= 'whether load checkpoint or not')
    parser.add_argument('--Y', type= str, default= '', help= 'input the y class type')
    parser.add_argument('--shift', type= int, default= 0, help= 'time step shift')

    args= parser.parse_args()

    chunk_size= args.chunk_size
    step_size= args.step_size
    lr= args.lr
    weight_decay= args.weight_decay
    batch_size= args.batch_size
    val_ratio= args.val_ratio
    fine_tune= args.fine_tune
    early_stop= args.early_stop
    resume= args.resume
    c_loss= args.c_loss
    Y= args.Y
    shift= args.shift
    patience= args.patience
    
    args.model_name= '%s/%s'%(args.model_name, Y)
    
    P_curve= Predict_Curve(args.model_name)
    
    if args.device== 'gpu': 
        args.device= 'cuda'
    device= torch.device(args.device)
    
    pre_df= dataframe.get_pretrain_df(shift)
    pre_dataset= Train_Dataset(chunk_size= chunk_size, df= pre_df, Y= Y, step_size= step_size)
    train_loader, valid_loader= split_dataset(pre_dataset, batch_size, val_ratio, True)

    model= Ensemble().to(device)
    checkpoint= Checkpoint(args)
        
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
                  
        print('\n Y: %s \t resume: %s'%(Y, resume))

        training_time= time.time()
        
        train_loss_per_epoch, train_loss_list_per_batch, batch_list= train(
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
                
        P_curve.eda_predict_curve(model, device, Y, 0, True)
    
    checkpoint.load_model(model)
    P_curve.eda_predict_curve(model, device, Y, 0, True)
    
        
if __name__== '__main__':
    
    main()
