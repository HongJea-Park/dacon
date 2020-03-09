# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:56:30 2020

@author: hongj
"""

from dataset.datasets import trn_dataset, shuffle_dataset
from model.model import LSTM_layer, FC_layer
from training.training import train, valid
from utils.Early_stopping import Early_stopping
from utils.Checkpoint import Checkpoint
from utils import argtype
from utils.custom_loss import mse_AIFrenz_torch
from visualization.losscurve import MSE_loss
import argparse
import torch
from torch.optim import Adam
import time
import numpy as np


def main():
    
    parser= argparse.ArgumentParser()
    
    parser.add_argument('--device', type= str, default= 'cpu', help= 'For cpu: \'cpu\', for gpu: \'gpu\'')
    parser.add_argument('--n_layers', type= int, default= 2, help= 'the number of layer of model')
    parser.add_argument('--n_hidden', type= int, default= 3, help= 'the number of hidden units of model')
    parser.add_argument('--drop_prob', type= float, default= .3, help= 'probability of dropout')
    parser.add_argument('--lr', type= float, default= 1e-2, help= 'learning rate')
    parser.add_argument('--weight_decay', type= argtype.check_float, default= '1e-5', help= 'weight_decay')
    parser.add_argument('--epoch', type= int, default= 100, help= 'epoch')
    parser.add_argument('--batch_size', type= int, default= 256, help= 'size of batches for training')
    parser.add_argument('--val_ratio', type= float, default= .2, help= 'validation set ratio')
    parser.add_argument('--model_name', type= str, default= 'lstm', help= 'model name to save')
    parser.add_argument('--fine_tune', type= argtype.boolean, default= False, help= 'whether fine tuning or not')
    parser.add_argument('--early_stop', type= argtype.boolean, default= True, help= 'whether apply early stopping or not')
    parser.add_argument('--per_batch', type= argtype.boolean, default= True, help= 'whether load checkpoint or not')
    parser.add_argument('--resume', type= argtype.boolean, default= False, help= 'whether load checkpoint or not')

    args= parser.parse_args()

    lr= args.lr
    weight_decay= args.weight_decay
    batch_size= args.batch_size
    val_ratio= args.val_ratio
    n_layers= args.n_layers
    n_hidden= args.n_hidden
    fine_tune= args.fine_tune
    early_stop= args.early_stop
    resume= args.resume
    
    args.model_name= '%s_w%s_l%s_h%s'%(args.model_name, weight_decay.replace('.', ''), n_layers, n_hidden)
    
    if args.device== 'gpu': 
        args.device= 'cuda'
    device= torch.device(args.device)
    
    if fine_tune:
        dataset= trn_dataset(chunk_size= 144, step_size= 3, fine_tune= fine_tune)

    else:
        dataset= trn_dataset(chunk_size= 144, step_size= 6)
    
    dataset_size= len(dataset)
    
    if dataset_size< batch_size:        
        batch_size= dataset_size
        
    train_loader, valid_loader= shuffle_dataset(dataset, batch_size, val_ratio)
    
    args.input_shape= dataset[0][0].size(1)
    
    seq_model= LSTM_layer(args).to(device)
    linear_model= FC_layer(args).to(device)
    checkpoint= Checkpoint(args)
        
    if resume:
        epoch, best_valid_loss= checkpoint.load_log(return_best= True)
        parameters= list(seq_model.parameters())+ list(linear_model.parameters())
        optimizer= Adam(parameters, lr= lr, weight_decay= float(weight_decay))
        checkpoint.load_checkpoint(seq_model, linear_model, optimizer)
        
    else:
        
        if fine_tune:
            checkpoint.load_model(seq_model, linear_model)
            parameters= list(linear_model.parameters())
            
        else:
            parameters= list(seq_model.parameters())+ list(linear_model.parameters())
        
        optimizer= Adam(parameters, lr= lr, weight_decay= float(weight_decay))
        epoch, best_valid_loss= 1, np.inf        
        
    criterion= mse_AIFrenz_torch
    
    if early_stop:
        early_stopping= Early_stopping(patience= 10)
    else:
        early_stopping= Early_stopping(patience= np.inf)    
        
    for e in range(epoch, args.epoch+ 1):
        
        training_time= time.time()
        
        train_loss_list_per_batch, batch_list, train_loss_per_epoch= train(seq_model,
                                                                           linear_model,
                                                                           train_loader, 
                                                                           device, 
                                                                           criterion, 
                                                                           optimizer, 
                                                                           args.epoch, 
                                                                           e,
                                                                           fine_tune= fine_tune)
        
        valid_loss= valid(seq_model, linear_model, valid_loader, device, criterion)
        
        print('\r\n Epoch: %3d/%3d train time: %5.2f avg train loss: %.6f valid_loss: %.6f \n'%(
                e,
                args.epoch,
                time.time()- training_time,
                train_loss_per_epoch,
                valid_loss))
        
        checkpoint.save_log(batch_list, e, train_loss_list_per_batch, train_loss_per_epoch, valid_loss)
        best_valid_loss, check= early_stopping.check_best(valid_loss, best_valid_loss)
        checkpoint.save_checkpoint(seq_model, linear_model, optimizer, check)
        
        MSE_loss(checkpoint, args.per_batch)
        
        if early_stopping.check_stop():
            
            break
        
        
if __name__== '__main__':
    
    main()
