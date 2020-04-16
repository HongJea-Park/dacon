# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:56:30 2020

@author: hongj
"""

import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import ConcatDataset
import time
import pandas as pd

from dataset import datasets
from dataset import dataframe
from model import regressor
from training.training import train, valid
from training import custom_loss
from utils.early_stopping import Early_stopping
from utils import argtype
from utils.checkpoint import Checkpoint
from vis.custom_visdom import Custom_Visdom
import predict


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type= str, default= 'gpu', 
                        help= 'For cpu: \'cpu\', for gpu: \'gpu\'')
    parser.add_argument('--chunk_size', type= int, default= 36, 
                        help= 'chunk size(sequence length)')
    parser.add_argument('--step_size', type= int, default= 1,
                        help= 'sequence split step')
    parser.add_argument('--lr', type= float, default= 5e-4, 
                        help= 'learning rate')
    parser.add_argument('--weight_decay', type= argtype.check_float, 
                        default= '1e-2', help= 'weight_decay')
    parser.add_argument('--epoch', type= argtype.epoch, default= 'inf', 
                        help= 'the number of epoch for training')
    parser.add_argument('--batch_size', type= int, default= 256, 
                        help= 'size of batches for training')
    parser.add_argument('--val_ratio', type= float, default= .3, 
                        help= 'validation set ratio')
    parser.add_argument('--model_name', type= str, default= 'main_model', 
                        help= 'model name to save')
    parser.add_argument('--transfer', type= argtype.boolean, default= False, 
                        help= 'whether fine tuning or not')
    parser.add_argument('--oversample_times', type= int, default= 30, 
                        help= 'the times oversampling times for fine tuning')
    parser.add_argument('--patience', type= int, default= 20, 
                        help= 'patience for early stopping')
    parser.add_argument('--c_loss', type= argtype.boolean, default= True, 
                        help= 'whether using custom loss or not')
    parser.add_argument('--predict', type= argtype.boolean, default= False, 
                        help= 'predict and save csv file or not')
    parser.add_argument('--filename', type= str, default= 'submission', 
                        help= 'csv file name to save predict result')
    parser.add_argument('--Y_list', type= argtype.str_to_list, default= 'Y12,Y15', 
                        help= 'target Y for pre-training')
    parser.add_argument('--window_size', type= int, default= 1, 
                        help= 'window size for moving average')
    parser.add_argument('--attention', type= argtype.boolean, default= True, 
                        help= 'select model using attention mechanism')

    args = parser.parse_args()
    
    data_dir = './data'

    if args.device == 'gpu': 
        args.device = 'cuda'
    device = torch.device(args.device)

    chunk_size = args.chunk_size
    step_size = args.step_size
    lr = args.lr
    weight_decay = args.weight_decay
    EPOCH = args.epoch
    batch_size = args.batch_size
    val_ratio = args.val_ratio
    model_name = args.model_name
    transfer_learning = args.transfer
    times = args.oversample_times
    patience = args.patience
    c_loss = args.c_loss
    pred = args.predict
    filename= args.filename
    Y_list = args.Y_list
    window_size = args.window_size
    attention = args.attention
    
    params = {'chunk_size': chunk_size,
              'step_size': step_size,
              'learning_rate': lr,
              'weight_decay': weight_decay,
              'epoch size': EPOCH,
              'batch_size': batch_size,
              'valid_ratio': val_ratio,
              'model_name': model_name,
              'transfer_learning': transfer_learning,
              'oversample_times': times,
              'early_stopping_patience': patience,
              'c_loss': c_loss,
              'pred': pred,
              'filename': filename,
              'Y_list': Y_list,
              'window_size': window_size,
              'attention': attention}
    
    Y = ''
    for y in Y_list:
        Y += y
    
    model_name = f'{model_name}/{Y}'
    
    Dataframe = dataframe.Dataframe(data_dir= data_dir)
    input_size = len(Dataframe.feature_cols)
    
    if attention:
        model = regressor.Attention_Regressor(input_size).to(device)
    else:
        model = regressor.BiLSTM_Regressor().to(device)
        
    checkpoint = Checkpoint(
            model_name= model_name, transfer_learning= transfer_learning)
    early_stopping = Early_stopping(patience= patience)
    vis = Custom_Visdom(model_name, transfer_learning)
    vis.print_params(params)
    
    if transfer_learning:
        
        dataset_list = []
        
        if attention:
        
            pre_df = Dataframe.get_pretrain_df()\
                    .iloc[-chunk_size+1:][Dataframe.feature_cols]
            df = Dataframe.get_y18_df()
            
            df = pd.concat([pre_df, df], axis= 0)
            
        else:
            df = Dataframe.get_y18_df()
        
        train_dataset = datasets.CustomSequenceDataset(
                chunk_size= chunk_size, 
                df= df, 
                Y= 'Y18', 
                step_size= step_size, 
                noise= True, 
                times= times)
        
        dataset_list.append(train_dataset)
        
        dataset = ConcatDataset(dataset_list)
        
        train_loader, valid_loader = datasets.split_dataset(
                dataset= dataset, 
                batch_size= batch_size, 
                val_ratio= val_ratio, 
                shuffle= True)
        
        checkpoint.load_model(model)
        
    else:
        
        dataset_list = []
        
        for y in Y_list:
    
            df = Dataframe.get_pretrain_df()
            df[y] = df[y].rolling(window= window_size, min_periods= 1).mean()
        
            dataset = datasets.CustomSequenceDataset(
                    chunk_size= chunk_size,
                    df= df,
                    Y= y,
                    step_size= step_size,
                    noise= False,
                    times= 1)
            
            dataset_list.append(dataset)
                  
        dataset = ConcatDataset(dataset_list)
        
        train_loader, valid_loader = datasets.split_dataset(
                dataset= dataset, 
                batch_size= batch_size, 
                val_ratio= val_ratio, 
                shuffle= True)
 
    optimizer = Adam(model.parameters(), 
                     lr= lr, 
                     weight_decay= float(weight_decay))

    if c_loss:
        criterion = custom_loss.mse_AIFrenz_torch
    else:
        criterion = nn.MSELoss()
            
    training_time = time.time()
    epoch = 0
    y_df = Dataframe.get_pretrain_df()[Y_list]
    y18_df = Dataframe.get_y18_df()[['Y18']]
    
    while epoch < EPOCH:
        
        print(f'\r Y: {Y} \
              chunk size: {chunk_size} \
              transfer: {transfer_learning}')
        
        epoch += 1
        train_loss_per_epoch, train_loss_list_per_batch, batch_list = train(
                model= model,
                train_loader= train_loader, 
                criterion= criterion, 
                optimizer= optimizer, 
                epoch= epoch,
                transfer_learning= transfer_learning,
                attention= attention,
                freeze_name= 'transfer_layer')
        
        valid_loss = valid(
                model= model,
                valid_loader= valid_loader, 
                criterion= criterion,
                attention= attention)

        iter_time = time.time()-training_time

        print(f'\r Epoch: {epoch:3d}/{str(EPOCH):3s}\t',
              f'train time: {int(iter_time//60):2d}m {iter_time%60:5.2f}s\t'
              f'avg train loss: {train_loss_per_epoch:7.3f}\t'
              f'valid loss: {valid_loss:7.3f}')
        
        checkpoint.save_log(
                batch_list, 
                epoch, 
                train_loss_list_per_batch, 
                train_loss_per_epoch, 
                valid_loss)
        
        early_stop, is_best = early_stopping(valid_loss)
        checkpoint.save_checkpoint(model, optimizer, is_best)
        
        vis.print_training(
                EPOCH, epoch, training_time, train_loss_per_epoch, 
                valid_loss, patience, early_stopping.counter)
        vis.loss_plot(checkpoint)
                
        print('-----'*17)
            
        y_true, y_pred, y_idx = predict.trainset_predict(
                model= model, 
                data_dir= data_dir, 
                Y= Y_list[0],
                chunk_size= chunk_size, 
                attention= attention,
                window_size= window_size)
            
        y18_true, y18_pred, y18_idx = predict.trainset_predict(
                model= model, 
                data_dir= data_dir, 
                Y= 'Y18', 
                chunk_size= chunk_size, 
                attention= attention,
                window_size= window_size)

        y_df['pred'] = y_pred
        y18_df['pred'] = y18_pred
        
        vis.predict_plot(y_df, 'pre')
        vis.predict_plot(y18_df, 'trans')
        vis.print_error()
        
        if early_stop:
            
            break
                        
    if transfer_learning:
        checkpoint.load_model(model, transfer_learningd= True)
    else:
        checkpoint.load_model(model, transfer_learningd= False)
        
    y_true, y_pred, y_idx = predict.trainset_predict(
            model= model, 
            data_dir= data_dir, 
            Y= Y_list[0],
            chunk_size= chunk_size, 
            attention= attention,
            window_size= window_size)
        
    y18_true, y18_pred, y18_idx = predict.trainset_predict(
            model= model, 
            data_dir= data_dir, 
            Y= 'Y18', 
            chunk_size= chunk_size, 
            attention= attention,
            window_size= window_size)

    y_df['pred'] = y_pred
    y18_df['pred'] = y18_pred
    
    vis.predict_plot(y_df, 'pre')
    vis.predict_plot(y18_df, 'trans')
    vis.print_error()
    
    if pred:
        
        predict.test_predict(
                model= model, 
                chunk_size= chunk_size, 
                filename= filename,
                attention= attention)
        
        
if __name__== '__main__':
    
    main()