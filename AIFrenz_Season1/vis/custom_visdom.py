# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:22:57 2020

@author: hongj
"""

import sys
import numpy as np
import time
from visdom import Visdom
from training.custom_loss import mse_AIFrenz


class Custom_Visdom:
    
    def __init__(self, model_name, transfer_learning):
        
        self.vis = Visdom()
        if transfer_learning:
            self.vis.env = f'{model_name}_tl'
        else:
            self.vis.env = model_name
        self.vis.close()
        self.mod = sys.modules[__name__]
        self.pred_error = {}
        

    def loss_plot(self, checkpoint):
        
        opts = dict(title= 'Loss curve',
                    legend= ['train loss', 'valid_loss'],
                    showlegend= True)
        
        length = 50
        
        try:             
            if len(checkpoint.epoch_list) < length:
                self._update_loss_plot(
                        checkpoint.epoch_list, 
                        checkpoint.train_loss_list_per_epoch,
                        checkpoint.valid_loss_list,
                        opts,
                        self.loss_plt)
            else:
                self._update_loss_plot(
                        checkpoint.epoch_list[-length:], 
                        checkpoint.train_loss_list_per_epoch[-length:],
                        checkpoint.valid_loss_list[-length:],
                        opts,
                        self.loss_plt)
        except:
            self.loss_plt = self.vis.line(
                    X= np.array(
                            [checkpoint.epoch_list, 
                             checkpoint.epoch_list]).T,
                    Y= np.array(
                            [checkpoint.train_loss_list_per_epoch, 
                             checkpoint.valid_loss_list]).T,
                    opts= opts)
        
        
    def predict_plot(self, y_df, target= 'pre'):

        if target not in ['pre', 'trans']:
            raise ValueError('Wrong target')
            
        opts = dict(legend= y_df.columns.tolist(), showlegend= True)
        
        for y_target in y_df.columns:
            if y_target != 'pred':
                mse = mse_AIFrenz(y_df[y_target], y_df['pred'])        
                self.pred_error[y_target] = mse
        
        if target == 'pre':
            opts['title'] = 'pretrain set prediction'
            try:
                self._update_predict_curve(y_df, opts, self.pretrain_pred)
            except:
                self.pretrain_pred = self.vis.line(
                        X= np.tile(y_df.index, (y_df.shape[1],1)).T,
                        Y= y_df.values,
                        opts= opts)

        else:
            opts['title'] = 'Y18 prediction'
            try:
                self._update_predict_curve(y_df, opts, self.trans_pred)
            except:
                self.trans_pred = self.vis.line(
                        X= np.tile(y_df.index, (y_df.shape[1],1)).T,
                        Y= y_df.values,
                        opts= opts)                
                
    def print_error(self):
        
        text = '<h4> Error for each target </h4><br>'
        
        for keys, values in self.pred_error.items():
            text += f'{keys}: {values:7.3f} <br>'
            
        try:
            self.vis.text(text, win= self.error, append= False)
        except:
            self.error = self.vis.text(text)
        
                
    def print_params(self, params):
        
        text = '<h4> Hyperparameter list </h4><br>'
        
        for keys, values in params.items():
            text += f'{keys}: {values} <br>'
        
        self.vis.text(text)
        
        
    def print_training(self, EPOCH, epoch, training_time, 
                       avg_train_loss, valid_loss, patience, counter):
        
        iter_time = time.time()-training_time
        
        text = '<h4> Training status </h4><br>'\
            f'\r Epoch: {epoch:3d}/{str(EPOCH):3s}<br>'\
            f'train time: {int(iter_time//60):2d}m {iter_time%60:5.2f}s<br>'\
            f'avg train loss: {avg_train_loss:7.3f}<br>'\
            f'valid loss: {valid_loss:7.3f}<br>'\
            f'\r EarlyStopping: {">"*counter + "-"*(patience-counter)} |<br>'\
            # f'{"-----"*17}<br>'
               
        try:
            self.vis.text(text, win= self.training, append= False)
        except:
            self.training = self.vis.text(text)
        
                
    def _update_predict_curve(self, y_df, opts, win):
        
        self.vis.line(
                X= np.tile(y_df.index, (y_df.shape[1],1)).T,
                Y= y_df.values, 
                opts= opts,
                win= win,
                update= 'replace')
        
        
    def _update_loss_plot(
            self, epoch_list, train_loss_list_per_epoch, 
            valid_loss_list, opts, win):
        
        self.vis.line(
                X= np.array([epoch_list, epoch_list]).T,
                Y= np.array([train_loss_list_per_epoch, valid_loss_list]).T,
                opts= opts,
                win= win,
                update= 'replace')
