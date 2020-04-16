# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:01:24 2020

@author: hongj
"""

import torch
import numpy as np
from utils.mkdir import mkdir


class Checkpoint:
    
    """
    """
    
    def __init__(self, model_name, transfer_learning= False):
        
        self.model_name = model_name
        self.transfer_learning = transfer_learning
        
        self.checkpoint_dir = \
            mkdir('../checkpoint/AIFrenz_Season1/%s/'%model_name)
            
        self.log_dir = '%s/log.log'%self.checkpoint_dir
        self.state_dir = '%s/state.tar'%self.checkpoint_dir
        self.model_dir = '%s/model.pth'%self.checkpoint_dir
            
        self.tl_log_dir = '%s/tl_log.log'%self.checkpoint_dir
        self.tl_state_dir = '%s/tl_state.tar'%self.checkpoint_dir
        self.tl_model_dir = '%s/tl_learning.pth'%self.checkpoint_dir
            
        self.batch_list = []
        self.epoch_list = []
        self.train_loss_list_per_batch = []
        self.train_loss_list_per_epoch = []
        self.valid_loss_list = []
        

    def save_log(
            self, batch_list, epoch, train_loss_list_per_batch, 
            train_loss_per_epoch, valid_loss
            ):
        
        """
        """
        
        self.batch_list.extend(batch_list)
        self.epoch_list.append(epoch)
        self.train_loss_list_per_batch.extend(train_loss_list_per_batch)
        self.train_loss_list_per_epoch.append(train_loss_per_epoch)
        self.valid_loss_list.append(valid_loss)
        
        self.num_batch = len(self.batch_list) // len(self.epoch_list)
        
        log = {'batch_list': self.batch_list,
               'epoch': self.epoch_list,
               'train_loss_per_batch': self.train_loss_list_per_batch,
               'train_loss_per_epoch': self.train_loss_list_per_epoch,
              ' valid_loss': self.valid_loss_list}
        
        if self.transfer_learning:
            torch.save(log, self.tl_log_dir)
        else:
            torch.save(log, self.log_dir)
            
            
    def load_log(self, return_best= False):
        
        """
        """
        
        print(f"\n loading log {self.log_dir}'")
        
        if self.transfer_learning:
            log = torch.load(self.tl_log_dir)
        else:
            log = torch.load(self.log_dir)

        self.batch_list = log['batch_list']
        self.epoch_list = log['epoch']
        self.train_loss_list_per_batch = log['train_loss_per_batch']
        self.train_loss_list_per_epoch = log['train_loss_per_epoch']
        self.valid_loss_list = log['valid_loss']
                
        if return_best:
            best_valid_loss = np.min(self.valid_loss_list)
            return self.epoch_list[-1] + 1, best_valid_loss


    def save_checkpoint(self, model, optimizer, is_best, save_state= True):
        
        """
        """
        
        if save_state:
        
            state = {'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict()}
            
            torch.save(state, self.state_dir)

        if is_best:
            
            if self.transfer_learning:
                torch.save(model.state_dict(), self.tl_model_dir)
            else:
                torch.save(model.state_dict(), self.model_dir)
                
                
    def load_checkpoint(self, model, optimizer):
        
        """
        """

        if self.transfer_learning:
            state = torch.load(self.tl_state_dir)
        else:
            state = torch.load(self.state_dir)
        
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
    
    
    def load_model(self, model, transfer_learningd= False):
        
        """
        """
        
        if transfer_learningd:
            model_state = torch.load(self.tl_model_dir)
        else:
            model_state = torch.load(self.model_dir)        
            
        model.load_state_dict(model_state)
        
