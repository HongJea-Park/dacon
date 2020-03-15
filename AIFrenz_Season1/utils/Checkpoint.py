# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:01:24 2020

@author: hongj
"""

import torch
import numpy as np
from utils.mkdir import mkdir


class Checkpoint():
    
    '''
    '''
    
    
    def __init__(self, args, lowerbetter= True):
        
        self.args= args
        self.checkpoint_dir= mkdir('../checkpoint/AIFrenz_Season1/%s/'%args.model_name)
                                
        self.lowerbetter= lowerbetter
            
        self.log_dir= '%s/log.log'%self.checkpoint_dir
        self.state_dir= '%s/state.tar'%self.checkpoint_dir
        self.model_dir= '%s/model.pth'%self.checkpoint_dir
            
        self.ft_log_dir= '%s/fine_tune_log.log'%self.checkpoint_dir
        self.ft_state_dir= '%s/fine_tune_state.tar'%self.checkpoint_dir
        self.ft_model_dir= '%s/model_fine_tune.pth'%self.checkpoint_dir
            
        self.batch_list= []
        self.epoch_list= []
        self.train_loss_list_per_batch= []
        self.train_loss_list_per_epoch= []
        self.valid_loss_list= []
        

    def save_log(self, batch_list, epoch, train_loss_list_per_batch, train_loss_per_epoch, valid_loss):
        
        '''
        '''
        
        self.batch_list.extend(batch_list)
        self.epoch_list.append(epoch)
        self.train_loss_list_per_batch.extend(train_loss_list_per_batch)
        self.train_loss_list_per_epoch.append(train_loss_per_epoch)
        self.valid_loss_list.append(valid_loss)
        
        self.num_batch= len(self.batch_list)// len(self.epoch_list)
        
        log= {'batch_list': self.batch_list,
              'epoch': self.epoch_list,
              'train_loss_per_batch': self.train_loss_list_per_batch,
              'train_loss_per_epoch': self.train_loss_list_per_epoch,
              'valid_loss': self.valid_loss_list}
        
        if self.args.fine_tune:
            torch.save(log, self.ft_log_dir)
        else:
            torch.save(log, self.log_dir)
            
            
    def load_log(self, return_best= False):
        
        '''
        Args:
            return_log: boolean type
        '''
        
        print("\n loading log '%s'"%self.log_dir)
        
        if self.args.fine_tune:
            log= torch.load(self.ft_log_dir)
        else:
            log= torch.load(self.log_dir)

        self.batch_list= log['batch_list']
        self.epoch_list= log['epoch']
        self.train_loss_list_per_batch= log['train_loss_per_batch']
        self.train_loss_list_per_epoch= log['train_loss_per_epoch']
        self.valid_loss_list= log['valid_loss']
                
        if return_best:
            
            if self.lowerbetter:
                best_valid_loss= np.min(self.valid_loss_list)
                
            else:
                best_valid_loss= np.max(self.valid_loss_list)
            
            return self.epoch_list[-1]+ 1, best_valid_loss


    def save_checkpoint(self, model, optimizer, is_best):
        
        '''
        Args:
            is_best: boolean type, whether current training step is best or not.
        '''
        
        state= {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
        
        torch.save(state, self.state_dir)

        if is_best:
            
            if self.args.fine_tune:
                torch.save(model.state_dict(), self.ft_model_dir)
            else:
                torch.save(model.state_dict(), self.model_dir)
            
    
    def load_model(self, model):
        
        '''
        #to do write note
        '''
        
        model_state= torch.load(self.model_dir)        
        model.load_state_dict(model_state)
        
        
    def load_checkpoint(self, model, optimizer):
        
        
        if self.args.fine_tune:
            state= torch.load(self.ft_state_dir)
        else:
            state= torch.load(self.state_dir)
        
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])

                        
        
if __name__== '__main__':
    
    pass
