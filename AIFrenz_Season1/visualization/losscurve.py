# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:07:26 2020

@author: hongj
"""

import matplotlib.pyplot as plt
from utils.mkdir import mkdir


def MSE_loss(checkpoint, per_batch= True):
    
    '''
    Args:
        args: args must contain dataset, anomaly_label, noise_ratio and model_name.
    '''
    
    png_dir= mkdir('../loss_curve/AIFrenz_Season1/%s/'%checkpoint.args.model_name)

    batch_len= len(checkpoint.batch_list)// len(checkpoint.epoch_list)
    loss_len= 20
    
    plt.figure()
    
    if len(checkpoint.epoch_list)< 20:
        
        if per_batch:
            plt.plot(checkpoint.batch_list, checkpoint.train_loss_list_per_batch, 'g', label= 'train loss', alpha= .5)
        
        else:        
            plt.plot(checkpoint.epoch_list, checkpoint.train_loss_list_per_epoch, 'g', label= 'train loss')
            
        plt.plot(checkpoint.epoch_list, checkpoint.valid_loss_list, 'r', label= 'valid loss')
        plt.xlim(0, checkpoint.epoch_list[-1]+ 1)
        
        xint= range(0, checkpoint.epoch_list[-1]+ 1, 2)
            
    else:
        
        if per_batch:
            plt.plot(checkpoint.batch_list[-(loss_len- 1)* batch_len:], checkpoint.train_loss_list_per_batch[-(loss_len- 1)* batch_len:], 'g', label= 'train loss', alpha= .5)
        
        else:        
            plt.plot(checkpoint.epoch_list[-loss_len:], checkpoint.train_loss_list_per_epoch[-loss_len:], 'g', label= 'train loss')
        
        plt.plot(checkpoint.epoch_list[-loss_len:], checkpoint.valid_loss_list[-loss_len:], 'r', label= 'valid loss')
        plt.xlim(checkpoint.epoch_list[-loss_len], checkpoint.epoch_list[-1]+ 1)
        xint= range(checkpoint.epoch_list[-loss_len], checkpoint.epoch_list[-1]+ 1, 2)
    
    plt.xticks(xint)
    plt.title('MSE loss curve')
    plt.grid(True)
    plt.legend(loc= 'upper right')
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss')
#    plt.xlim(0, checkpoint.epoch_list[-1]+ 1).
    
    while True:
        
        try:
            
            if checkpoint.args.fine_tune:
                plt.savefig('%s/loss_ft.png'%png_dir)
            else:
                plt.savefig('%s/loss.png'%png_dir)
                
            break
        
        except:
            print('check directory or permission')
            
    plt.close()
