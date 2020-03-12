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
    
    png_dir= mkdir('../loss_curve/AIFrenz_Season1')
    
    if checkpoint.args.fine_tune:
        png_dir= '%s/%s_fine_tune'%(png_dir, checkpoint.args.model_name)
    else:
        png_dir= '%s/%s'%(png_dir, checkpoint.args.model_name)
    
    plt.figure()
    
    if per_batch:
        plt.plot(checkpoint.batch_list, checkpoint.train_loss_list_per_batch, 'g', label= 'train loss', alpha= .5)
    
    else:        
        plt.plot(checkpoint.epoch_list, checkpoint.train_loss_list_per_epoch, 'g', label= 'train loss')
        
    plt.plot(checkpoint.epoch_list, checkpoint.valid_loss_list, 'r', label= 'valid loss')
    
    plt.title('MSE loss curve')
    plt.grid(True)
    plt.legend(loc= 'upper right')
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss')
    plt.xlim(0, checkpoint.epoch_list[-1]+ 1)
    
    while True:
        try:
            plt.savefig('%s.png'%png_dir)
            break
        except:
            print('check directory or permission'
              )
    plt.close()
