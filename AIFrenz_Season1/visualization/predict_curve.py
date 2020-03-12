# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 23:36:49 2020

@author: hongj
"""


import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
from utils.mkdir import mkdir
from dataset import dataframe
import torch
from utils import custom_loss
import numpy as np


def trend_curve(arr_2d):
    
    png_dir= mkdir('../img/AIFrenz_Season1/trend')
        
    cm= plt.get_cmap('gist_rainbow')
    cNorm= colors.Normalize(vmin= 0, vmax= 17)
    scalarMap= mplcm.ScalarMappable(norm= cNorm, cmap= cm)
    
    fig= plt.figure()
    
    ax= fig.add_subplot(111)
    ax.set_prop_cycle(color= [scalarMap.to_rgba(i) for i in range(18)])

    for i in range(arr_2d.shape[1]):
        ax.plot(arr_2d[:, i], alpha= .5, label= 'Y%s'%str(i).zfill(2))
    
    ax.set_xlabel('id')
    ax.set_ylabel('Target')
    ax.legend()
    plt.savefig('%s.png'%png_dir)
    plt.close()


def compare(arr1, arr2, *args):
    
    name1, name2, mean, std= *args,
    png_dir= mkdir('../img/AIFrenz_Season1/%s_%s'%(name1, name2))
    
    fig= plt.figure(figsize= (9, 4))
    
    ax= fig.add_subplot(111)
    ax.plot(arr1, alpha= .5, label= name1, color= 'g')
    ax.plot(arr2, alpha= .5, label= name2, color= 'r')
    ax.legend()
    ax.set_xlabel('id')
    ax.set_ylabel('Target')
    ax.set_title('Compare %s and %s (mean: %.6f, std: %.6f)'%(
            name1, 
            name2,
            mean,
            std), fontsize= 8)

    plt.savefig('%s.png'%png_dir)
    plt.close()
        

def pretrain_predict_curve(model, device, Y_col):
    
    model.to(device).eval()
        
    pretrain_df= dataframe.get_pretrain_df()
    train_df= dataframe.get_train_df()
    
    train_input= torch.from_numpy(train_df.values).float().to(device)[None, :, :]
    y_pred, _= model(train_input)
    y_pred= y_pred.to('cpu')
    
    pre_idx= pretrain_df.index
    
    pre_pred= y_pred[pre_idx].detach().numpy()
    pre_true= pretrain_df[Y_col].values
    
    loss_mean= custom_loss.mse_AIFrenz(pre_pred, pre_true)
    loss_std= custom_loss.mse_AIFrenz_std(pre_pred, pre_true)
        
    *args1,= ('%s_pred'%Y_col[0], '%s_true'%Y_col[0], loss_mean, loss_std)
    compare(pre_pred, pre_true, *args1)
    
    y18_df= dataframe.get_fine_df()

    y18_idx= np.setdiff1d(train_df.index, pre_idx)
    
    y18_pred= y_pred[y18_idx].detach().numpy()
    y18_true= y18_df['Y18'].values
    
    loss_mean= custom_loss.mse_AIFrenz(y18_pred, y18_true)
    loss_std= custom_loss.mse_AIFrenz_std(y18_pred, y18_true)
    
    *args2,= ('%s_pred'%Y_col[0], 'Y18', loss_mean, loss_std)
    compare(y18_pred, y18_true, *args2)
        

if __name__== '__main__':
    
    pass