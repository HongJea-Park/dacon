# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 23:36:49 2020

@author: hongj
"""


import matplotlib.pyplot as plt
from utils.mkdir import mkdir
from dataset import dataframe
import torch
from training import custom_loss

class Predict_Curve():
    
    def __init__(self, name, fine_tune= False):
        
        self.root_dir= mkdir('../img/AIFrenz_Season1/%s/'%name)
        self.fine_tune= fine_tune
    
    
    def compare(self, tuple1, tuple2, save_option= False, *args):
        
        arr1, id1_= tuple1
        arr2, id2_= tuple2
        name1, name2, mean, std= *args,
        
        png_dir= '%s/%s_%s'%(self.root_dir, name1, name2)
        
        fig= plt.figure(figsize= (9, 4))
        
        ax= fig.add_subplot(111)
        ax.plot(id1_, arr1, alpha= .5, label= name1, color= 'r')
        ax.plot(id2_, arr2, alpha= .5, label= name2, color= 'g')
        ax.legend()
        ax.set_xlabel('id')
        ax.set_ylabel('Target')
        ax.set_title('Compare %s and %s (mean: %.6f, std: %.6f)'%(
                name1, 
                name2,
                mean,
                std), fontsize= 8)
    
        if save_option:
            
            if self.fine_tune:
                plt.savefig('%s_ft.png'%png_dir)
            else:
                plt.savefig('%s.png'%png_dir)
            
            plt.close()
            
    
    def eda_predict_curve(self, model, device, Y_col, shift= 0, save_option= False):
        
        if isinstance(Y_col, list): Y_col= Y_col[0]
        
        self.Y_predcit_curve(model, device, Y_col, shift, save_option)
        self.Y18_predict_curve(model, device, shift, save_option)
        
    
    def Y_predcit_curve(self, model, device, Y_col, shift= 0, save_option= False):    
    
        model.to(device).eval()
            
        X_cols= dataframe.sort_Xcols()
        pretrain_df= dataframe.get_pretrain_df(shift)
        
        pre_id_= pretrain_df.index
        pre_input= torch.from_numpy(pretrain_df[X_cols].values).float().to(device)[None, :, :]
        pre_pred, _= model(pre_input)
        pre_pred= pre_pred.to('cpu').detach().numpy().reshape(-1)
        pre_true= pretrain_df[Y_col].values
        
        loss_mean= custom_loss.mse_AIFrenz(pre_true, pre_pred)
        loss_std= custom_loss.mse_AIFrenz_std(pre_true, pre_pred)
            
        *args1,= ('%s_true'%Y_col, '%s_pred'%Y_col, loss_mean, loss_std)
        self.compare((pre_true, pre_id_), (pre_pred, pre_id_), save_option, *args1)
    
        
    def Y18_predict_curve(self, model, device, shift= 0, save_option= False):
        
        model.to(device).eval()
            
        X_cols= dataframe.sort_Xcols()
        ft_df= dataframe.get_fine_df(shift)
        
        ft_id_= ft_df.index
        ft_input= torch.from_numpy(ft_df[X_cols].values).float().to(device)[None, :, :]
        ft_pred, _= model(ft_input)
        ft_pred= ft_pred.to('cpu').detach().numpy().reshape(-1)
        ft_true= ft_df['Y18'].values
    
        loss_mean= custom_loss.mse_AIFrenz(ft_true, ft_pred)
        loss_std= custom_loss.mse_AIFrenz_std(ft_true, ft_pred)
        
        *args2,= ('Y18', 'Y18_pred', loss_mean, loss_std)
        self.compare((ft_true, ft_id_), (ft_pred, ft_id_), save_option, *args2)
    
        

if __name__== '__main__':
    
    pass