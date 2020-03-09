# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:35:58 2020

@author: hongj
"""

import numpy as np
import torch


def mse_AIFrenz(y_true, y_pred):
    
    diff = abs(y_true- y_pred)
    
    less_then_one= np.where(diff< 1, 0, diff)
    
    score= np.average(np.average(less_then_one** 2, axis= 0))
    
    return score


def mse_AIFrenz_torch(y_true, y_pred):
    
    diff= abs(y_true- y_pred)
    less_then_one= torch.where(diff< 1, torch.zeros(diff.size()), diff)
    score= (less_then_one** 2).mean()
    
    return score