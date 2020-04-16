# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:50:34 2020

@author: hongj
"""

import torch


def torch_device(model):
    
    if next(model.parameters()).is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device