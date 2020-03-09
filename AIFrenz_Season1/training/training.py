# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 02:26:30 2020

@author: hongj
"""

import torch
import torch.nn as nn


def train(model, train_loader, device, criterion, optimizer, EPOCH, epoch, fine_tune):
    
    '''
    '''
    
    train_loss= 0
    train_loss_list= []
    batch_list= []
    num_data= 0
    
    model.train()
    if fine_tune:
        freeze_parameters(model)
        
    for batch, (input, target) in enumerate(train_loader):
        
        batch_size= input.size(0)
        
        input, target= input.to(device), target.to(device)
        target= target.view(-1, 1)
        h= model.init_hidden(batch_size)
        
        model_output, h= model(input, h)
        
        loss= criterion(model_output, target)
        train_loss+= loss.item()* batch_size
        num_data+= batch_size
        
        train_loss_list.append(loss)
        batch_list.append(epoch- 1+ num_data/ len(train_loader.sampler))
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        if batch% 10== 0:
                        
            print('\r Epoch: %3d/%3d [%6d/%6d (%5.2f%%)] \ttrain loss: %.6f'%(
                    epoch,
                    EPOCH,
                    num_data,
                    len(train_loader.sampler),
                    100* batch/ len(train_loader),
                    loss.item()))
            
    return train_loss_list, batch_list, train_loss/ len(train_loader.sampler)
            
            
def valid(model, valid_loader, device, criterion):
    
    '''
    '''
    
    valid_loss= 0
    model.eval()
    
    with torch.no_grad():
    
        for _, (input, target) in enumerate(valid_loader):
            
            batch_size= input.size(0)
            
            input, target= input.to(device), target.to(device)
            target= target.view(-1, 1)
            h= model.init_hidden(batch_size)
            
            model_output, h= model(input, h)
            
            valid_loss+= criterion(model_output, target).item()* batch_size
            
    return valid_loss/ len(valid_loader.sampler)


def freeze_parameters(model):
    
    for name, p in model.named_parameters():
        
        if 'lstm' in name:
            
            p.requires_grad= False


if __name__== '__main__':
    
    pass
