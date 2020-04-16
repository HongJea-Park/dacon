# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 02:26:30 2020

@author: hongj
"""

import torch
from utils.torch_device import torch_device


def train(
        model, train_loader, criterion, optimizer, epoch, transfer_learning, 
        attention= False, freeze_name= None):

    train_loss = 0
    train_loss_list = []
    batch_list = []
    num_data = 0
    
    device = torch_device(model)
    model.train()
    
    if transfer_learning:
        freeze_parameters(model, freeze_name)
        
    for batch, (input, target) in enumerate(train_loader):

        batch_size = input.size(0)
        
        input, target = input.to(device), target.to(device)
        
        if attention:        
            target = target[:, -1, :]
        else:
            target = target.view(-1, 1)
            
        model_output, _ = model(input)
        
        loss = criterion(model_output, target)
        train_loss += loss.item() * batch_size
        num_data += batch_size
        
        train_loss_list.append(loss.item())
        batch_list.append(epoch- 1 + (num_data / len(train_loader.sampler)))
        
        optimizer.zero_grad()
        loss.backward()
                    
        optimizer.step()
                    
    avg_train_loss = train_loss / num_data
    
    return avg_train_loss, train_loss_list, batch_list
            
            
def valid(model, valid_loader, criterion, attention= False):
    
    valid_loss = 0
    num_data = 0
    
    device = torch_device(model)
    
    with torch.no_grad():
    
        for _, (input, target) in enumerate(valid_loader):
            
            batch_size = input.size(0)
            
            input, target = input.to(device), target.to(device)

            if attention:        
                target = target[:, -1, :]
            else:
                target = target.view(-1, 1)
                    
            model_output, _ = model(input)
            
            valid_loss += criterion(model_output, target).item() * batch_size
            num_data += batch_size
            
    avg_valid_loss = valid_loss / num_data
    
    return avg_valid_loss


def freeze_parameters(model, tuning_name):
    
    for name, p in model.named_parameters():
        
        if tuning_name not in name:
            
            p.requires_grad = False

