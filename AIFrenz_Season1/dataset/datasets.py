# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from dataset import dataframe
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


class Train_Dataset(Dataset):
    
    '''
    '''
    
    def __init__(self, chunk_size, df, Y= None, step_size= 6):
        
        self.chunk_size= chunk_size
        self.step_size= step_size
        self.Y= [Y]
        
        self.X_cols= dataframe.sort_Xcols()
        self.df= df
        
        self.input= self.df[self.X_cols].values
        self.true= self.df[self.Y].values
        
        num_x, num_y= self.input.shape[0], self.true.shape[1]
        
        if self.step_size== 1: self.num_seq= ((num_x- self.chunk_size+ 1))
        else: self.num_seq= ((num_x- self.chunk_size+ 1)// self.step_size)+ 1
        
        self.dataset_len= self.num_seq* num_y
            
        self.input= torch.from_numpy(self.input).float()
        self.true= torch.from_numpy(self.true).float()
        

    def __len__(self):
        
        return self.dataset_len
 
    
    def __getitem__(self, idx):
        
        remainder= idx% self.num_seq
        input= self.input[remainder: remainder+ self.chunk_size]
        
        quotient= idx// self.num_seq
        
        input= self.input[remainder: remainder+ self.chunk_size]
        true= self.true[remainder: remainder+ self.chunk_size, quotient]
        
        return input, true

    
def split_dataset(dataset, batch_size, val_ratio, shuffle= True):
    
    dataset_size= len(dataset)
    
    indices= list(range(dataset_size))
    split= int(np.floor(val_ratio* dataset_size))
    
    if shuffle:
        np.random.shuffle(indices)

    train_indices, val_indices= indices[split:], indices[:split]

    train_sampler= SubsetRandomSampler(train_indices)
    valid_sampler= SubsetRandomSampler(val_indices)

    train_loader= DataLoader(dataset, batch_size= batch_size, 
                           sampler= train_sampler, pin_memory= True)
    valid_loader= DataLoader(dataset, batch_size= batch_size,
                           sampler= valid_sampler, pin_memory= True)

    return train_loader, valid_loader
    
    
if __name__== '__main__':
    
    df= dataframe.get_pretrain_df()
    
    dataset= Train_Dataset(chunk_size= 30, step_size= 3, fine_tune= True)

    batch_size= 256
    validation_split= .5
    shuffle= True
    random_seed= 42
    
    dataset_size= len(dataset)
    indices= list(range(dataset_size))
    split= int(np.floor(validation_split* dataset_size))
    
    if shuffle :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
    train_indices, val_indices= indices[split:], indices[:split]

    train_sampler= SubsetRandomSampler(train_indices)
    valid_sampler= SubsetRandomSampler(val_indices)

    trn_loader= DataLoader(dataset, batch_size= batch_size, 
                           sampler= train_sampler)
    val_loader= DataLoader(dataset, batch_size= batch_size,
                           sampler= valid_sampler)
    
    for batch_idx, (x, target) in enumerate(trn_loader):
        
        if batch_idx% 10== 0:
            print(x, target)
