# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


class statistics_class():
    
    pass


class trn_dataset(Dataset):
    
    '''
    '''
    
    def __init__(self, chunk_size, step_size= 3, statistics= None, fine_tune= False):
        
        self.chunk_size= chunk_size
        self.step_size= step_size
        
        self.df= pd.read_csv('./data/train.csv')
        
        self.X_cols= ['X%s'%str(i).zfill(2) for i in range(40)]
        
        for zero in np.array(self.X_cols)[self.df[self.X_cols].mean(axis= 0)== 0]: self.X_cols.remove(zero)
        
        if fine_tune:
            self.Y_cols= ['Y18']
        else:
            self.Y_cols= ['Y%s'%str(i).zfill(2) for i in range(18)]
            
        self.labels= self.df[self.Y_cols].dropna()
        self.idx= self.labels.index
        self.inputs= self.df[self.X_cols].loc[self.idx]
        
        if isinstance(statistics, statistics_class):
            
            self.mean= statistics.mean
            self.std= statistics.std
            
        else:
            
            self.mean= self.df[self.X_cols].mean(axis= 0).values
            self.std= self.df[self.X_cols].std(axis= 0).values+ 1e-256
            
            self.statistics= statistics_class()
            self.statistics.mean= self.mean
            self.statistics.std= self.std
            
        self.norm_df= self.normalization(self.inputs)        
        self.labels= self.labels.values
        
        self.seq_len= (len(self.idx)- self.chunk_size)// self.step_size
        self.dataset_len= self.seq_len* self.labels.shape[1]


    def __len__(self):
        
        return self.dataset_len
 
    
    def __getitem__(self, idx):
        
        quotient= idx// self.seq_len
        remainder= idx% self.seq_len
        
        input_= torch.from_numpy(self.norm_df[remainder: remainder+ self.chunk_size]).float()
        label_= torch.from_numpy(self.labels[remainder: remainder+ self.chunk_size, quotient]).float()
        
        return input_, label_
    
    
    def normalization(self, df):
        
        return ((df- self.mean)/ self.std).values
    
    
def shuffle_dataset(dataset, batch_size, val_ratio):
    
    dataset_size= len(dataset)
    
    indices= list(range(dataset_size))
    split= int(np.floor(val_ratio* dataset_size))
    
    np.random.seed(42)
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
    
    dataset= trn_dataset(chunk_size= 30, step_size= 3)

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
