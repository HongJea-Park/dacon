# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import dataframe


class Statistics(object):
    
    pass


class CustomSequenceDataset(Dataset):
    
    """
    """
    
    def __init__(
            self, chunk_size, df, Y= 'Y18', 
            step_size= 2, noise= False, times= 1):
        
        self.chunk_size = chunk_size
        self.step_size = step_size
        self.noise = noise
        self.times = times
        self.y = Y
        
        self.DF = dataframe.Dataframe('')
        
        self.n_cols = self.DF.n_cols
        self.df = df
        
        self.X = self.df[self.DF.feature_cols].values
        self.Y = self.df[self.y].values
        
        self.num_x = self.X.shape[0]
        
        if self.step_size == 1: 
            self.num_seq = ((self.num_x - self.chunk_size + 1))
        else: 
            self.num_seq = ((self.num_x - self.chunk_size + 1) \
                            // self.step_size)
            
        self.X = torch.from_numpy(self.X).float()
        self.Y = torch.from_numpy(self.Y).float()
        
    def __len__(self):
        
        return ((self.X.shape[0] - self.chunk_size + 1) // self.step_size) \
                    * self.times
 
    def __getitem__(self, idx):
        
        idx //= self.times
        idx *= self.step_size
                
        X = self.X[idx : (idx + self.chunk_size)]
        Y = self.Y[idx : (idx + self.chunk_size)].view(-1, 1)
        
        if self.noise:
            noise_Y = Y + (torch.rand((self.chunk_size, 1)) - 0.5)
            return X, noise_Y
        else:
            return X, Y
        
    
def split_dataset(dataset, batch_size, val_ratio, shuffle= True):
    
    dataset_size = len(dataset)
    
    indices = list(range(dataset_size))
    split = int(np.floor(val_ratio * dataset_size))
    
    if shuffle:
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size= batch_size, 
                              sampler= train_sampler, pin_memory= True)
    valid_loader = DataLoader(dataset, batch_size= batch_size, 
                              sampler= valid_sampler, pin_memory= True)

    return train_loader, valid_loader
    
    
if __name__ == '__main__':
    
    Dataframe= dataframe.Dataframe('./data')
    df= Dataframe.get_pretrain_df()
    
    dataset= CustomSequenceDataset(
            chunk_size= 24, df= df, Y= 'Y09', 
            step_size= 5, noise= True, times= 50
            )

    batch_size = 256
    validation_split = 0.5
    shuffle = False
    random_seed = 42
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    
    if shuffle :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size= batch_size, 
                            sampler= train_sampler)
    val_loader = DataLoader(dataset, batch_size= batch_size,
                            sampler= valid_sampler)
    
    for batch_idx, (x, target) in enumerate(train_loader):
        
        if batch_idx % 10 == 0:
            print(target)
