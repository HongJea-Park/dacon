# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:18:27 2020

@author: hongj
"""


import pandas as pd


def sort_Xcols():
    
    cols= [['X00', 'X07', 'X28', 'X31', 'X32'],
           ['X01', 'X06', 'X22', 'X27', 'X29'],
           ['X02', 'X03', 'X18', 'X24', 'X26'],
           ['X04', 'X10', 'X21', 'X36', 'X39'],
           ['X05', 'X08', 'X09', 'X23', 'X33'],
           ['X11', 'X14', 'X16', 'X19', 'X34'],
           ['X12', 'X20', 'X30', 'X37', 'X38'],
           ['X13', 'X15', 'X17', 'X25', 'X35']]
        
    X_cols= []
        
    for col in cols:
            
        X_cols.extend(col)
        
    return X_cols


def sort_idx(idx, reverse= False):
    
    if reverse:
        return idx.sort_values(ascending= False)
    else:
        return idx


def normalization(df):
        
    X_cols= sort_Xcols()
    train_df= pd.read_csv('./data/train.csv')[X_cols]
    
    mean= train_df.mean(axis= 0)
    std= train_df.std(axis= 0)+ 1e-256
    
    return ((df- mean)/ std)


def get_pretrain_df(reverse= False):
    
    df= pd.read_csv('./data/train.csv')
    
    idx= df['id'].sort_values().index
    idx= sort_idx(idx, reverse)
    
    X_cols= sort_Xcols()
    Y_cols= ['Y%s'%str(i).zfill(2) for i in range(18)]
    
    df= df[X_cols+ Y_cols].reindex(idx).dropna()
    
    df[X_cols]= normalization(df[X_cols])

    return df


def get_train_df(reverse= False):
    
    df= pd.read_csv('./data/train.csv')
    
    idx= df['id'].sort_values().index
    idx= sort_idx(idx, reverse)
    
    X_cols= sort_Xcols()
    
    df[X_cols]= normalization(df[X_cols])

    return df[X_cols].reindex(idx)


def get_fine_df(reverse= False):
    
    df= pd.read_csv('./data/train.csv')
    
    idx= df['id'].sort_values().index
    idx= sort_idx(idx, reverse)

    X_cols= sort_Xcols()
    Y_cols= ['Y18']
    
    df= df[X_cols+ Y_cols].reindex(idx).dropna()
    
    df[X_cols]= normalization(df[X_cols])

    return df


def get_test_df(reverse= False):
    
    df= pd.read_csv('./data/test.csv')
    
    idx= df['id'].sort_values().index
    idx= sort_idx(idx, reverse)
    
    X_cols= sort_Xcols()
    
    df= normalization(df)
    
    return df[X_cols].reindex(idx)


def get_Y18_df(reverse= False):
    
    X_cols= sort_Xcols()
    
    df1= pd.read_csv('./data/train.csv')[['id']+ X_cols]
    df2= pd.read_csv('./data/test.csv')[['id']+ X_cols]
    
    df= pd.concat([df1, df2], axis= 0)
    
    idx= df['id'].sort_values().index
    idx= sort_idx(idx, reverse)
    
    return df[X_cols].reindex(idx)

