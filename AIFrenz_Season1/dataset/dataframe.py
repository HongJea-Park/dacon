# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:18:27 2020

@author: hongj
"""


import pandas as pd
import numpy as np


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


def normalization(df):
        
    X_cols= sort_Xcols()
    train_df= pd.read_csv('./data/train.csv')[X_cols]
    
    mean= train_df.mean(axis= 0)
    std= train_df.std(axis= 0)+ 1e-256
    
    return ((df- mean)/ std)


def get_pretrain_df(shift= 0):
    
    df= pd.read_csv('./data/train.csv', index_col= 'id')
    
    X_cols= sort_Xcols()
    Y_cols= ['Y%s'%str(i).zfill(2) for i in range(18)]
    
    df= df[X_cols+ Y_cols].dropna()
    
    X, Y= df[X_cols], df[Y_cols]
    Y.index+= shift
    
    df= pd.concat([X, Y], axis= 1).dropna()
    
    df[X_cols]= normalization(df[X_cols])

    return df


def get_fine_df(shift= 0):
    
    df= pd.read_csv('./data/train.csv', index_col= 'id')

    X_cols= sort_Xcols()
    
    df= df[X_cols+ ['Y18']].dropna()
    
    X, Y= df[X_cols], df['Y18']
    Y.index+= shift
    
    df= pd.concat([X, Y], axis= 1).dropna()
    
    df[X_cols]= normalization(df[X_cols])

    return df


def get_fine_df_oversampling(times= 100):
    
    df= pd.read_csv('./data/train.csv', index_col= 'id')

    X_cols= sort_Xcols()
    Y_cols= ['Y18']
    
    df= df[X_cols+ Y_cols].dropna()    
    df[X_cols]= normalization(df[X_cols])
    
    new_df= pd.DataFrame()
    
    for _ in range(times):
        new_df= pd.concat([new_df, df], axis= 0).reset_index(drop= True)
    
    new_df['Y18']+= (np.random.random()- .5)* 2

    return new_df


def get_test_df():
    
    df= pd.read_csv('./data/test.csv', index_col= 'id')
    df= normalization(df)
    
    return df[sort_Xcols()]

