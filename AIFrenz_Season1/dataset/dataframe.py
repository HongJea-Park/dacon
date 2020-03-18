# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:18:27 2020

@author: hongj
"""


import pandas as pd
import numpy as np
import copy


def sort_Xcols():
    
    cols= [['m_cos', 'm_sin', 'h_cos', 'h_sin'],
           ['X00', 'X07', 'X28', 'X31', 'X32'],
           ['X01', 'X06', 'X22', 'X27', 'X29'],
           ['X02', 'X03', 'X18', 'X24', 'X26'],
           ['X04', 'X10', 'X21', 'X36', 'X39'],
           ['X05', 'X08', 'X09', 'X23', 'X33'],
           ['X11', 'X34'],
           ['X12', 'X20', 'X30', 'X37', 'X38']]
#           ['X13', 'X15', 'X17', 'X25', 'X35']]
        
    X_cols= []
        
    for col in cols:
            
        X_cols.extend(col)
                        
    return X_cols


def normalization(df):
    
    df= copy.deepcopy(df)
            
    df[['X00', 'X07', 'X28', 'X31', 'X32']]= standard_normalization(df, ['X00', 'X07', 'X28', 'X31', 'X32'])
    df[['X01', 'X06', 'X22', 'X27', 'X29']]= standard_normalization(df, ['X01', 'X06', 'X22', 'X27', 'X29'])
    df[['X02', 'X03', 'X18', 'X24', 'X26']]= minmax_normalization(df, ['X02', 'X03', 'X18', 'X24', 'X26'])
    df[['X04', 'X10', 'X21', 'X36', 'X39']]= log_normalization(df, ['X04', 'X10', 'X21', 'X36', 'X39'])
    df[['X05', 'X08', 'X09', 'X23', 'X33']]= standard_normalization(df, ['X05', 'X08', 'X09', 'X23', 'X33'])
    df[['X11', 'X34']]= log_normalization(df, ['X11', 'X34'])
    df[['X12', 'X20', 'X30', 'X37', 'X38']]= standard_normalization(df, ['X12', 'X20', 'X30', 'X37', 'X38'])
        
    return df


def standard_normalization(df, col_list):
    
    train_df= pd.read_csv('./data/train.csv', index_col= 'id')
    
    mean= train_df[col_list].mean(axis= 0)
    std= train_df[col_list].std(axis= 0)+ 1e-256
    
    df[col_list]= (df[col_list]- mean)/ std
    
    return df[col_list]


def minmax_normalization(df, col_list):
    
    train_df= pd.read_csv('./data/train.csv', index_col= 'id')
    
    min_= train_df[col_list].min(axis= 0)
    max_= train_df[col_list].max(axis= 0)
    
    df[col_list]= (df[col_list]- min_)/ (max_- min_)
    
    return df[col_list]


def log_normalization(df, col_list):
    
    return np.log10(df[col_list]+ 1)


def get_pretrain_df(shift= 0):
    
    df= pd.read_csv('./data/train.csv', index_col= 'id')
    df= get_time(df)
    
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
    df= get_time(df)

    X_cols= sort_Xcols()
    
    df= df[X_cols+ ['Y18']].dropna()
    
    X, Y= df[X_cols], df['Y18']
    Y.index+= shift
    
    df= pd.concat([X, Y], axis= 1).dropna()
    
    df[X_cols]= normalization(df[X_cols])

    return df


def get_fine_df_oversampling(times= 100):
    
    df= pd.read_csv('./data/train.csv', index_col= 'id')
    df= get_time(df)

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
    df= get_time(df)
    df= normalization(df)
    
    return df[sort_Xcols()]


def get_time(df):
    
    minute= (df.index.values% 144).astype(int)
    hour= (df.index.values% 144/ 6).astype(int)
    
    df['m_cos']= np.cos(np.pi* 2* minute/ 144)
    df['m_sin']= np.sin(np.pi* 2* minute/ 144)
    df['h_cos']= np.cos(np.pi* 2* hour/ 24)
    df['h_sin']= np.sin(np.pi* 2* hour/ 24)
    
    return df
    