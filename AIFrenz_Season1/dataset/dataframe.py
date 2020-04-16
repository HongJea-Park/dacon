# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:18:27 2020

@author: hongj
"""


import pandas as pd
import numpy as np


class Dataframe():
    
    def __init__(self, data_dir):
        
        self.data_dir = data_dir

        self.feature_dict = {
                'feature_temp': 
                    ['X00', 'X07', 'X28', 'X31', 'X32'],
                'feature_solar': 
                    ['X11', 'X34'],
                'feature_rain': 
                    ['X04', 'X10', 'X21', 'X36', 'X39'],
                'feature_press': 
                    ['X01', 'X06', 'X22', 'X27', 'X29'],
#                'feature_sea': 
#                    ['X05', 'X08', 'X09', 'X23', 'X33'],
                'feature_wind': 
                    ['X02', 'X03', 'X18', 'X24', 'X26'],
                'feature_humidity': 
                    ['X12', 'X20', 'X30', 'X37', 'X38'],
#                'feature_dir': 
#                    ['X13', 'X15', 'X17', 'X25', 'X35']
                }
        
        self.derived_feature_dict = {
                'time':
                     ['time_0'],
#                'lagged_temp': 
#                    ['lagged_%s'%v 
#                         for v in self.feature_dict['feature_temp']],
                'lagged_solar': 
                    ['lagged_%s'%v 
                         for v in self.feature_dict['feature_solar']],
#                'lagged_rain': 
#                    ['lagged_%s'%v 
#                         for v in self.feature_dict['feature_rain']],
#                'lagged_press': 
#                    ['lagged_%s'%v 
#                         for v in self.feature_dict['feature_press']],
#                'lagged_sea': 
#                    ['lagged_%s'%v 
#                         for v in self.feature_dict['feature_sea']],
#                'lagged_wind':
#                    ['lagged_%s'%v 
#                         for v in self.feature_dict['feature_wind']],
#                'lagged_humidity': 
#                    ['lagged_%s'%v 
#                         for v in self.feature_dict['feature_humidity']],
#                'lagged_dir':
#                    ['lagged_%s'%v 
#                         for v in self.feature_dict['feature_dir']]
                }
            
        self.feature_dict.update(self.derived_feature_dict)
        
        if len(self.data_dir) > 0:
            self.train_df = pd.read_csv(
                    '%s/train.csv'%self.data_dir, index_col= 'id')
            self.derived_train_df = self.make_feature(self.train_df)
            self.test_df = pd.read_csv(
                    '%s/test.csv'%self.data_dir, index_col= 'id')
            self.derived_test_df = self.make_feature(self.test_df)
            
#        self.feature_dict['feature_rain'] = []
            
        self.feature_cols = []
        for key, cols in sorted(self.feature_dict.items()):
            self.feature_cols.extend(cols)
            
        self.n_cols = len(self.feature_cols)
        

    def get_time(self, df):

        df['time_0'] = np.cos((2* np.pi * ((df.index.values+64) % 144) / 144))
        
        return df
            
    
    def lagged_feature(self, df, cols, cum= False):
    
        df = df.copy(deep= True)
        
        for col in cols:
            
            df['lagged_%s'%col] = df[col] - df[col].shift(1)
            df['lagged_%s'%col] = df['lagged_%s'%col].fillna(0)
            
            if cum:
                df.loc[df['lagged_%s'%col] < 0, 'lagged_%s'%col] = 0
            
        return df
    
    
    def make_feature(self, df):
        
        df = df.copy(deep= True)
        
        df = self.get_time(df)
#        df = self.lagged_feature(
#                df, self.feature_dict['feature_temp'])
        df = self.lagged_feature(
                df, self.feature_dict['feature_solar'], cum= True)
#        df = self.lagged_feature(
#                df, self.feature_dict['feature_rain'], cum= True)
#        df = self.lagged_feature(
#                df, self.feature_dict['feature_press'])
#        df = self.lagged_feature(
#                df, self.feature_dict['feature_sea'])
#        df = self.lagged_feature(
#                df, self.feature_dict['feature_wind'])
#        df = self.lagged_feature(
#                df, self.feature_dict['feature_humidity'])
#        df = self.lagged_feature(
#                df, self.feature_dict['feature_dir'])

        return df
    
    
    def preprocessing(self, df):
        
        df = df.copy(deep= True)
        
#        self.standard_normalization(
#                df, self.feature_dict['time'])
                
        self.standard_normalization(
                df, self.feature_dict['feature_temp'])
        self.minmax_normalization(
                df, self.feature_dict['feature_solar'])
        self.log_normalization(
                df, self.feature_dict['feature_rain'])
        self.standard_normalization(
                df, self.feature_dict['feature_press'])
#        self.standard_normalization(
#                df, self.feature_dict['feature_sea'])
        self.minmax_normalization(
                df, self.feature_dict['feature_wind'])
        self.minmax_normalization(
                df, self.feature_dict['feature_humidity'])
#        self.minmax_normalization(
#                df, self.feature_dict['feature_dir'])

#        self.standard_normalization(
#                df, self.feature_dict['lagged_temp'])
        self.minmax_normalization(
                df, self.feature_dict['lagged_solar'])        
#        self.log_normalization(
#                df, self.feature_dict['lagged_rain'])
#        self.standard_normalization(
#                df, self.feature_dict['lagged_press'])
#        self.standard_normalization(
#                df, self.feature_dict['lagged_sea'])
#        self.standard_normalization(
#                df, self.feature_dict['lagged_wind'])
#        self.standard_normalization(
#                df, self.feature_dict['lagged_humidity'])
#        self.standard_normalization(
#                df, self.feature_dict['lagged_dir'])
            
        return df
    
    
    def standard_normalization(self, df, col_list):
        
        train_df = self.derived_train_df.copy(deep= True)
        
        mean = train_df[col_list].mean(axis= 0)
        std = train_df[col_list].std(axis= 0) + 1e-256
        
        df[col_list] = (df[col_list] - mean) / std
            
        
    def minmax_normalization(self, df, col_list):
        
        train_df = self.derived_train_df.copy(deep= True)
        
        min_ = train_df[col_list].min(axis= 0)
        max_ = train_df[col_list].max(axis= 0)
        
        train_df[col_list] = (train_df[col_list] - min_) / (max_ - min_)
        
        df[col_list] = (df[col_list] - min_) / (max_ - min_)
    
    
    def log_normalization(self, df, col_list):
        
        train_df = self.derived_train_df.copy(deep= True)
        
        train_df[col_list] = np.log(train_df[col_list] + 1)

        min_ = train_df[col_list].min(axis= 0)
        max_ = train_df[col_list].max(axis= 0)
        
        df[col_list] = np.log(df[col_list] + 1)
        df[col_list] = (df[col_list] - min_) / (max_ - min_)
    
    
    def get_pretrain_df(self):
        
        df = self.derived_train_df.copy(deep= True)
        df = self.preprocessing(df)
        
        y_cols = ['Y%s'%str(i).zfill(2) for i in range(18)]
        
        df = df[self.feature_cols + y_cols].dropna()
        
        df = pd.concat(
                objs= [df[self.feature_cols], df[y_cols]], 
                axis= 1).dropna()

        return df
    
    
    def get_y18_df(self):
        
        df = self.derived_train_df.copy(deep= True)
        df = self.preprocessing(df)
        
        y = df['Y18'].dropna()

        df = pd.concat(
                objs= [df[self.feature_cols], y], 
                axis= 1).dropna()

    
        return df
        
    
    def get_test_df(self):
        
        df = self.derived_test_df.copy(deep= True)
        df = df[self.feature_cols]
        df = self.preprocessing(df)
        
        return df
