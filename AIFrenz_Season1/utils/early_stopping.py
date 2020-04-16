# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:00:03 2020

@author: hongj
"""

class Early_stopping():
    
    """
    """ 
    
    def __init__(self, patience, delta= 0):
        
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.is_best = False
        
    
    def __call__(self, valid_loss):
        
        """
        """
            
#        score = -valid_loss

        if self.best_score is None:
            
            self.best_score = valid_loss
            self.is_best = True
            
        elif valid_loss > self.best_score + self.delta:
            
            self.counter += 1
            self.is_best = False
            
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            self.best_score = valid_loss
            self.counter = 0
            self.is_best = True

        print('\r EarlyStopping', \
              '>'*self.counter + '|'*(self.patience-self.counter), '| \n')
        
        return self.early_stop, self.is_best


