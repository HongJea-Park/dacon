# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:00:03 2020

@author: hongj
"""

class Early_stopping():
    
    '''
    '''
    
    
    def __init__(self, patience, lowerbetter= True):
        
        self.stop= False
        self.count= 0
        self.patience= patience
        self.lowerbetter= lowerbetter
        
    
    def check_best(self, valid_measure, best_measure):
        
        '''
        '''
        
        if self.lowerbetter:
            
            if best_measure< valid_measure:
                self.count+= 1
                return best_measure, False
                
            else:
                self.count= 0
                return valid_measure, True
            
        else:
            
            if best_measure> valid_measure:
                self.count+= 1
                return best_measure, False
                
            else:
                self.count= 0
                return valid_measure, True
            
    
    def check_stop(self):
        
        '''
        '''
            
        if self.patience<= self.count:
            return True
        
        else:
            return False

