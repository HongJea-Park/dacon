# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:11:31 2020

@author: hongj
"""

import argparse
import re
import numpy as np


def restricted_float(x):
    
    '''
    '''
    
    try:
        x = float(x)
        
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
        
    return x


def boolean(x):
    
    '''
    '''
    
    return str(x).lower()== 'true'


def check_float(x):
    
    '''
    '''
    
    regex= re.compile('1e-[0-9]+')
    
    if regex.match(x):
        
        return x
        
    else:
        
        raise argparse.ArgumentTypeError("%r not a float type" % (x,))
        
        
def str_to_list(x):
    
    '''
    '''
    
    x= np.array(x.replace(' ', '').split(','))    
    Y_list= np.array(['Y%s'%str(i).zfill(2) for i in range(18)])
    
    Y_list= Y_list[np.in1d(Y_list, x)]
    
    return Y_list.tolist()
            