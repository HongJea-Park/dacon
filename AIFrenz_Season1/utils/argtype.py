# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:11:31 2020

@author: hongj
"""

import argparse
import re


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
        