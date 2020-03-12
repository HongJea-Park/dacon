# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 01:24:42 2020

@author: hongj
"""


import os


def mkdir(base_dir):
    
    base_split= base_dir.replace('../', '').split('/')
    
    folder= base_split[:-1]
    file= base_split[-1]
    
    dir_= '..'
    for f in folder:
        
        dir_+= '/%s'%f
        
        if not os.path.isdir(dir_):
            os.mkdir(dir_)
            
    return '%s/%s'%(dir_, file)
