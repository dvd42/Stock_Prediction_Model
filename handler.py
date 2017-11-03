# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 19:10:07 2017

@author: Diego
"""

import numpy as np

# Split data to have a training and an evaluation set
def split_data(x, y, train_ratio=0.8):
    
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    
    n_train = int(np.floor(x.shape[0]*train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:]
    
    x_train = x[indices_train, :]
    y_train = y[indices_train]
    x_val = x[indices_val, :]
    y_val = y[indices_val]
    
    return x_train, y_train, x_val, y_val


#standarize data
def standarize(x_train, x_val):
    
    mean = x_train.mean(0)
    std = x_train.std(0)
    
    x_t = x_train -mean[None, :]
    x_t /=std[None, :]
    x_v = x_val -mean[None, :]
    x_v /= std[None, :]
    
    return (x_t, x_v)


