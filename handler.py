# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 19:10:07 2017

@author: Diego
"""

from __future__ import print_function
import numpy as np
import sys
import os
import pandas as pd

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


def process_runtime_arguments():
    
    # Process command line parameters
    if len(sys.argv) < 2:
        print ("Usage: main.py, dataset.csv, -s scale ,-d variations,-v to store in wd plots")
        sys.exit(1)
 
    data = pd.read_csv(sys.argv[1])
    argvs = []
    
    for i in range(2,len(sys.argv)):
        argvs.append(sys.argv[i])
        
    
    verbose = True if "-v" in argvs else False
    scale = int(argvs[argvs.index('-s') + 1]) if "-s" in argvs else 1
    variations = int(argvs[argvs.index('-d') + 1]) if "-d" in argvs else 1
    
    
    return verbose,scale,variations,data

def create_dir(scale,verbose):
    
    if not os.path.exists("Errors"):
        os.makedirs("Errors")
    
    if not verbose:
        if not os.path.exists("3D/Scale " +str(scale)):
            os.makedirs("3D/Scale " +str(scale))
        
        if not os.path.exists("Regressions/Scale " + str(scale)):
            os.makedirs("Regressions/Scale " + str(scale))
        
        if not os.path.exists("Histograms/Scale " + str(scale)):
            os.makedirs("Histograms/Scale " + str(scale))



def add_file_header(scale,variations):
    
    print ("Running model with:",file=open("Errors/Best Error.txt","a+"))
    print ("Scale: %d " % scale,file=open("Errors/Best Error.txt","a+"))
    print ("Variations: %d" % variations,file=open("Errors/Best Error.txt","a+"))
    
    print ("Running model with:",file=open("Errors/Standarized Regression.txt","a+"))
    print ("Scale: %d " % scale,file=open("Errors/Standarized Regression.txt","a+"))
    print ("Variations: %d" % variations,file=open("Errors/Standarized Regression.txt","a+"))
    
    print ("Running model with:",file=open("Errors/Unstandardized Regression.txt","a+"))
    print ("Scale: %d " % scale,file=open("Errors/Unstandardized Regression.txt","a+"))
    print ("Variations: %d" % variations,file=open("Errors/Unstandardized Regression.txt","a+"))
    
    print ("Running model with:",file=open("Errors/Single Regression.txt","a+"))
    print ("Scale: %d " % scale,file=open("Errors/Single Regression.txt","a+"))
    print ("Variations: %d" % variations,file=open("Errors/Single Regression.txt","a+"))


def add_file_footer():
    
    print ("\n",file=open("Errors/Standarized Regression.txt","a+"))
    print ("\n",file=open("Errors/Unstandardized Regression.txt","a+"))
    print ("\n",file=open("Errors/Single Regression.txt","a+"))
    print ("\n",file=open("Errors/Best Error.txt","a+"))
        

