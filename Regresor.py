import numpy as np
import pandas as pd
import sys

import functions as f


# Process command line parameters
if len(sys.argv) < 2:
    print "Usage: main.py, dataset.csv, verbose ,variations"
    exit(1)

if len(sys.argv) < 3:
    variations = 100
    verbose = False
    
elif "-v" == sys.argv[2]:
    verbose = True
    
else:
    variations = int(sys.argv[2])
   
dataset = sys.argv[1]


variations = 20
verbose = False
#Processing data from dataset
data = pd.read_csv("data_akbilgic.csv")
X = data.iloc[1:, 3:].values.astype('float64')
X = (X+1) * 1000
Y = data.iloc[1:, 2].values.astype('float64')
Y = (Y+1) * 1000
tags = data.iloc[0,3:].values



#We generate evaluation and training set randomly
for i in range(1,5):
        
    error_2attributes = []
    error_standardized = []
    error_unstandardized = []
    error_single_parameter = []
    ratio = 0.9 - i/float(10)
    
    for j in range(variations):
        x_train, y_train, x_val, y_val = f.split_data(X, Y,train_ratio = ratio)
        
        # TODO draw 3d plot with the 2 best attributes 
        error_2attributes.append(f.draw_3d(x_val[:,:2],np.reshape(y_val,(y_val.shape[0],1)),ratio,j))
        error_single_parameter.append(f.single_parameter_regression(x_train,y_train, x_val,y_val,ratio,i,j,tags,verbose))
        error_standardized.append(f.standardized_regression(x_train,y_train, x_val,y_val,ratio))
        error_unstandardized.append(f.unstandardized_regression(x_train,y_train, x_val,y_val,ratio))

    print "Error 2 attributtes " + str(reduce(lambda x, y: x + y, error_2attributes) / len(error_2attributes)) + " " + str(ratio)
    print "Standardized Error: " + str(reduce(lambda x, y: x + y, error_standardized) / len(error_standardized)) + " " + str(ratio)
    print "Unstandardized Error: " + str(reduce(lambda x, y: x + y, error_unstandardized) / len(error_unstandardized)) + " " + str(ratio)
    
    for i in range(tags.size):
        print "Error with attribute: " + str(tags[i]) + " " + str(reduce(lambda x,y: x + y, [error_single_parameter[k][i] for k in range(variations)]) / variations) + " " + str(ratio)
        
    
    
    
    
    
    