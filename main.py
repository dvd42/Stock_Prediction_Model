from __future__ import print_function
import numpy as np
import pandas as pd
import sys
import os

import regression as r
import handler as h


# Process command line parameters
if len(sys.argv) < 2:
    print ("Usage: main.py, dataset.csv, -s scale ,-d variations,-v to store in wd plots")
    exit(1)


argvs = []

for i in range(2,len(sys.argv)):
    argvs.append(sys.argv[i])
    

verbose = True if "-v" in argvs else False
scale = int(argvs[argvs.index('-s') + 1]) if "-s" in argvs else 1
variations = int(argvs[argvs.index('-d') + 1]) if "-d" in argvs else 1

if not os.path.exists("Errors"):
    os.makedirs("Errors")

if not os.path.exists("3D"):
    os.makedirs("3D")

if not os.path.exists("Regressions"):
    os.makedirs("Regressions")

if not os.path.exists("Histograms"):
    os.makedirs("Histograms")
                
                
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

print ("Running model with:")
print ("Scale: %d " % scale)
print ("Variations: %d" % variations)



dataset = sys.argv[1]

#Processing data from dataset
data = pd.read_csv(dataset)
X = data.iloc[1:, 3:].values.astype('float64')
X = (X+1) * scale
Y = data.iloc[1:, 2].values.astype('float64')
Y = (Y+1) * scale
tags = data.iloc[0,3:].values



#We generate evaluation and training set randomly
for i in range(1,5):
        
    error = []
    standardized_error = []
    unstandardized_error = []
    error_single_parameter = []
    ratio = 0.9 - i/float(10)
    
    for j in range(variations):
        x_train, y_train, x_val, y_val = h.split_data(X, Y,train_ratio = ratio)
        error_single_parameter.append(r.single_parameter_regression(x_train,y_train, x_val,y_val,ratio,i,j,tags,verbose))
        
        #Get the 2 attributes with the smallest error
        min1 = error_single_parameter[j].index(sorted(error_single_parameter[j])[0])
        min2 = error_single_parameter[j].index(sorted(error_single_parameter[j])[1])
        
        error.append(r.draw_3d(x_val[:,[min1,min2]],np.reshape(y_val,(y_val.shape[0],1)),ratio,j,verbose,tags,min1,min2))
        standardized_error.append(r.standardized_regression(x_train,y_train, x_val,y_val,ratio))
        unstandardized_error.append(r.unstandardized_regression(x_train,y_train, x_val,y_val,ratio))

    
    error = reduce(lambda x, y: x + y, error) / len(error)
    standardized_error = reduce(lambda x, y: x + y, standardized_error) / len(standardized_error)
    unstandardized_error = reduce(lambda x, y: x + y, unstandardized_error) / len(unstandardized_error)
    
    print ("Error best 2 attributtes (%s,%s): %f Ratio: %.1f " % (tags[min1],tags[min2],error,ratio),file=open("Errors/Best Error.txt","a+"))
    print ("Standardized Error %f Ratio: %.1f " % (standardized_error,ratio),file=open("Errors/Standarized Regression.txt","a+"))
    print ("Unstandardized Error: %f Ratio: %.1f " % (unstandardized_error,ratio),file=open("Errors/Unstandardized Regression.txt","a+"))
    
    for i in range(tags.size):
        error = [error_single_parameter[k][i] for k in range(variations)]
        print ("Error with attribute %s: %f Ratio: %.1f" % (tags[i],reduce(lambda x,y: x + y, error ) / variations,ratio),file=open("Errors/Single Regression.txt","a+"))
    print("---------------------------------------------",file=open("Errors/Single Regression.txt","a+"))
    
    
print ("\n",file=open("Errors/Standarized Regression.txt","a+"))
print ("\n",file=open("Errors/Unstandardized Regression.txt","a+"))
print ("\n",file=open("Errors/Single Regression.txt","a+"))
print ("\n",file=open("Errors/Best Error.txt","a+"))
    
    
    
    