# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 19:37:44 2017

@author: Diego
"""

from __future__ import print_function
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.linear_model import LinearRegression

import numpy as np

import handler as h
import Custom_Regresor as cr

# Evaluate hypotesis
def mean_squared_error(v1, v2):
    return((v1-v2)**2).mean()

# Call regression from sklearn
def regression(x,y):
    #obj reg sklearn
    regr = LinearRegression()

    # Call to train model with data x to obtain y
    regr.fit(x, y)
    return regr


def unstandardized_regression(X_train,Y_train, X_val,Y_val,ratio):
    #Regression with all the parameters at the same time
    r = regression(X_train, Y_train)
    return mean_squared_error(Y_val, r.predict(X_val))


# Regression with each parameter individually"""
def single_parameter_regression(X_train,Y_train,X_val,Y_val,ratio,iteration,variation,scale,tags,verbose,custom,alpha,epsilon,max_iter,path):
    
    color_list = ['red','dodgerblue','green','slateblue','lime','maroon','orange']
    
    error = []
    
    for i in range(X_train.shape[1]):
        x_t = X_train[:, i]
        x_v = X_val[:, i]
        x_t = np.reshape(x_t,(x_t.shape[0],1))
        x_v = np.reshape(x_v,(x_v.shape[0],1))
        
        
       
        if not custom:
            regr = regression(x_t, Y_train)
        else:
            regr = cr.Regressor(1,1,alpha)
            regr.train(max_iter,epsilon,X_train,Y_train)
        
        error.append(mean_squared_error(Y_val, regr.predict(x_v)))
                
        
        # Results plot
        if variation == 0:
            plt.figure(str(i) + " Ratio: " + str(ratio))
            plt.title("Regression with parameter: " + tags[i] + " Split Ratio: " + str(ratio))
            plt.scatter(x_v[:, 0], Y_val, c=color_list[i])
            plt.plot(x_v[:, 0], regr.predict(x_v), label="Prediction", c='black')
            plt.xlabel(str(tags[i]) + " Stock Index ")
            plt.ylabel("Istanbul Stock Index")
            if verbose:
                plt.draw()
                plt.pause(0.5)
            else:
                plt.savefig(path + "Regressions/Scale " + str(scale) + "/Regression " + str(tags[i]) + " " + str(ratio) + ".png",bbox_inches='tight')
            plt.close()
        
        if iteration == 1 and variation == 0:
            #Histogram plot
            plt.figure("Histogram " + str(i))
            plt.title("Histogram Attribute: " + tags[i])
            plt.hist(x_t,bins=13,range=[np.min(x_t[:,0]),np.max(x_t[:,0])],color=color_list[i])
            if verbose:
                plt.draw()
                plt.pause(0.5)
            else:
                plt.savefig("Histograms/Scale " + str(scale) + "/Histogram Attribute" + str(tags[i]) + ".png",bbox_inches='tight')
            plt.close()
            
    return error
        
       
def standardized_regression(X_train,Y_train, X_val, Y_val,ratio):
    
    # Regression with standarized attributes
    x_s_train, x_s_val = h.standarize(X_train,X_val)
    s_regr = regression(x_s_train,Y_train)
    
    return mean_squared_error(Y_val, s_regr.predict(x_s_val))
 

def draw_3d(validation,result,ratio,variation,verbose,scale,tags,min1,min2):
    regr = regression(validation, result)
    prediction = regr.predict(validation)

    # Add 1's
    A = np.hstack((validation,np.ones([prediction.shape[0],1])))
    W = np.linalg.lstsq(A,prediction)[0]


    # Create meshes attached to the dotted area to represent the surface
    malla = (range(20) + 0 * np.ones(20)) / 10
    malla_x1 = malla * (max(validation[:, 0]) - min(validation[:, 0])) / 2 + min(validation[:, 0])
    malla_x2 = malla * (max(validation[:, 1]) - min(validation[:, 1])) / 2 + min(validation[:, 1])

    #Pair ones in mesh x1 with the ones in mesh x2
    xplot , yplot = np.meshgrid(malla_x1,malla_x2)

    # Create the surface
    zplot = W[0] * xplot + W[1] * yplot + W[2]

    # Draw points and the surface     
    if variation == 0:
        plt3d = plt.figure('Best 2 attributes ' + str(ratio)).gca(projection='3d')
        plt3d.set_xlabel(tags[min1] + ' Stock Index')
        plt3d.set_ylabel(tags[min2] + ' Stock Index')
        plt3d.set_zlabel('Istanbul Stock Index')
        plt3d.plot_surface(xplot,yplot,zplot, color='red')
        plt3d.scatter(validation[:,0],validation[:,1],result)
        
        for angle in range(0, 361,30):
            plt3d.view_init(30, angle)
            if verbose:
                plt.draw()
                plt.pause(0.1)
            else:    
                plt.savefig("3D/Scale " + str(scale) + "/Best 2 attributes " + str(ratio) + " " + str(angle) + ".png",bbox_inches='tight')
        plt.close()
               
        
    
    return mean_squared_error(result,prediction)



def get_best_attributes(error,pos):
    
      min1 = error[pos].index(sorted(error[pos])[0])
      min2 = error[pos].index(sorted(error[pos])[1])
    
      return min1,min2
  
  
def store_mean_error(ratio,iteration,tags,verbose,variations,scale,X,Y,path,custom,alpha,epsilon,max_iter):
    
    
    error = []
    standardized_error = []
    error_single_parameter = []
    unstandardized_error = []
    
    for j in range(variations):
        x_train, y_train, x_val, y_val = h.split_data(X, Y,train_ratio = ratio)
        error_single_parameter.append(single_parameter_regression(x_train,y_train, x_val,y_val,ratio,iteration,j,scale,tags,verbose,custom,alpha,epsilon,max_iter,path))
        
        #Get the 2 attributes with the smallest error
        min1,min2 = get_best_attributes(error_single_parameter,j)
        
        error.append(draw_3d(x_val[:,[min1,min2]],np.reshape(y_val,(y_val.shape[0],1)),ratio,j,verbose,scale,tags,min1,min2))
        standardized_error.append(standardized_regression(x_train,y_train, x_val,y_val,ratio))
        unstandardized_error.append(unstandardized_regression(x_train,y_train, x_val,y_val,ratio))

    
    error = reduce(lambda x, y: x + y, error) / len(error)
    standardized_error = reduce(lambda x, y: x + y, standardized_error) / len(standardized_error)
    unstandardized_error = reduce(lambda x, y: x + y, unstandardized_error) / len(unstandardized_error)
    
    print ("Error best 2 attributtes (%s,%s): %f Ratio: %.1f " % (tags[min1],tags[min2],error,ratio),file=open("Errors/Best Error.txt","a+"))
    print ("Standardized Error %f Ratio: %.1f " % (standardized_error,ratio),file=open("Errors/Standarized Regression.txt","a+"))
    print ("Unstandardized Error: %f Ratio: %.1f " % (unstandardized_error,ratio),file=open("Errors/Unstandardized Regression.txt","a+"))
    
    for i in range(tags.size):
        error = [error_single_parameter[k][i] for k in range(variations)]
        print ("Error with attribute %s: %f Ratio: %.1f" % (tags[i],reduce(lambda x,y: x + y, error ) / variations,ratio),file=open(path + "Errors/Single Regression.txt","a+"))
    print("---------------------------------------------",file=open(path + "Errors/Single Regression.txt","a+"))

    








