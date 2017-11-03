# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 19:37:44 2017

@author: Diego
"""

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np


import handler as h

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
def single_parameter_regression(X_train,Y_train,X_val,Y_val,ratio,iteration,variation,tags,verbose):
    
    color_list = ['red','dodgerblue','green','slateblue','lime','maroon','orange']
    
    error = []
    
    for i in range(X_train.shape[1]):
        x_t = X_train[:, i]
        x_v = X_val[:, i]
        x_t = np.reshape(x_t,(x_t.shape[0],1))
        x_v = np.reshape(x_v,(x_v.shape[0],1))
        
        regr = regression(x_t, Y_train)
        error.append(mean_squared_error(Y_val, regr.predict(x_v)))
                
        
        # Results plot
        if verbose and variation == 0:
            plt.figure(str(i) + " Ratio: " + str(ratio))
            plt.title("Regression with parameter: " + tags[i] + " Split Ratio: " + str(ratio))
            plt.scatter(x_v[:, 0], Y_val, c=color_list[i])
            plt.plot(x_v[:, 0], regr.predict(x_v), label="Prediction", c='black')
            plt.xlabel(str(tags[i]) + " Stock Index ")
            plt.ylabel("Istanbul Stock Index")
            plt.savefig("Regressions/Regression with parameter" + str(tags[i]) + str(ratio) + ".png",bbox_inches='tight')
            plt.close()
        
        if iteration == 1 and variation == 0 and verbose:
            #Histogram plot
            plt.figure("Histogram " + str(i))
            plt.title("Histogram Attribute: " + tags[i])
            plt.hist(x_t,bins=13,range=[np.min(x_t[:,0]),np.max(x_t[:,0])],color=color_list[i])
            plt.savefig("Histograms/Histogram Attribute" + str(tags[i]) + ".png",bbox_inches='tight')
            plt.close()
            
    return error
        
       
def standardized_regression(X_train,Y_train, X_val, Y_val,ratio):
    
    # Regression with standarized attributes
    x_s_train, x_s_val = h.standarize(X_train,X_val)
    s_regr = regression(x_s_train,Y_train)
    
    return mean_squared_error(Y_val, s_regr.predict(x_s_val))
 

def draw_3d(validation,result,ratio,variation,verbose,tags,min1,min2):
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
    if variation == 0 and verbose:
        plt3d = plt.figure('Best 2 attributes ' + str(ratio)).gca(projection='3d')
        plt3d.set_xlabel(tags[min1] + ' Stock Index')
        plt3d.set_ylabel(tags[min2] + ' Stock Index')
        plt3d.set_zlabel('Istanbul Stock Index')
        plt3d.plot_surface(xplot,yplot,zplot, color='red')
        plt3d.scatter(validation[:,0],validation[:,1],result)
        
        for angle in range(0, 361,60):
            plt3d.view_init(30, angle)
            plt.savefig('3D/Best 2 attributes ' + str(ratio) + str(angle) + ".png",bbox_inches='tight')
        plt.close()
               
        
    
    return mean_squared_error(result,prediction)
