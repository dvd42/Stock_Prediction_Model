import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import math
from sklearn.linear_model import LinearRegression
import pandas as pd


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


def unstandardized_regression(X_train,Y_train, X_val,Y_val,ratio):
    #Regression with all the parameters at the same time
    r = regression(X_train, Y_train)
    return mean_squared_error(Y_val, r.predict(X_val))


# Regression with each parameter individually"""
def single_parameter_regression(X_train,Y_train,X_val,Y_val,ratio,iteration,variation,plot_histogram=False,plot_results=True):
    
    color_list = ['red','dodgerblue','green','slateblue','lime','maroon','orange']
    
    tags = data.iloc[0,3:].values
    error = []
    
    for i in range(X_train.shape[1]):
        x_t = X_train[:, i]
        x_v = X_val[:, i]
        x_t = np.reshape(x_t,(x_t.shape[0],1))
        x_v = np.reshape(x_v,(x_v.shape[0],1))
        
        regr = regression(x_t, Y_train)
        error.append(mean_squared_error(Y_val, regr.predict(x_v)))
                
        if plot_results and variation == 0:
            # Results plot
            plt.figure(str(i) + " Ratio: " + str(ratio))
            plt.title("Regression with parameter: " + tags[i] + " Split Ratio: " + str(ratio))
            plt.scatter(x_v[:, 0], Y_val, c=color_list[i])
            plt.plot(x_v[:, 0], regr.predict(x_v), label="Prediction", c='black')
            plt.show()

        
        if plot_histogram and iteration == 1 and variation == 0:
            #Histogram plot
            plt.figure("Histogram " + str(i))
            plt.title("Histogram Attribute: " + tags[i])
            plt.hist(x_t,bins=13,range=[np.min(x_t[:,0]),np.max(x_t[:,0])],color=color_list[i])
            plt.show()
            
    return error
        
       
def standardized_regression(X_train,Y_train, X_val, Y_val,ratio):
    
    # Regression with standarized attributes
    x_s_train, x_s_val = standarize(X_train,X_val)
    s_regr = regression(x_s_train,Y_train)
    
    return mean_squared_error(Y_val, s_regr.predict(x_s_val))
 

def draw_3d(validation,result,ratio,variation):
    regr = regression(validation, result)
    prediction = regr.predict(validation)

    #add 1's
    A = np.hstack((validation,np.ones([prediction.shape[0],1])))
    W = np.linalg.lstsq(A,prediction)[0]

    #draw
    #crear malla acoplada a la zona de punts per a representar el pla
    malla = (range(20) + 0 * np.ones(20)) / 10
    malla_x1 = malla * (max(validation[:, 0]) - min(validation[:, 0])) / 2 + min(validation[:, 0])
    malla_x2 = malla * (max(validation[:, 1]) - min(validation[:, 1])) / 2 + min(validation[:, 1])

    #aparellem els uns de la malla x1 amb els de la malla x2
    xplot , yplot = np.meshgrid(malla_x1,malla_x2)

    #creem la superficie
    zplot = W[0] * xplot + W[1] * yplot + W[2]

    #dibuixem punts i superficie
     
    if variation == 0:
        plt3d = plt.figure('Prova 3d ' + str(ratio)).gca(projection='3d')
        plt3d.plot_surface(xplot,yplot,zplot, color='red')
        plt3d.scatter(validation[:,0],validation[:,1],result)
        plt.show()
        
    return mean_squared_error(result,prediction)


#reading data from dataset
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
    
    for j in range(20):
        x_train, y_train, x_val, y_val = split_data(X, Y,train_ratio = ratio)
        
        #TODO draw 3d plot with the 2 best attributes 
        # TODO print mean error for each attribute
        #error_2attributes.append(draw_3d(x_val[:,:2],np.reshape(y_val,(y_val.shape[0],1)),ratio,j))
        #error_standardized.append(standardized_regression(x_train,y_train, x_val,y_val,ratio))
        error_single_parameter.append(single_parameter_regression(x_train,y_train, x_val,y_val,ratio,i,j,plot_histogram=True,plot_results=True))
        #error_unstandardized.append(unstandardized_regression(x_train,y_train, x_val,y_val,ratio))

    #print "Error 2 attributtes " + str(reduce(lambda x, y: x + y, error_2attributes) / len(error_2attributes)) + " " + str(ratio)
    #print "Standardized Error: " + str(reduce(lambda x, y: x + y, error_standardized) / len(error_standardized)) + " " + str(ratio)
    #print "Unstandardized Error: " + str(reduce(lambda x, y: x + y, error_unstandardized) / len(error_unstandardized)) + " " + str(ratio)
    print 
    
    
    
    
    
    
    