import numpy as np
from matplotlib import pyplot as plt
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


# Reading data from dataset"""
data = pd.read_csv("data_akbilgic.csv")
X = data.iloc[1:, 3:].values.astype('float64')
Y = data.iloc[1:, 2].values.astype('float64')
error = np.ones(np.shape(Y))

#We generate evaluation and training set randomly 
x_train, y_train, x_val, y_val = split_data(X, Y)


def unstandardized_regression():
    #Regression with all the parameters at the same time
    r = regression(x_train, y_train)
    error = mean_squared_error(y_val, r.predict(x_val))
    print("Error Unstandarized ", error)
    
    # Prediction Plot
    plt.title("Regression on all unstandardized parameters")
    plt.plot(y_val,label="Test Set",c='red')
    plt.plot(r.predict(x_val),label="Prediction",c='blue')
    plt.legend()
    plt.show()
    
    
# Regression with each parameter individually"""
def single_parameter_regression(plot_histogram=False,plot_results=True):
    
    color_list = ['red','dodgerblue','green','slateblue','lime','maroon','orange']
    
    tags = data.iloc[0,3:].values
    
    for i in range(x_train.shape[1]):
        x_t = x_train[:, i]
        x_v = x_val[:, i]
        x_t = np.reshape(x_t,(x_t.shape[0],1))
        x_v = np.reshape(x_v,(x_v.shape[0],1))
        
        regr = regression(x_t, y_train)
        error = mean_squared_error(y_val, regr.predict(x_v))
        
        if plot_results:
            # Results plot
            plt.figure(i)
            plt.title("Regression with parameter: " + tags[i])
            plt.scatter(x_v[:, 0], y_val,c=color_list[i])
            plt.plot(x_v[:, 0], regr.predict(x_v), c='black')
            plt.show()

            
            print("Error for attribute ",tags[i], error)
        
        if plot_histogram:
            #Histogram plot
            plt.figure(i * 7)
            plt.title("Histogram Attribute: " + tags[i])
            plt.hist(x_t,bins=13,range=[np.min(x_t[:,0]),np.max(x_t[:,0])],color=color_list[i])
            plt.show()
            
       
def standardized_regression():
    
    # Regression with standarized attributes
    x_s_train, x_s_val = standarize(x_train,x_val)
    s_regr = regression(x_s_train,y_train)
    error = mean_squared_error(y_val, s_regr.predict(x_s_val))
    
    #Results plot
    plt.title("Regression on all standarized parameters")
    plt.plot(y_val,label="Test Set",c='red')
    plt.plot(s_regr.predict(x_s_val),label="Prediction",c='blue')
    plt.legend()
    plt.show()


    
    print("Error standarized ", error)


single_parameter_regression(plot_histogram=False,plot_results=True)
unstandardized_regression()
standardized_regression()

