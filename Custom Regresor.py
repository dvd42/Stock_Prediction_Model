import pandas as pd
import numpy as np
import handler as h

def cost(v1, v2):
    return(((v1-v2)**2).mean())/2
def difx0(v1,v2,alpha):
    return (v1-v2).mean() * alpha 
def difx1(v1,v2,alpha):
    return ((v1-v2)*v1).mean() * alpha


# Processing data from dataset

data = pd.read_csv("data_akbilgic.csv")    

X = data.iloc[1:, 3:].values.astype('float64')
X = (X+1) * 1000
Y = data.iloc[1:, 2].values.astype('float64')
Y = (Y+1) * 1000
tags = data.iloc[0,3:].values


class Regressor(object):
    def __init__(self,theta0,theta1,alpha):
        self.theta0 = theta0
        self.theta1 = theta1
        self.alpha = alpha
        
        
    def predict(self, x_train):
        self.prediction = np.zeros(x_train.shape[0])
        for i in range(x_train.shape[0]):
            self.prediction[i] = self.theta1*x_train[i,0] + self.theta0 
            

    def __update(self, hy, y):
        #prediccio hy i real y
        self.theta1 -= difx1(hy,y,self.alpha)
        self.theta0 -= difx0(hy,y,self.alpha)
        

    def train(self, max_iter, epsilon,x_train,y_train):
        
        while(self.theta0 > epsilon or self.theta1 > epsilon or max_iter != 0):
            print self.theta0
            print self.theta1
            self.predict(x_train)
            self.__update(self.prediction,y_train)
            max_iter -= 1


x_train,y_train,x_val,y_val = h.split_data(X,Y)

regressor = Regressor(10,10,0.00001)

regressor.train(10000,1,x_train,y_train)
regressor.predict(x_val)
print cost(regressor.prediction,y_val)