import pandas as pd
import numpy as np

def cost(v1, v2):
    return(((v1-v2)**2).mean())/2
def difx0(v1,v2):
    return(v1-v2).mean()
def difx1(v1,v2):
    return ((v1-v2)*v1).mean()

data = pd.read_csv("data_akbilgic.csv")
X = data.iloc[1:, 3:].values.astype('float64')
Y = data.iloc[1:, 2].values.astype('float64')

class Regressor(object):
    def __init__(self,theta0,theta1,alpha):
        self.theta0 = theta0
        self.theta1 = theta1
        self.alpha = alpha

    def predict(self, X):
        self.prediction = np.zeros(X.shape[0])
        for i in range(self.X.shape[0]):
            self.prediction[i] = self.theta0 + self.theta1*X[i]

        pass

    def __update(self, hy, y):
        #prediccio hy i real y
        temp0 = difx0(hy,y)
        self.theta1 = difx1(hy,y)
        self.theta0 = temp0

        pass

    def train(self, max_iter, epsilon):


        pass
