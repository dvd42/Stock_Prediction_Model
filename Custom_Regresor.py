import numpy as np


class Regressor(object):
    def __init__(self,theta0,theta1,alpha):
        self.theta0 = theta0
        self.theta1 = theta1
        self.alpha = alpha
        self.old_theta0 = 1000000
        self.old_theta1 = 1000000

    def difx0(self,y_train,prediction):
        return (self.theta0 + self.theta1 * prediction - y_train).mean() * self.alpha

    def difx1(self,y_train,prediction):
        return ((self.theta0 + self.theta1 * prediction - y_train)*prediction).mean() * self.alpha

    def predict(self, x_train):
        prediction = np.zeros(x_train.shape[0])
        for i in range(x_train.shape[0]):
            prediction[i] = self.theta1*x_train[i,0] + self.theta0 
        
        return prediction

    def __update(self, y_train,prediction):
        #prediccio hy i real y
        temp1 = self.theta1 - self.difx1(y_train,prediction)
        temp0 = self.theta0 - self.difx0(y_train,prediction)
        self.old_theta0 = self.theta0
        self.old_theta1 = self.theta1
        self.theta0 = temp0
        self.theta1 = temp1
         

    def train(self, max_iter, epsilon,x_train,y_train):
        while(self.old_theta0 - self.theta0 > epsilon and self.old_theta1 - self.theta1 > epsilon and max_iter > 0):
            print self.theta0
            print self.theta1
            prediction = self.predict(x_train)
            self.__update(y_train,prediction)
            max_iter -= 1
