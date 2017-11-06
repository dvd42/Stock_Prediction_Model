import numpy as np
from matplotlib import pyplot as plt


def mean_squared_error(v1, v2):
    return((v1-v2)**2).mean()

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

    def plot(self,cost,verbose,scale,tag,alpha,epsilon):
       
        a = []
        b = []
        for i in range(len(cost)):
            a.append(cost[i])
            b.append(i)
        
        plt.title("Gradient descent for " + tag  + "\n alpha: " + str(alpha) + "\n epsilon: "  + str(epsilon))
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.plot(b,a)
        
        if verbose:
            plt.draw()
            plt.pause(0.5)
        else:
            plt.savefig("Custom/Descent/Scale " + str(scale) +"/Descent" + str(tag) + ".png",bbox_inches='tight')
        
        plt.close()
        
        

    def __update(self, y_train,prediction):
        #prediccio hy i real y
        temp1 = self.theta1 - self.difx1(y_train,prediction)
        temp0 = self.theta0 - self.difx0(y_train,prediction)
        self.old_theta0 = self.theta0
        self.old_theta1 = self.theta1
        self.theta0 = temp0
        self.theta1 = temp1
         

    def train(self, max_iter, epsilon,x_train,y_train):
        cost = []
        i = 0
        while(self.old_theta0 - self.theta0 > epsilon and self.old_theta1 - self.theta1 > epsilon and i < max_iter):
            prediction = self.predict(x_train)
            self.__update(y_train,prediction)
            cost.append(mean_squared_error(prediction,y_train))
            i += 1
    
        return cost