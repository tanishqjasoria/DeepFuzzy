import numpy as np
import tensorflow as tf
from random import sample
from matplotlib.pyplot import imshow
import matplotlib as plt

def save_mnist10():
    np.save('y_10input_onehot',  y_train_onehot[:,[1,3,5,7,9,11,13,15,17,19]])  
    np.save('x_10input_flat',  x_train_flat[:,[1,3,5,7,9,11,13,15,17,19]])
    np.save('y_10test_onehot',  y_test_onehot[:,[3,2,1,18,4,8,11,0,61,7]])  
    np.save('x_10test_flat',  x_test_flat[:,[3,2,1,18,4,8,11,0,61,7]] )
    
def load_mnist10():
    x_train_flat = np.load("x_10input_flat.npy")
    y_train_onehot = np.load("y_10input_onehot.npy")
    y_test_onehot = np.load('y_10test_onehot.npy')
    x_test_flat = np.load('x_10test_flat.npy')
    return x_train_flat, y_train_onehot, y_test_onehot, x_test_flat

def initialize(self, initializer = 'random'):
        W = []
        b = []
        for i in range(len(self.layers)-1):
            W.append(np.random.rand(self.layers[i+1],self.layers[i])*0.02)
            b.append(np.random.rand(self.layers[i+1],1))
            assert(W[i].shape == (self.layers[i+1], self.layers[i]))
            assert(b[i].shape == (self.layers[i+1], 1))
        self.parameters['W'] = W
        self.parameters['b'] = b
        
def sigma(x):
    return 1.0 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0,x)

def forwardProptrain(self):
    self.act=[]
    self.act.append(self.train_data)
    for i in range(len(self.layers)-2):
        z = np.dot(self.parameters['W'][i], self.act[-1])
        self.act.append(relu(z + self.parameters['b'][i]))  #relu
    self.act.append(softmax(np.dot(self.parameters['W'][len(self.layers)-2], self.act[-1])))

def forwardProptest(self):
    self.act=[]
    self.act.append(self.test_data)
    for i in range(len(self.layers)-2):
        z = np.dot(self.parameters['W'][i], self.act[-1])
        self.act.append(relu(z + self.parameters['b'][i]))  #relu
    self.act.append(softmax(np.dot(self.parameters['W'][len(self.layers)-2], self.act[-1])))
    
    
def compCosttrain(self):
    interm = np.dot(np.log(self.act[-1]).T,self.y_train_onehot)
    cost = -1.0/self.y_train_onehot.shape[1]*np.sum(np.trace(interm))
    print(cost)
    
def compCosttest(self):
    interm = np.dot(np.log(self.act[-1]).T,self.y_train_onehot)
    cost = -1.0/self.y_train_onehot.shape[1]*np.sum(np.trace(interm))
    print(cost)
    
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def backProp(self):
    dZ = self.act[-1] - self.y_train_onehot
    m = self.y_train_onehot.shape[1]
    dW = 1.0/m * np.dot(dZ, self.act[-2].T)
    db = 1.0/m * np.dot(dZ, self.act[-2].T)
    dA_prev = np.dot(self.parameters['W'][-1].T, dZ)
    self.parameters['W'][-1] = self.parameters['W'][-1] - learning_rate*dW
    self.parameters['b'][-1] = self.parameters['b'][-1] - learning_rate*db
    
    for i in reversed(range(len(self.layers)-2)):
        dA = dA_prev 
        dZ = linear_activation_backward(dA,self.act[i+1],"relu")
        dW = 1.0/m * np.dot(dZ, self.act[i].T)
        db = 1.0/m*  np.sum(np.array(dZ),axis=1,keepdims=True)
        dA_prev = np.dot(self.parameters['W'][i].T, dZ)
        self.parameters['W'][i] = self.parameters['W'][i] - learning_rate*dW
        self.parameters['b'][i] = self.parameters['b'][i] - learning_rate*db

def linear_activation_backward(dA,cache,activation):
    if(activation=="sigmoid"):
        act =  cache 
        return np.multiply(np.multiply(dA, act), 1-act)
    if(activation=="relu"):
        act = cache
        act[act>0] = 1
        act[act<0] = 0
        return np.multiply(dA, act)

def check_accuracy_train():
    y = test.tdy
    test.train_data = test.tdx[:,]
    test.y_train_onehot = test.tdy[:,:100]
    m = y.shape[1]
    forwardProptrain(test)
    y_hat = np.argmax(test.act[-1], axis = 0)
    pred = y_hat
    exp = np.argmax(y, axis = 0)
    error = np.sum(exp!=pred)
    # Calculate accuracy
    return (m - error)/m * 100

def check_accuracy_test():
    mat = np.zeros([10,10])
    y = test.y_test_onehot
    test.train_data = test.test_data[:,]
    test.y_train_onehot = test.y_test_onehot[:,:100]
    m = y.shape[1]
    forwardProptrain(test)
    y_hat = np.argmax(test.act[-1], axis = 0)
    pred = y_hat
    exp = np.argmax(y, axis = 0)
    p = exp!=pred
    error = np.sum(exp!=pred)
    for i in range(m):
        mat[exp[i]][pred[i]] =  mat[exp[i]][pred[i]] + 1
    # Calculate accuracy
    return (m - error)/m * 100, mat

