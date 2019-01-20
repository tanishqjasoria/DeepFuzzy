#!/usr/bin/env python
# coding: utf-8

import matplotlib as plt
from matplotlib.pyplot import imshow
import numpy as np
from random import sample

class dnn:
    W = []
    b = []
    parameters = dict()
    act = []
    def __init__(self, layers, x_train, x_test, y_train, y_test, minibatch_size, learning_rate=0.01, iterations=100):
        self.layers = layers
        self.x_train_batch = x_train[:, :minibatch_size]
        self.x_test = x_test
        self.y_train_batch = y_train[:, :minibatch_size]
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train
        self.batch = minibatch_size
        self.iter = iterations
        self.learning_rate = learning_rate
    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    @staticmethod
    def relu(x):
        return np.maximum(0,x)
    @staticmethod
    def linear_activation_backward(dA,cache,activation):
        if(activation=="sigmoid"):
            act =  cache 
            return np.multiply(np.multiply(dA, act), 1-act)
        if(activation=="relu"):
            act = cache
            act[act>0] = 1
            act[act<0] = 0
            return np.multiply(dA, act)
    def initialize(self, initializer = 'random'):
        if initializer == 'random':
            for i in range(len(self.layers)-1):
                self.W.append(np.random.rand(self.layers[i+1],self.layers[i])*0.02)
                self.b.append(np.random.rand(self.layers[i+1],1))
                assert(self.W[i].shape == (self.layers[i+1], self.layers[i]))
                assert(self.b[i].shape == (self.layers[i+1], 1))
        elif initializer == 'xavier':
            pass
        self.parameters['W'] = self.W
        self.parameters['b'] = self.b
    def forwardProp(self):
        self.act=[]
        self.act.append(self.x_train_batch)
        for i in range(len(self.layers)-2):
            z = np.dot(self.parameters['W'][i], self.act[-1])
            self.act.append(dnn.relu(z + self.parameters['b'][i]))  #relu
        self.act.append(dnn.softmax(np.dot(self.parameters['W'][len(self.layers)-2], self.act[-1])))
    def compCost(self):
        interm = np.dot(np.log(self.act[-1]).T,self.y_train_batch)
        cost = -1.0/self.y_train_batch.shape[1]*np.sum(np.trace(interm))
        return cost
    def backProp(self):
        m = self.y_train_batch.shape[1]
        dZ = self.act[-1] - self.y_train_batch
        dW = 1.0/m * np.dot(dZ, self.act[-2].T)
        db = 1.0/m * np.dot(dZ, self.act[-2].T)
        dA_prev = np.dot(self.parameters['W'][-1].T, dZ)
        self.parameters['W'][-1] = self.parameters['W'][-1] - self.learning_rate*dW
        self.parameters['b'][-1] = self.parameters['b'][-1] - self.learning_rate*db
        for i in reversed(range(len(self.layers)-2)):
            dA = dA_prev 
            dZ = dnn.linear_activation_backward(dA,self.act[i+1],"relu")
            dW = 1.0/m * np.dot(dZ, self.act[i].T)
            db = 1.0/m*  np.sum(np.array(dZ),axis=1,keepdims=True)
            dA_prev = np.dot(self.parameters['W'][i].T, dZ)
            self.parameters['W'][i] = self.parameters['W'][i] - self.learning_rate*dW
            self.parameters['b'][i] = self.parameters['b'][i] - self.learning_rate*db
    def check_accuracy(self, y, x):
        mat = np.zeros([10,10])
        m = y.shape[1]
        buff_x = self.x_train_batch
        self.x_train_batch = x[:,]
        self.forwardProp()
        pred = np.argmax(self.act[-1], axis = 0)
        exp = np.argmax(y, axis = 0)
        error = np.sum(exp!=pred)
        self.x_train_batch = buff_x
        for i in range(m):
            mat[exp[i]][pred[i]] =  mat[exp[i]][pred[i]] + 1
        # Calculate accuracy
        return (m - error)/m * 100, mat
    def train(self):
        for i in range(self.iter):
                if(i%2==0):
                    idx = np.random.randint(self.x_train.shape[1], size=self.batch)
                    self.x_train_batch = self.x_train[:,idx]
                    self.y_train_batch = self.y_train[:,idx]
                self.forwardProp()
                cost = self.compCost()
                self.backProp()
                if(i%100 == 0):
                    Accuracy, _ = self.check_accuracy(self.y_train,self.x_train)
                    print("Accuracy: ", Accuracy)
                    self.saveWeights()
    def test(self):
        self.forwardProp()
        Accuracy, mat = self.check_accuracy(self.y_test,self.x_test)
        print("Test Accuracy", Accuracy )
        print("Confusion Matrix:")
        print(mat)
    def saveWeights(self):
        np.save('W',self.parameters['W'])
        np.save('b',self.parameters['b'])
    def loadWeights(self):
        try:
            W = np.load('W.npy')
            b = np.load('b.npy')
            self.parameters['W'] = W
            self.parameters['b'] = b
        except:
            print("Not able to load weights!")
            print("Initializing Weights...")
            self.initialize()
            print("Done!")
