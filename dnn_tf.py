#!/usr/bin/env python
# coding: utf-8
# softmax cross-entropy loss function!
# categorical classification DNN

import matplotlib as plt
from matplotlib.pyplot import imshow
import numpy as np
from random import sample
import tensorflow as tf

class dnn:
  W = []
  b = []
  parameters = dict()
  act = []
  def __init__(self, layers, x_train, x_test, y_train, y_test, minibatch_size, learning_rate=0.0001, training_epochs=100):
    self.layers = layers
    self.x_test = x_test
    self.y_test = y_test
    self.x_train = x_train
    self.y_train = y_train
    self.batch = minibatch_size
    self.training_epochs = training_epochs
    self.learning_rate = learning_rate
    self.x = tf.placeholder("float", [None, self.layers[0]])
    self.y = tf.placeholder("float", [None, self.layers[-1]])                                                                                                                                                                                                                                                                                                                                                                                        
  def initialize(self, initializer = 'random'):
    print(self.layers)
    if initializer == 'random':
      for i in range(len(self.layers)-1):
        self.W.append(tf.Variable(tf.random_normal([self.layers[i], self.layers[i+1]])))
        self.b.append(tf.Variable(tf.random_normal([self.layers[i+1]])))
    elif initializer == 'xavier':
      pass
    self.parameters['W'] = self.W
    self.parameters['b'] = self.b
  def forwardProp(self):
    self.act=[]
    self.act.append(self.x)
    print(self.x.shape)
    for i in range(len(self.layers)-2):
      z = tf.matmul(self.act[-1], self.parameters['W'][i])
      self.act.append(tf.nn.relu(tf.add(z, self.parameters['b'][i])))  #relus
    self.act.append(tf.add((tf.matmul(self.act[-1], \
                                      self.parameters['W'][len(self.layers)-2])),self.parameters['b'][len(self.layers)-2]))
  def train(self):
    print(self.act[-1].shape)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.act[-1], labels=self.y))
    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for epoch in range(self.training_epochs):
        avg_cost = 0.0
        total_batch = int(len(self.x_train) / self.batch)
        x_train_batch = np.array_split(self.x_train, total_batch)
        y_train_batch = np.array_split(self.y_train, total_batch)
        for i in range(total_batch):
          batch_x, batch_y = x_train_batch[i], y_train_batch[i]
          buff, c = sess.run([optimizer, cost], 
                  feed_dict={
                    self.x: batch_x, 
                    self.y: batch_y,
                  })
          avg_cost += c / total_batch
        if epoch % 100 == 0:
          print("Epoch:", '%04d' % (epoch+1), "cost=", \
            "{:.9f}".format(avg_cost))
      print("Optimization Finished!")
      correct_prediction = tf.equal(tf.argmax(self.act[-1], 1), tf.argmax(self.y, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      print("Test Accuracy:", accuracy.eval({self.x: self.x_test, self.y: self.y_test}))
  # def saveWeights(self):
  #     np.save('W',self.parameters['W'])
  #     np.save('b',self.parameters['b'])
  # def loadWeights(self):
  #     try:
  #         W = np.load('W.npy')
  #         b = np.load('b.npy')
  #         self.parameters['W'] = W
  #         self.parameters['b'] = b
  #     except:
  #         print("Not able to load weights!")
  #         print("Initializing Weights...")
  #         self.initialize()
  #         print("Done!")
