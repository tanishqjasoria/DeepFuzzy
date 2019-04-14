#!/usr/bin/env python
# coding: utf-8

# In[268]:


import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow.keras.datasets as datasets
from torch_dataload import MyDataset
import nn_fuzzy
import matplotlib.pyplot as plt
import time
import math


# In[269]:


def pi_membership_function(r,c,radius):
    norm = abs(r-c)
    if radius == 0:
        return np.zeros(norm.shape)
    else:
        for i in range(len(r)):
            if norm[i] <= radius and norm[i] >= radius/2:
                norm[i] = (2*((1-norm[i]/radius)**2))
#                 print("Case 1")
            elif norm[i] < radius/2 and norm[i] >= 0:
                norm[i] = (1 - 2*((norm[i]/radius)**2))
#                 print("Case 2")
            else:
                norm[i] = 0
#                 print("Case 3")
    return norm


# In[270]:


def input_features(x_train, iid = 1):
    fdenom = 1
    if iid == 1:
        F_max = np.array([255] * x_train.shape[1])
        F_min = np.array([0] * x_train.shape[1])
    else:
        F_max = np.ndarray.max(x_train, axis = 0)
        F_min = np.ndarray.min(x_train, axis = 0)
    lambda_medium = 0.5*(F_max - F_min)
    c_medium = F_min + lambda_medium
#     print(c_medium)
    lambda_low = (1/fdenom)*(c_medium-F_min)
#     print(lambda_low)
    c_low = c_medium - 0.5 * lambda_low
#     print(c_low)
    lambda_high = (1/fdenom) * (F_max - c_medium)
#     print(lambda_high)
    c_high = c_medium + 0.5 * lambda_high
#     print(c_high)
    features = {
        'c_low' : c_low,
        'c_medium' : c_medium,
        'c_high' : c_high,
        'lambda_low' : lambda_low,
        'lambda_medium' : lambda_medium,
        'lambda_high' : lambda_high,
    }
    return features
    


# In[271]:


def input_fuzzify(x_train, features, cnn = 0):
    c_low = features['c_low']
    c_medium = features['c_medium']
    c_high = features['c_high']
    lambda_low = features['lambda_low']
    lambda_medium = features['lambda_medium']
    lambda_high = features['lambda_high']
    x_train = x_train.T
    if cnn == 1:
        x_train_low = []
        x_train_medium = []
        x_train_high = []
        for i in range(x_train.shape[0]):
            x_train_low.append(pi_membership_function(x_train[i],c_low[i],lambda_low[i]))
            x_train_medium.append(pi_membership_function(x_train[i],c_medium[i],lambda_medium[i]))
            x_train_high.append(pi_membership_function(x_train[i],c_high[i],lambda_high[i]))
        x_train_new = np.stack([np.array(x_train_low).T, np.array(x_train_medium).T, np.array(x_train_high).T], axis = 1)
        return np.array(x_train_new)
    else:
        x_train_new = []
        for i in range(x_train.shape[0]):
            x_train_new.append(pi_membership_function(x_train[i],c_low[i],lambda_low[i]))
            x_train_new.append(pi_membership_function(x_train[i],c_medium[i],lambda_medium[i]))
            x_train_new.append(pi_membership_function(x_train[i],c_high[i],lambda_high[i]))
        return np.array(x_train_new).T


# In[272]:


def getDataset(name, nClass):
    if name=="mnist":
        dataset = datasets.mnist
    (x_train, y_train),(x_test, y_test) = dataset.load_data()     #downloading and loading the dataset
    x_train, x_test = x_train, x_test            #normalizing the input data
    x_train_flat = x_train.reshape(x_train.shape[0],-1)         #making dataset suitable for input in Fully Connected layer
    x_test_flat = x_test.reshape(x_test.shape[0],-1)          #making dataset suitable for input in Fully Connected layer
    y_train_onehot = np.eye(nClass)[y_train]                    #converting to one hot vectors
    y_test_onehot = np.eye(nClass)[y_test]                     #converting to one hot vectors
    print(x_train_flat.shape)
    print(y_train_onehot.shape)
    x_train_batch = np.array_split(x_train_flat, int(60000/128))
    print(x_train_batch[2].shape)
    return x_train_flat,x_test_flat,y_train_onehot,y_test_onehot

x_train, x_test, y_train, y_test = getDataset("mnist", 10)


# In[273]:


features = input_features(x_train)


# In[282]:


x_train_fuzzy = input_fuzzify(x_train, features, cnn =1)


# In[283]:


x_train_fuzzy.shape


# In[284]:


x_train_fuzzy = x_train_fuzzy.reshape(-1, 3, 28, 28)


# In[285]:


index = 0
plt.imshow(x_train[index].reshape(28,28), cmap='Greys')


# In[286]:


plt.imshow(x_train_fuzzy[index][0], cmap='Greys')


# In[287]:


plt.imshow(x_train_fuzzy[index][1].reshape(28,28), cmap='Greys')


# In[288]:


plt.imshow(x_train_fuzzy[index][2].reshape(28,28), cmap='Greys')


# In[289]:


np.save('x_train_fuzzy', x_train_fuzzy)


# In[290]:


x_test_fuzzy = input_fuzzify(x_test, features, cnn =1)


# In[291]:


np.save('x_test_fuzzy', x_test_fuzzy)


# In[199]:


# pi_membership_function(r,c_low[0], lambda_low[0])


# # In[159]:


# pi_membership_function(r, c_medium[0], lambda_medium[0])


# # In[200]:


# pi_membership_function(r, c_high[0], lambda_high[0])


# # In[234]:


# x_train.shape


# # In[195]:


# test = x_train[0:2]


# # In[191]:


# test = test.reshape(-1,28,28)


# # In[196]:


# print(test.shape)
# print(test.astype(int))


# # In[247]:


# low = []
# medium = []
# high = []
# for i in x_train[0:100].T:
#     print(i.shape)
#     low.append(pi_membership_function(i, c_low[0], lambda_low[0]))


# # In[236]:


# low = np.array(low)


# # In[237]:


# plt.imshow(low.T.reshape(-1,28,28)[5], cmap='Greys')


# # In[183]:


# plt.imshow(test.reshape(28,28), cmap = 'Greys')


# # In[201]:


# for i in test.T:
#     print(i)


# # In[232]:


# x_train.T.shape


# # In[233]:


# test.T.shape


# # In[ ]:




