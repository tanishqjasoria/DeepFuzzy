
import numpy as np
import time
import math

"""
This module will consists of functions that could be used for
fuzzification of the input and output of a Neural Network.
"""

def pi_membership_function(r,c,radius):
  """
  This is the default membership function that can be used to
  fuzzify the input dataset for a neural network.
  As per our use case this function here is defined for only one dimension.
  TODO: Include more memebership function or testing purposes
  :param r: the patter values
  :param c: central point
  :param radius: radius of the pi function
  :return: the membership value of the given function
  """

  eucledian_norm = abs(r-c)

  if radius == 0:
    # Check whether this must return zeros or ones
    # Opinion: I think it should be ones
    return np.zeros(eucledian_norm.shape)
  else:
    norm_by_radius = eucledian_norm/radius
    for i in range(len(r)):

      if eucledian_norm[i] <= radius and eucledian_norm[i] >= radius/2:
        eucledian_norm[i] = (2*((1-norm_by_radius[i])**2))

      elif eucledian_norm[i] < radius/2 and eucledian_norm[i] >= 0:
        eucledian_norm[i] = (1 - 2*((norm_by_radius[i])**2))

      else:
        eucledian_norm[i] = 0

  return eucledian_norm

def input_fuzzify(x_train, iid = 1, cnn = 0):
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
  x_train = x_train.T
  #     print(x_train)
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


# def input_fuzzify(x_train):
#     fdenom = 2
# #     F_max = np.array([255]*784)
# #     F_min = np.array([0]*784)
#     F_max = np.ndarray.max(x_train, axis = 0)
#     F_min = np.ndarray.min(x_train, axis = 0)
#     lambda_medium = 0.5*(F_max - F_min)
#     c_medium = F_min + lambda_medium
# #     print(c_medium)
#     lambda_low = (1/fdenom)*(c_medium-F_min)
# #     print(lambda_low)
#     c_low = c_medium - 0.5 * lambda_low
# #     print(c_low)
#     lambda_high = (1/fdenom) * (F_max - c_medium)
# #     print(lambda_high)
#     c_high = c_medium + 0.5 * lambda_high
# #     print(c_high)
#     x_train = x_train.T
# #     print(x_train)
#     x_train_new = []
#     for i in range(len(F_max)):
#         x_train_new.append(pi_membership_function(x_train[i],c_low[i],lambda_low[i]))
#         x_train_new.append(pi_membership_function(x_train[i],c_medium[i],lambda_medium[i]))
#         x_train_new.append(pi_membership_function(x_train[i],c_high[i],lambda_high[i]))
#     return np.array(x_train_new)    



"""Output fuzification functions"""
def output_normalize(x_train, y_train):
  shape  = y_train.shape[1]
  mean = np.zeros((shape,x_train.shape[1]))
  standard_deviation = np.zeros((shape, x_train.shape[1]))
  no_of_belongings = np.zeros(shape)
  for i in range(x_train.shape[0]):
    for j in range(y_train.shape[1]):
      if(y_train[i][j]==1):
        no_of_belongings[j] = no_of_belongings[j]+1
        mean[j]= mean[j]+x_train[i]

  print(mean.shape)
  # print(mean[0][0:100])
  for j in range(y_train.shape[1]):
    mean[j] = mean[j]/no_of_belongings[j]
    # print(no_of_belongings[j])

  # print(mean.shape)
  # print(k)
  # print(mean[0][0:100])
  for i in range(x_train.shape[0]):
    for j in range(y_train.shape[1]):
      if(y_train[i][j]==1):
        k = x_train[i] - mean[j]
        k = np.square(k)
        standard_deviation[j] = standard_deviation[j]+k

  for j in range(y_train.shape[1]):
    standard_deviation[j] = standard_deviation[j]/(no_of_belongings[j] - 1)
    for i in range(len(standard_deviation[j])):
      if standard_deviation[j][i] == 0:
        standard_deviation[j][i] = 1

  standard_deviation = np.sqrt(standard_deviation)
  print(standard_deviation.shape)
  weighted_distance = np.zeros((x_train.shape[0],shape))
  for i in range(x_train.shape[0]):
    for j in range(y_train.shape[1]):  #10
      weighted_distance[i][j] = math.sqrt(np.sum(np.square(np.divide(x_train[i] - mean[j],standard_deviation[j]))))
  return weighted_distance, mean

def output_membership_function(n_class, pattern):
  # Out is an array on n_class dimension, containing membership of a feature vector to each class
  denom_generator = 5
  expo_generator = 2
  out = pattern / denom_generator
  out = np.power(out, expo_generator)
  out = np.reciprocal(1 + out)
  return out

def membership_enhancement(membership):
  if membership < 0.5:
    enhc = 2 * np.power(membership, 2)
  else:
    enhc = 1 - np.power(1-membership, 2)
  return enhc

def output_fuzzify(x_train, y_train):
  # Normalize the output fot class membership
  weighted_distance, _ = output_normalize(x_train, y_train)
  # Get class memberships for each input pattern
  membership = output_membership_function(10, weighted_distance)
  # Enhance the membership values if needed (the membership values are too fuzzified)
  enhanced_membership = (membership)
  return enhanced_membership

"""Fuzzification"""
def fuzzify_dataset(x_train, x_test, y_train, y_test, iid = 0, cnn = 0):
  start = time.time()
  x_train_fuzzy = input_fuzzify(x_train, iid, cnn)
  end = time.time()
  print("Time taken to fuzzify - x_train :", end = ' ')
  print(end-start)
  start = time.time()
  x_test_fuzzy = input_fuzzify(x_test)
  end = time.time()
  print("Time taken to fuzzify - x_test :", end = ' ')
  print(end-start)
  start = time.time()
  y_train_fuzzy = output_fuzzify(x_train, y_train)
  end = time.time()
  print("Time taken to fuzzify - y_train :", end = ' ')
  print(end-start)
  start = time.time()
  y_test_fuzzy = y_test
  end = time.time()
  print("Time taken to fuzzify - y_train :", end = ' ')
  print(end-start)
  return x_train_fuzzy, x_test_fuzzy, y_train_fuzzy, y_test_fuzzy

