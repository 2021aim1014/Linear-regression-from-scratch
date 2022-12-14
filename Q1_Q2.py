# -*- coding: utf-8 -*-
"""2021AIM1014-assignment1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yVf6otePzAAEW1U2ETbdPXjB1JRKpMNk
"""

import random

def create_data_set(N):
  data = []
  for i in range(N):
    y = random.choice([0, 1])
    if y == 1:
      # If y = 1 then x1 ∼ U(2, 7) and x2 ∼ U(4, 6). where U(a, b) represent the uniform distribution between a, b.
      x1 = random.uniform(2, 7)
      x2 = random.uniform(4, 6)
      data.append([x1, x2, y])
    else:
      # If y = 0 then x1 ∼ U(0, 2) ∪ U(7, 9) and x2 ∼ U(1, 3) ∪ U(6, 8).
      temp1 = random.uniform(0, 2)
      temp2 = random.uniform(7, 9) 
      x1 = random.choice([temp1, temp2])
      temp1 = random.uniform(1, 3)
      temp2 = random.uniform(6, 8) 
      x2 = random.choice([temp1, temp2])
      data.append([x1, x2, y])

  return data

data_set = create_data_set(30)
# print(data_set)

import matplotlib.pyplot as plt
# import numpy as np

def filter_data_set(data_set):
  d1 = []
  d2 = []
  d3 = []
  d4 = []
  for data in data_set:
    if(data[2] == 1):
      d1.append(data[0])
      d2.append(data[1])
    else:
      d3.append(data[0])
      d4.append(data[1])
  return d1, d2, d3, d4

def plot_data_set(data_set):
  d1, d2, d3, d4 = filter_data_set(data_set)
  plt.title('Dataset')
  plt.xlabel('x1')
  plt.ylabel('x2')
  # plt.axis([0, 10, 0, 10])
  plt.plot(d1, d2, 'rs')
  plt.plot(d3, d4, 'g^')
  plt.show()

plot_data_set(data_set)

import matplotlib.patches as mpatches

def circle_hypothesis():
  d1, d2, d3, d4 = filter_data_set(data_set)
  
  # most specific hypothesis
  s_x1_min = min(d1)
  s_x1_max = max(d1)
  s_x2_min = min(d2)
  s_x2_max = max(d2)

  #most general hypothesis
  g_x1_min = 0
  g_x1_max = 10
  g_x2_min = 0
  g_x2_max = 10
  for i in range(len(d3)):
    if d3[i] <= 2:
      g_x1_min = max(g_x1_min, d3[i])
    else:
      g_x1_max = min(g_x1_max, d3[i])

    if d4[i] <= 3:
      g_x2_min = max(g_x2_min, d4[i])
    else:
      g_x2_max = min(g_x2_max, d4[i])

  # print(g_x1_min, g_x1_max, g_x2_min, g_x2_max)

  plt.title('Dataset')
  plt.xlabel('x1')
  plt.ylabel('x2')
  # plt.axis([0, 10, 0, 10])
  plt.plot(d1, d2, 'rs')
  plt.plot(d3, d4, 'g^')
  s_x = (s_x1_min+s_x1_max)/2
  s_y = (s_x2_min+s_x2_max)/2
  s_r = max(s_x1_max - s_x1_min, s_x2_max - s_x2_min) / 2
  circle1 = plt.Circle((s_x, s_y), s_r, fill = False, color='r')
  g_x = (g_x1_min+g_x1_max)/2
  g_y = (g_x2_min+g_x2_max)/2
  g_r = max(g_x1_max - g_x1_min, g_x2_max - g_x2_min) / 2
  circle2 = plt.Circle((g_x, g_y), g_r, fill = False, color='b')
  plt.gca().add_patch(circle1)
  plt.gca().add_patch(circle2)
  plt.show()

circle_hypothesis()

def rectangular_hypothesis():
  d1, d2, d3, d4 = filter_data_set(data_set)
  
  # most specific hypothesis
  s_x1_min = min(d1)
  s_x1_max = max(d1)
  s_x2_min = min(d2)
  s_x2_max = max(d2)

  #most general hypothesis
  g_x1_min = 0
  g_x1_max = 10
  g_x2_min = 0
  g_x2_max = 10
  for i in range(len(d3)):
    if d3[i] <= 2:
      g_x1_min = max(g_x1_min, d3[i])
    else:
      g_x1_max = min(g_x1_max, d3[i])

    if d4[i] <= 3:
      g_x2_min = max(g_x2_min, d4[i])
    else:
      g_x2_max = min(g_x2_max, d4[i])

  # print(g_x1_min, g_x1_max, g_x2_min, g_x2_max)

  plt.title('Dataset')
  plt.xlabel('x1')
  plt.ylabel('x2')
  # plt.axis([0, 10, 0, 10])
  plt.plot(d1, d2, 'rs')
  plt.plot(d3, d4, 'g^')
  rect1=mpatches.Rectangle((s_x1_min,s_x2_min),(s_x1_max-s_x1_min),(s_x2_max-s_x2_min), fill = False,color = "purple")
  rect2=mpatches.Rectangle((g_x1_min,g_x2_min),(g_x1_max-g_x1_min),(g_x2_max-g_x2_min), fill = False,color = "orange")
  plt.gca().add_patch(rect1)
  plt.gca().add_patch(rect2)
  plt.show()

rectangular_hypothesis()

"""Observations:

For most specific hypothesis:
0 false positive

FOr most general hypothesis:
0 false negative
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math

def generate_data_set(N):
  x = np.random.uniform(0, 2 * np.pi, size=(N, 1))
  y = np.cos((2 * np.pi * x) + (x / 2 * np.pi) + np.random.normal(0, 0.004))
  data_set = []
  for i in range(len(x)):
    data_set.append((x[i][0], y[i][0]))
  data_set = np.array(data_set)
  return data_set

def calculate_optimal_w(data_set, M):
  X = []
  Y = []
  for x, y in data_set:
    t = [x**i for i in range(M+1)]
    X.append(t)
    Y.append(y)
  X = np.array(X)
  Y = np.array(Y)
  X_T = np.transpose(X)

  t1 = np.dot(X_T, X)
  t2 = np.linalg.inv(t1)
  t3 = np.dot(t2, X_T)
  t4 = np.dot(t3, Y)
  return t4, X

def make_predictions(data_set, M):
  w_star, X = calculate_optimal_w(data_set, M)
  y_predict = np.dot(X, w_star)
  return y_predict

def plot_graph(M, x, y, y_predict): 
  title = 'M = ' + str(M)
  plt.title(title)
  plt.xlabel('x: input')
  plt.ylabel('y: output')
  plt.plot(x, y, 'r^', x, y_predict, 'bo-')
  plt.show()

print(data_set)

N = 20
data_set = generate_data_set(N)
x = data_set[:,0]
y = data_set[:,1]
M = [1, 2, 3, 5, 7, 10]
for each in M:
  y_predict = make_predictions(data_set, each)
  plot_graph(each, x, y, y_predict)

N = 1000
data_set = generate_data_set(N)
x = data_set[:,0]
y = data_set[:,1]
M = [1, 2, 3, 5, 7, 10]
for each in M:
  y_predict = make_predictions(data_set, each)
  plot_graph(each, x, y, y_predict)

"""Observations:

For less number of data points:
1. For less value of M, the predictions are not accurate for training data but works better for test data.
2. As M values increses, it tries to map to all the training points. But may perform poorly on tst data.

For large number of data points:
1. As the size of training set increses, as the M value is incresed, the predicts becomes much accurate
"""