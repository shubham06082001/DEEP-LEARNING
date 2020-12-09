#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 20:59:17 2020

@author: shubhamkumar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets

dataset= pd.read_csv('Credit_Card_Applications.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# feature scaling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))

X = sc.fit_transform(X)

# training the som

from minisom import MiniSom 

som = MiniSom(x = 10, y = 10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)

som.train_random(X, num_iteration=100)

# visualising the results

from pylab import bone, pcolor, colorbar, plot, show

bone()

pcolor(som.distance_map().T)

colorbar()

# o = circle, s = square
markers = ['o', 's']

# r = red, g = green
colors = ['r', 'g']

# red = no approval, green = got approval
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize=10,
         markeredgewidth = 2
         )
show()   
    
# Finding the frauds

# mappings = som.win_map(X)
# frauds = np.concatenate((mappings[(5,3)], mappings[(8,3)]), axis = 0)
# frauds = sc.inverse_transform(frauds)

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)

# part 2 - going from unsupervised to supervised deep learning

# creating the matrix of features
customers = dataset.iloc[:, 1:].values

# creating the dependent variable

is_fraud = np.zeros(len(dataset))

for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1
        

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(customers)

y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)

y_pred = y_pred[y_pred[:, 1].argsort()]
















