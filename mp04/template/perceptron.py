# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020



import numpy as np
import math

def trainPerceptron(train_set, train_labels,  max_iter):
    #Write code for Mp4
    W = np.zeros((1, len(train_set[0])))
    b = 0
    alpha = 0.01
    for epoch in range(max_iter):
        for x, label in zip(train_set, train_labels):
            y_hat = np.dot(W, x) + b
            if y_hat > 0  and label == 0:
                W -= x * alpha
                b -= alpha
            elif y_hat <= 0 and label == 1:
                W += x * alpha
                b += alpha
    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, max_iter):
    #Write code for Mp4
    W, b = trainPerceptron(train_set, train_labels, max_iter)
    predictions = []
    for x in dev_set:
        y_hat = np.dot(W, x) + b
        predictions.append(1 if y_hat > 0 else 0)
    return predictions



