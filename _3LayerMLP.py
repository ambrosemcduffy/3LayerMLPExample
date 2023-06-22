import numpy as np


def initializeParameters(nx, nh, ny):
    w1 = np.random.randn(nh, nx) * 0.01
    w2 = np.random.randn(ny, nh) * 0.01
    
    b1 = np.zeros((nh, 1))
    b2 = np.zeros((ny, 1))
    return [w1, w2, b1, b2]


def crossEntropy(m, y, yhat):
    return (-1.0/m) * np.sum(y*np.log(yhat) + ((1-y) * np.log(1-yhat)))


def forward(weights, X):
    w1, w2, b1, b2 = weights 
    z1 = np.dot(w1, X) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    return [z1, a1, z2]


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def optimization(learningRate, weights, grads):
    w1, w2, b1, b2 = weights
    dw2, dw1, db1, db2 = grads
    
    w2 = w2 - learningRate * dw2
    w1 = w1 - learningRate * dw1
    b1 = b1 - learningRate * db1
    b2 = b2 - learningRate * db2
    return [w1, w2, b1, b2]


def backprop(a2, a1, X, w2, y, m):
    dz2 = a2-y
    dw2 = (1.0/m) * np.dot(dz2, a1.T)
    db2 = (1.0/m)  * np.sum(dz2, axis=1, keepdims=True)
    
    dz1 = np.multiply(np.dot(w2.T, dz2), 1-np.power(a1, 2))
    dw1 = (1.0/m) * np.dot(dz1, X.T)
    db1 = (1.0/m) * np.sum(dz1, axis=1, keepdims=True)    
    return [dw2, dw1, db1, db2]


def sigmoidPrime(h):
    return sigmoid(h) * (1 - sigmoid(h))

def simplifiedPrime(h):
    return h / ((1 + np.exp(-h))**2)

def ambrosePrime(h):
    return (1+np.exp(-h)) * (0) - (1) * (np.exp(-h)) * (-h)/((1+np.exp(-h))**2)

def toOneHot(labels, num_classes):
    one_hot = np.zeros((labels.shape[1], num_classes))
    for i in range(labels.shape[1]):
        label = labels[:, i]
        if np.max(label) >= num_classes:
            raise ValueError("Label value exceeds number of classes")
        one_hot[i][label] = 1
    return one_hot

from dataEvaluation import getAccuracy

import dataProcessor
xTrain, yTrain, xTest, yTest = dataProcessor.getDataset(flattenImages=True)
yTrain = toOneHot(yTrain, 5).T
yTest = toOneHot(yTest, 5).T


stepSize = 0.001

weights = initializeParameters(xTrain.shape[0], 80, 5)

for i in range(5000):
    z1, a1, z2 = forward(weights, xTrain)
    yhat = sigmoid(z2)
    cost = crossEntropy(xTrain.shape[1], yTrain, yhat)
    grads = backprop(yhat, a1, xTrain, weights[1], yTrain, xTrain.shape[1])
    weights = optimization(stepSize, weights, grads)
    if i % 100 == 0:
        z1_test, a1_test, z2_test = forward(weights, xTest)
        yhat_test = sigmoid(z2_test)
        trainAcc = getAccuracy(yhat, yTrain)
        testAcc = getAccuracy(yhat_test, yTest)
        print("epoch:{} -- Cost {} -- train-accuracy {} -- test-accuracy {}".format(i, cost, trainAcc, testAcc))