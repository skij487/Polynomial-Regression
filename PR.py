'''
Polynomial Regression
'''
import numpy as np
import math

def CrossValidation(X, Y, K, M):
    N = Y.shape[0]
    J = []
    W = math.ceil(N/M)
    for i in range(M):
        tstart = W * i
        tend = min(W * (i + 1), N)
        trainidx = list(range(0, tstart)) + list(range(tend, N))
        t_cost, t_wt, t_prediction = PRLearn(X, Y[trainidx], K)
        testdata = np.mean(Y[tstart:tend], axis=0)
        cost, prediction = PRPredict(X, testdata, K, t_wt)
        J.append(cost)
    return np.mean(J)

def Transform(X, K):
    m = X.shape[0]
    X_t = np.ones((m, 1))
    for j in range(K+1):
        if j != 0:
            X_t = np.append(X_t, np.power(X, j).reshape(-1,1), axis=1)
    return X_t

def Normalize(X):
    X_n = np.ones_like(X)
    X_n[:, 1:] = (X[: , 1:] - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)
    return X_n

def PRPredict(X, Y, K, wt):
    X_t = Transform(X, K)
    X_n = Normalize(X_t)    
    prediction = np.dot(X_n, wt)
    error = prediction - Y
    cost = np.mean(np.power(error, 2)) / 2
    return cost, prediction

def PRLearn(X, Y, K):
    # X = elevangle
    # Y = depth
    # K = degree
    lr = 0.01
    max_iter = 100000
    m = X.shape[0]
    X_t = Transform(X, K)
    X_n = Normalize(X_t)
    wt = np.zeros(K+1)
    prev_wt = np.ones_like(wt)
    conv = False
    count = 0
    while not conv and count < max_iter:
        prev_wt = wt
        h = np.dot(X_n, wt)
        error = h - Y
        if np.array(Y).ndim != 1:
            error = np.mean(error, axis=0)
        wt = wt - lr * (1/m) * np.dot(X_n.T, error)
        conv = all(np.isclose(prev_wt, wt))
        count += 1
    cost, prediction = PRPredict(X, Y, K, wt)
    return cost, wt, prediction
