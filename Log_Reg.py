import pandas as pd
import numpy as np


class Log_Reg:
    def __init__(self, data, target_variable='class'):
        self.X = data.drop(target_variable, axis=1)
        self.Y = data[target_variable]

    def fit(self, max_iter=1000, eta=0.1, mu=0.1):
        self.loss_steps, self.W = self.gradient_descent(self.X, self.Y, max_iter, eta, mu)


    def predict(self, H):
        Z = -H.dot(self.W)
        P = self.softmax(Z)
        return P.idxmax(axis=1)

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(self, X, y, max_iter, eta, mu):
        Y = pd.get_dummies(y)
        W = np.random.normal(size=(X.shape[1], len(y.unique())))
        step = 0
        step_lst = []
        loss_lst = []
        W_lst = []

        while step < max_iter:
            step += 1
            W -= eta * self.gradient(X, Y, W, mu)
            step_lst.append(step)
            W_lst.append(W)
            loss_lst.append(self.loss(X, Y, W))

        df = pd.DataFrame({'step': step_lst, 'loss': loss_lst})

        return df, W

    def softmax(self, Z):
        P = Z.applymap(lambda x: np.exp(x))
        return P.divide(P.sum(axis=1), axis=0)

    def loss(self, X, y, W):
        Z = -X.dot(W)
        n = X.shape[0]
        loss = 1 / n * (np.trace(-Z.dot(y.T)) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
        return loss

    def gradient(self, X, y, W, mu):
        Z = -X.dot(W)
        P = self.softmax(Z)
        n = X.shape[0]
        gd = 1 / n * X.T.dot(y - P) + 2 * mu * W
        return gd
