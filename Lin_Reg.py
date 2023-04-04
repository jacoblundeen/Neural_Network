import numpy as np


class LinearRegression:
    def __init__(self, eta, iterations):
        self.eta = eta
        self.iterations = iterations

    # Function for model training
    def fit(self, data, target_variable=''):
        self.X = data.drop(target_variable, axis=1).values
        self.Y = data[target_variable].values
        self.m, self.n = self.X.shape

        # weight initialization
        self.W = np.zeros(self.n)
        self.b = 0

        # gradient descent learning

        for i in range(self.iterations):
            self.update_weights()
        return self

    # Helper function to update weights in gradient descent

    def update_weights(self):
        Y_pred = self.predict(self.X)

        # calculate gradients
        dW = - (2 * (self.X.T).dot(self.Y - Y_pred)) / self.m
        db = - 2 * np.sum(self.Y - Y_pred) / self.m

        # update weights
        self.W = self.W - self.eta * dW
        self.b = self.b - self.eta * db
        return self

    # Hypothetical function  h( x )
    def predict(self, X):
        return X.dot(self.W) + self.b