import pandas as pd
import numpy as np
from copy import deepcopy


class Auto_NN:
    def __init__(self, data, target_variable, prediction_type, num_nodes, alpha):
        self.X = data.drop(target_variable, axis=1).values
        self.Y = data[target_variable].values
        # if num_nodes is None:
        self.m, self.n = self.X.shape
        self.num_node1 = num_nodes[0]
        self.num_node2 = num_nodes[1]
        # else:
        #     # self.m = num_nodes
        #     self.m = self.X.shape[0]
        #     self.n = self.X.shape[1]
        self.X = self.X.T
        self.num_classes = len(np.unique(self.Y))
        self.W1 = np.random.rand(num_nodes[0], self.n)  # Encoder Layer
        self.b1 = np.random.rand(num_nodes[0], 1)
        self.W2 = np.random.rand(self.n, num_nodes[0])  # Decoder Layer
        self.b2 = np.random.rand(self.n, 1)
        if prediction_type == 'classification':
            self.W3 = np.random.rand(self.num_classes, self.n)  # Prediction Layer (classification)
            self.b3 = np.random.rand(self.num_classes, 1)
        else:
            self.W3 = np.random.rand(1, self.n)  # Prediction Layer (regression)
            self.b3 = np.random.rand(1, 1)
        self.pred_type = prediction_type
        self.alpha = alpha

    # Calculate sigmoid
    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(z))

    def sigmoid_deriv(self, z):
        return z * (1 - z)

    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    def mse_deriv(self, pred):
        return 2 / self.m * (pred - self.Y)

    def forward_prop(self, X, encoder):
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        if encoder:
            A2 = Z2
            return Z1, A1, Z2, A2
        else:
            if self.pred_type == 'classification':
                A2 = self.softmax(Z2)
            else:
                A2 = Z2
            return Z1, A1, Z2, A2

    def one_hot(self):
        one_hot_Y = pd.get_dummies(self.Y).to_numpy()
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def backward_prop(self, Z1, A1, Z2, A2):
        if self.pred_type == 'classification':
            dZ2 = A2
        else:
            dZ2 = self.mse_deriv(A2)
        dW2 = 1 / self.m * dZ2.dot(A1.T)
        db2 = 1 / self.m * np.sum(dZ2, axis=1)
        dZ1 = self.W2.T.dot(dZ2) * self.sigmoid_deriv(Z1)
        dW1 = 1 / self.m * dZ1.dot(self.X.T)
        db1 = 1 / self.m * np.sum(dZ1, axis=1)
        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2):
        self.W1 -= self.alpha * dW1
        self.b1 -= self.alpha * np.expand_dims(db1, axis=1)
        self.W2 -= self.alpha * dW2
        self.b2 -= self.alpha * np.expand_dims(db2, axis=1)

    def get_predictions(self, A2):
        if self.pred_type == 'classification':
            return np.argmax(A2, 0)
        else:
            return np.mean(A2, axis=0)

    def make_predictions(self, X):
        _, _, _, A2 = self.forward_prop(X, False)
        predictions = self.get_predictions(A2)
        return predictions

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size

    def init(self):
        self.W1 = np.random.rand(self.num_node1, self.n)  # Decoder Layer
        self.b1 = np.random.rand(self.num_node1, 1)
        if self.pred_type == 'classification':
            self.W2 = np.random.rand(self.num_classes, self.num_node1)  # Prediction Layer (classification)
            self.b2 = np.random.rand(self.num_classes, 1)
        else:
            self.W2 = np.random.rand(1, self.num_node1)  # Prediction Layer (regression)
            self.b2 = np.random.rand(1, 1)

    def gradient_descent(self, iterations):
        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward_prop(self.X, True)
            dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, A2)
            self.update_params(dW1, db1, dW2, db2)
        self.X = A2
        self.init()
        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward_prop(self.X, False)
            dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, A2)
            self.update_params(dW1, db1, dW2, db2)
            # if i % 25 == 0:
            #     print("Iteration: ", i)
            #     predictions = self.get_predictions(A2)
            #     if self.pred_type == 'classification':
            #         loss = self.cross_entropy_loss(A2)
            #         print("Cross Entropy loss: ", round(loss, 3))
            #     else:
            #         print(self.mse(predictions))
