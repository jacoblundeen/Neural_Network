import pandas as pd
import numpy as np
from statistics import mean


# The Neural Network class
class Neural_Network:
    # Initiation function. Reads in the data set, target variable, prediction type, number of hidden nodes for each
    # hidden layer, and the alpha value. We initialize the hidden layers and the output layers depend on the prediction
    # type. If it is classification, the output layer depends on the number of classes. If it is regression, it's a
    # single value.
    def __init__(self, data, target_variable, prediction_type, num_nodes, alpha):
        self.X = data.drop(target_variable, axis=1).values
        self.Y = data[target_variable].values
        self.m, self.n = self.X.shape
        self.X = self.X.T
        self.num_classes = len(np.unique(self.Y))
        self.W1 = np.random.rand(num_nodes[0], self.n) * np.sqrt(1 / self.n)
        self.b1 = np.random.rand(num_nodes[0], 1)
        self.W2 = np.random.rand(num_nodes[1], num_nodes[0]) * np.sqrt(1 / num_nodes[0])
        self.b2 = np.random.rand(num_nodes[1], 1)
        if prediction_type == 'classification':
            self.W3 = np.random.rand(self.num_classes, num_nodes[1]) * np.sqrt(1 / num_nodes[1])
            self.b3 = np.random.rand(self.num_classes, 1)
        else:
            self.W3 = np.random.rand(1, num_nodes[1])
            self.b3 = np.random.rand(1, 1)
        self.pred_type = prediction_type
        self.alpha = alpha

    # Calculate sigmoid
    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(z))

    # Calculate sigmoid derivative
    def sigmoid_deriv(self, z):
        return z * (1 - z)

    # Calculate softmax
    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    # Forward propagation function. Both hidden layers use the sigmoid activation function. For classification, we use
    # the softmax function to determine the probabilities of each class. For regression, we output the linear
    # combination of the second layer.
    def forward_prop(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.sigmoid(Z2)
        Z3 = self.W3.dot(A2) + self.b3
        if self.pred_type == 'classification':
            A3 = self.softmax(Z3)
        else:
            A3 = Z3
        return Z1, A1, Z2, A2, Z3, A3

    # Helper function to one hot encode the target variable for classification
    def one_hot(self):
        one_hot_Y = pd.get_dummies(self.Y).to_numpy()
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def mse(self, pred):
        return np.mean((pred - self.Y) ** 2)

    def mse_deriv(self, pred):
        return 2 / self.m * (pred - self.Y)

    def cross_entropy_loss(self, A3):
        return np.sum((-1 * self.one_hot()) * np.log(A3))

    # Backpropagation function. Calculates the loss function for either prediction type and then propagates the error
    # back through the network.
    def backward_prop(self, Z1, A1, Z2, A2, Z3, A3):
        if self.pred_type == 'classification':
            one_hot_Y = self.one_hot()
            dZ3 = 2 * (A3 - one_hot_Y)
        else:
            dZ3 = self.mse_deriv(A3)
        dW3 = 1 / self.m * dZ3.dot(A2.T)
        db3 = 1 / self.m * np.sum(dZ3, axis=1)
        dZ2 = self.W3.T.dot(dZ3) * self.sigmoid_deriv(Z2)
        dW2 = 1 / self.m * dZ2.dot(A1.T)
        db2 = 1 / self.m * np.sum(dZ2, axis=1)
        dZ1 = self.W2.T.dot(dZ2) * self.sigmoid_deriv(Z1)
        dW1 = 1 / self.m * dZ1.dot(self.X.T)
        db1 = 1 / self.m * np.sum(dZ1, axis=1)
        return dW1, db1, dW2, db2, dW3, db3

    # Function to update the weights and bias of the network
    def update_params(self, dW1, db1, dW2, db2, dW3, db3):
        self.W1 -= self.alpha * dW1
        self.b1 -= self.alpha * np.expand_dims(db1, axis=1)
        self.W2 -= self.alpha * dW2
        self.b2 -= self.alpha * np.expand_dims(db2, axis=1)
        self.W3 -= self.alpha * dW3
        self.b3 -= self.alpha * np.expand_dims(db3, axis=1)

    # Function to return the predictions for either prediction type
    def get_predictions(self, A3):
        if self.pred_type == 'classification':
            return np.argmax(A3, 0)
        else:
            return np.mean(A3, axis=0)

    # After the network is trained, make predictions on the test set
    def make_predictions(self, X):
        _, _, _, _, _, A3 = self.forward_prop(X)
        predictions = self.get_predictions(A3)
        return predictions

    # Calculate the accuracy for classification
    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size

    # Gradient Descent function to move back and forth through the neural network
    def gradient_descent(self, iterations):
        loss_list = []
        iter_list = []
        loss_df = pd.DataFrame(columns=['Iteration', 'Loss'])
        for i in range(iterations):
            Z1, A1, Z2, A2, Z3, A3 = self.forward_prop(self.X)
            dW1, db1, dW2, db2, dW3, db3 = self.backward_prop(Z1, A1, Z2, A2, Z3, A3)
            self.update_params(dW1, db1, dW2, db2, dW3, db3)
            # if i % 25 == 0:
            #     print("Iteration: ", i)
            #     predictions = self.get_predictions(A3)
            #     if self.pred_type == 'classification':
            #         loss = self.cross_entropy_loss(A3)
            #         print("Cross Entropy loss: ", round(loss, 3))
            #         # loss_list.append(loss)
            #         # iter_list.append(i)
            #         # temp_df = pd.DataFrame(data={'Iteration': i, 'Loss': loss}, index=[0])
            #         # loss_df = pd.concat([loss_df, temp_df], ignore_index=True)
            #     else:
            #         print(self.mse(predictions))
        # loss_df = pd.DataFrame(list(zip(loss_list, iter_list)), columns=['Loss', 'Iterations'])
        # loss_df.to_csv('cancer_loss.csv')
        # return self.W1, self.b1, self.W2, self.b2
