#!/usr/bin/env python3
"""
defines DeepNeuralNetwork class that defines
a deep neural network performing multiclass classification
"""

import numpy as np


class DeepNeuralNetwork:
    """
    class that represents a deep neural network
    performing multiclass classification
    """

    def __init__(self, nx, layers):
        """
        class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev = nx
        for i, layer in enumerate(layers, 1):
            if type(layer) is not int or layer <= 0:
                raise TypeError("layers must be a list of positive integers")

            self.__weights["W{}".format(i)] = (
                np.random.randn(layer, prev) * np.sqrt(2 / prev)
            )
            self.__weights["b{}".format(i)] = np.zeros((layer, 1))
            prev = layer

    @property
    def L(self):
        """getter for __L"""
        return self.__L

    @property
    def cache(self):
        """getter for __cache"""
        return self.__cache

    @property
    def weights(self):
        """getter for __weights"""
        return self.__weights

    def forward_prop(self, X):
        """
        calculates the forward propagation of the network
        """
        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            W = self.__weights["W{}".format(i)]
            b = self.__weights["b{}".format(i)]
            Z = np.matmul(W, self.__cache["A{}".format(i - 1)]) + b

            if i == self.__L:
                t = np.exp(Z)
                A = t / np.sum(t, axis=0, keepdims=True)
            else:
                A = 1 / (1 + np.exp(-Z))

            self.__cache["A{}".format(i)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
        calculates the cost of the model using multiclass cross-entropy
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        evaluates the neural network's predictions
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        prediction = np.zeros_like(A)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        calculates one pass of gradient descent on the network
        """
        m = Y.shape[1]
        weights_copy = self.__weights.copy()

        for i in range(self.__L, 0, -1):
            A_curr = cache["A{}".format(i)]
            A_prev = cache["A{}".format(i - 1)]

            if i == self.__L:
                dZ = A_curr - Y
            else:
                dZ = np.matmul(weights_copy["W{}".format(i + 1)].T, dZ) * (
                    A_curr * (1 - A_curr)
                )

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            self.__weights["W{}".format(i)] = (
                self.__weights["W{}".format(i)] - alpha * dW
            )
            self.__weights["b{}".format(i)] = (
                self.__weights["b{}".format(i)] - alpha * db
            )

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        trains the neural network
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        if graph:
            import matplotlib.pyplot as plt
            points = []
            x_points = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)

            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)

                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))

                if graph:
                    x_points.append(i)
                    points.append(cost)

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(x_points, points)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        saves the instance object to a file in pickle format
        """
        import pickle

        if type(filename) is not str:
            return

        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        loads a pickled DeepNeuralNetwork object from a file
        """
        import pickle

        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
