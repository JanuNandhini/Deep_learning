import numpy as np
class Perceptron:

    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def activation(self, x):
      if x > 0:
        return 1
      else:
        return 0


    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):

            for i in range(n_samples):

                linear_output = np.dot(X[i], self.weights) + self.bias

                prediction = self.activation(linear_output)

                error = y[i] - prediction

                self.weights += self.learning_rate * error * X[i]

                self.bias += self.learning_rate * error

    def predict(self, X):

        predictions = []

        for x in X:
            linear_output = np.dot(x, self.weights) + self.bias
            predictions.append(self.activation(linear_output))

        return np.array(predictions)

X_and = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y_and = np.array([0,0,0,1])

model_and = Perceptron(learning_rate=0.1, epochs=20)
model_and.fit(X_and, y_and)

print("AND Gate Prediction")
print(model_and.predict(X_and))


X_or = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y_or = np.array([0,1,1,1])

model_or = Perceptron(learning_rate=0.1, epochs=20)
model_or.fit(X_or, y_or)

print("\nOR Gate Prediction")
print(model_or.predict(X_or))
