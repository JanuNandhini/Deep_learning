import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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


X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = Perceptron(
    learning_rate=0.01,
    epochs=100
)

model.fit(X_train, y_train)


predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy :", accuracy)

print("\nActual Values")
print(y_test)

print("\nPredicted Values")
print(predictions)
