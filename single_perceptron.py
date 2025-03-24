import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_data = pd.read_csv("ExampleTrainDataset.csv")
test_data = pd.read_csv("ExampleTestDataset.csv")

X_train = train_data[['x1', 'x2', 'x3']].values
y_train = train_data['Y'].values


#initializing weights
X_cols = X_train.shape[1]
weights = np.random.uniform(-0.1, 0.1, X_cols)

learning_rate = 0.01 #Use sigmoid activation function for the perceptron. Learning rate = 0.01
iterations = 50 #Perform 50 iterations of learning

losses = []

#Use sigmoid activation function for the perceptron. Learning rate = 0.01
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# training part 2.a
for i in range(iterations):
    z = np.dot(X_train, weights)
    y_pred = sigmoid(z)

    #Use squared error as the loss function
    errors = y_pred - y_train
    loss = np.mean(errors ** 2)
    losses.append(loss)

    gradient = np.dot(X_train.T, errors * sigmoid_derivative(z))

    # Updating weights
    weights -= learning_rate * gradient

# Plot a graph of training Loss with respect to iterations.
plt.figure(figsize=(8, 5))
plt.plot(range(1, iterations + 1), losses, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss vs Iterations')
plt.grid()
plt.show()

print("Weights:", weights)

# 2.b) (5 points) Using the trained perceptron, perform predictions on “ExampleTestDataset.csv” and report the test loss

X_test = test_data[['x1', 'x2', 'x3']].values
y_test = test_data['Y'].values

z_test = np.dot(X_test, weights)
y_test_pred = sigmoid(z_test)

test_loss = np.mean((y_test_pred - y_test) ** 2)
print("Test Loss:", test_loss)

#Ashwabh Bhatnagar