import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



train_data = pd.read_csv("ExampleTrainDataset.csv")
test_data = pd.read_csv("ExampleTestDataset.csv")

# Prepare the training data
X_train = train_data[['x1', 'x2', 'x3']].values
y_train = train_data['Y'].values.reshape(-1, 1)


#initializing weights and biases
X_cols = X_train.shape[1]
hidden_neurons = 4
output_neurons = 1
weights_input_hidden = np.random.uniform(-0.1, 0.1, (X_cols, hidden_neurons))
bias_hidden = np.zeros((1, hidden_neurons))
weights_hidden_output = np.random.uniform(-0.1, 0.1, (hidden_neurons, output_neurons))
bias_output = np.zeros((1, output_neurons))


learning_rate = 0.01 #Use sigmoid activation function for all perceptron. Learning rate = 0.01
iterations = 50 #Perform 50 iterations of learning.

losses = []

#Use sigmoid activation function for all perceptron. Learning rate = 0.01
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# training part 2.a
for i in range(iterations):
    # Forward pass part
    hidden_input = np.dot(X_train, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    # Use squared error as the loss function
    errors = final_output - y_train
    loss = np.mean(errors ** 2)
    losses.append(loss)

    # Backpropagation part
    output_error = errors * sigmoid_derivative(final_input)
    hidden_error = np.dot(output_error, weights_hidden_output.T) * sigmoid_derivative(hidden_input)

    # Gradients for weights and biases
    grad_weights_hidden_output = np.dot(hidden_output.T, output_error)
    grad_bias_output = np.sum(output_error, axis=0, keepdims=True)
    grad_weights_input_hidden = np.dot(X_train.T, hidden_error)
    grad_bias_hidden = np.sum(hidden_error, axis=0, keepdims=True)

    # Updating weights
    weights_hidden_output -= learning_rate * grad_weights_hidden_output
    bias_output -= learning_rate * grad_bias_output
    weights_input_hidden -= learning_rate * grad_weights_input_hidden
    bias_hidden -= learning_rate * grad_bias_hidden


# Plot a graph of training Loss with respect to iterations. Include this graph in your report with the final weight vectors.
plt.figure(figsize=(8, 5))
plt.plot(range(1, iterations + 1), losses, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss vs Iterations (MLP)')
plt.grid()
plt.show()


print("Input-Hidden Weights:", weights_input_hidden)
print("Hidden-Output Weights:", weights_hidden_output)


# b) (5 points)  Using the trained MLP, perform predictions on “ExampleTestDataset.csv” and report the test loss
X_test = test_data[['x1', 'x2', 'x3']].values
y_test = test_data['Y'].values.reshape(-1, 1)

hidden_input_test = np.dot(X_test, weights_input_hidden) + bias_hidden
hidden_output_test = sigmoid(hidden_input_test)
final_input_test = np.dot(hidden_output_test, weights_hidden_output) + bias_output
final_output_test = sigmoid(final_input_test)

test_loss = np.mean((final_output_test - y_test) ** 2)
print("Test Loss:", test_loss)

#Ashwabh Bhatnagar