import numpy as np

# XOR input and output
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [1], [1], [0]])


def sigmoid(x):
    return 1 / (1 + np.exp(-x)) #This squashes values into the range [0,1]


def sigmoid_derivative(x): #used in backpropagation to adjust weights
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x) #replaces negative values with 0. Used in the hidden layer to make it nonlinear.

def relu_derivative(x):
    return (x > 0).astype(float) # This returns 1 for positive x and 0 for negative x


np.random.seed(42)
input_size = 2
hidden_size = 2
output_size = 1

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))


lr = 0.1
epochs = 10000


for epoch in range(epochs):
  
    z1 = np.dot(X, W1) + b1 #dot product of inputs and weights plus bias
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

 
    loss = np.mean((a2 - y) ** 2)

   
    d_a2 = 2 * (a2 - y) / y.size #tells how much the output error contributes to loss
    d_z2 = d_a2 * sigmoid_derivative(a2)
    d_W2 = np.dot(a1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = np.dot(d_z2, W2.T) #Moves error back to the hidden layer
    d_z1 = d_a1 * relu_derivative(z1)
    d_W1 = np.dot(X.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    
    W2 -= lr * d_W2
    b2 -= lr * d_b2
    W1 -= lr * d_W1
    b1 -= lr * d_b1 #recalculating weights using gradient descent

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nFinal Predictions:")
predictions = sigmoid(np.dot(relu(np.dot(X, W1) + b1), W2) + b2)
print(np.round(predictions, 3))
