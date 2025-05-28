import numpy as np
import matplotlib.pyplot as plt
x = np.array([50, 60, 70, 80, 90])
y = np.array([150, 180, 210, 240, 270])

w = 0.0
b = 0.0
alpha = 0.0001
epochs = 1000

n = len(x)
for i in range(epochs):
    y_pred = w * x + b
    error = y_pred - y

    dw = (2/n) * np.dot(x, error)
    db = (2 / n) * np.sum(error)

    w -= alpha *dw
    b -= alpha *db

    if i % 100 == 0:
        mse = np.mean(error**2)
        print(f"Epoch {i}: MSE = {mse:.2f}, w = {w:.2f}, b = {b:.2f}")

x_new = 85
y_new = w * x_new + b
print(f"\nPredicted price for {x_new}m² house: £{y_new:.2f}k")

# Step 5: Plot the result
plt.scatter(x, y, color='blue', label='Data')
x_line = np.linspace(40, 100, 100)
y_line = w * x_line + b
plt.plot(x_line, y_line, color='red', label='Regression Line')

plt.xlabel("Size (m²)")
plt.ylabel("Price (£1000s)")
plt.title("Linear Regression: House Price Prediction")
plt.legend()
plt.grid(True)
plt.show()