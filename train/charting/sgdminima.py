"""
    sgdminima.py
    Visualize local and global minima in Stochastic Gradident Descent
    2024-12-11 0.1 armw Initial DRAFT

Copyright (C) 2024 ParkCircus Productions; All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

    Usage:
        sgdminima.py

This script illustrates the challenge with SGD arriving at a global minimum.

This visualization helps in understanding how momentum tweaking can possibly help in reaching global minimum.
"""
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """
    A simple function with multiple local minima.
    """
    return np.sin(3 * x) + x**2 - 0.7 * x

def df(x):
    """
    Derivative of the function f(x).
    """
    return 3 * np.cos(3 * x) + 2 * x - 0.7

def stochastic_gradient_descent(x0, learning_rate, epochs):
    """
    Performs stochastic gradient descent.

    Args:
        x0: Initial guess for the minimum.
        learning_rate: Learning rate for the descent.
        epochs: Number of iterations.

    Returns:
        A tuple containing the final x value and a list of x values
        traversed during the descent.
    """
    x = x0
    x_history = [x0]
    for _ in range(epochs):
        x -= learning_rate * df(x)  # Stochastic update (using only the current x)
        x_history.append(x)
    return x, x_history

# Parameters
learning_rate = 0.01
epochs = 100

# Initial guesses for different minima
initial_guesses = [-2, 0, 2]

# Perform SGD for each initial guess
results = []
for x0 in initial_guesses:
    final_x, x_history = stochastic_gradient_descent(x0, learning_rate, epochs)
    results.append((final_x, x_history))

# Plot the function and the paths taken by SGD
x_values = np.linspace(-2, 2, 100)
y_values = f(x_values)

plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label="Function")

for i, (final_x, x_history) in enumerate(results):
    plt.plot(x_history, f(np.array(x_history)), marker='o', label=f"SGD from x0={initial_guesses[i]}")

plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.title("Stochastic Gradient Descent: Local vs. Global Minima")
plt.grid(True)
plt.show()