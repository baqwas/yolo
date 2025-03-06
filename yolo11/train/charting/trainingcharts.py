#!/usr/bin/env python3
"""
    epochvsaccuracy.py
    Illustrate epoch versus versus accuracy
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
        epochvsaccuracy.py

The plot helps visualize the relationship between training set size, training error, and test error,
which is crucial for understanding model performance and identifying potential issues
like overfitting or underfitting.

Overfitting: The training score increases significantly, while the test score plateaus or even decreases.
Underfitting: Both training and test scores remain low and plateau early.
Good Fit: Both training and test scores improve and converge to a high value as the training set size increases.

"""
import matplotlib.pyplot as plt
import numpy as np

def generate_data(num_samples=100, noise=0.2):
  """
  Generates sample data for the plot.

  Args:
    num_samples: Number of data points to generate.
    noise: Amount of noise to add to the data.

  Returns:
    x_values: Array of x-values.
    y_values: Array of y-values.
  """
  x_values = np.linspace(-1, 1, num_samples)
  y_values = np.sin(5 * x_values) + noise * np.random.randn(num_samples)
  return x_values, y_values

def plot_learning_curves(train_sizes, train_scores, test_scores, title):
  """
  Plots learning curves for training and validation data.

  Args:
    train_sizes: Array of training set sizes.
    train_scores: Array of training set scores.
    test_scores: Array of test set scores.
  """
  plt.figure(figsize=(8, 6))
  plt.plot(train_sizes, train_scores, label='Training score')
  plt.plot(train_sizes, test_scores, label='Test score')
  plt.ylabel("Score")
  plt.xlabel("Traing/Test")
  plt.title(title)
  plt.legend(loc="best")
  plt.grid(True)
  plt.ylim(0, 1)
  plt.xlim(0, 100)
  plt.show()

# Generate sample data
x, y = generate_data()

# Create a placeholder for learning curves
train_sizes, train_scores, test_scores = [], [], []
# Simulate learning curves for underfitting
for i in range(10, len(x), 10):
  train_sizes.append(i)
  # Simulate underfitting: Both scores remain low and plateau
  train_scores.append(0.5 * (i / len(x)))
  test_scores.append(0.4 * (i / len(x)))

# Plot the learning curves
plot_learning_curves(train_sizes, train_scores, test_scores,
                     "Underfitting")

# Create a placeholder for learning curves
train_sizes, train_scores, test_scores = [], [], []

# Simulate learning curves for overfitting
for i in range(10, len(x), 10):
  train_sizes.append(i)
  # Simulate overfitting: Training score increases rapidly, test score plateaus or decreases
  train_scores.append(i / len(x) + 0.1 * np.sin(i/20))
  test_scores.append(i / len(x) - 0.1 + np.random.rand() * 0.05)

# Plot the learning curves
plot_learning_curves(train_sizes, train_scores, test_scores,
                     "Overfitting")

# Create a placeholder for learning curves
train_sizes, train_scores, test_scores = [], [], []

# Simulate learning curves for good fit
for i in range(10, len(x), 10):
  train_sizes.append(i)
  # Simulate good fit: Both scores increase and converge to a high value
  train_scores.append(0.95 - 0.05 * np.exp(-i / 100))
  test_scores.append(0.9 - 0.05 * np.exp(-i / 100))

# Plot the learning curves
plot_learning_curves(train_sizes, train_scores, test_scores,
                     "Reasonable Good Fit")

plt.show()
