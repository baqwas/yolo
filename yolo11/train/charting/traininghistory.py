#!/usr/bin/env python3
"""
    traininghistory.py
    Visualize the trend in accuracies with increasing epochs
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
        traininghistory.py

This script effectively demonstrates how to visualize the improvement of accuracy
as the number of training epochs increases. You can modify the accuracies list
to simulate different training scenarios, such as:
    Early Stopping: The accuracy plateaus or even slightly decreases after a certain number of epochs.
    Overfitting: The accuracy on the training data continues to increase, but
        the accuracy on a validation set starts to decrease.

This visualization helps in understanding the training process and
identifying the optimal number of epochs for a given model and dataset.
The script clearly illustrates how the model's accuracy improves as the number of training epochs increases.
"""
import matplotlib.pyplot as plt

def plot_training_history(epochs, train_accuracies, test_accuracies):
  """
  Plots the training and testing accuracy over epochs.

  Args:
    epochs: A list of epochs (integers).
    train_accuracies: A list of corresponding training accuracies.
    test_accuracies: A list of corresponding testing accuracies.
  """
  plt.figure(figsize=(8, 6))
  plt.plot(epochs, train_accuracies, label='Training Accuracy', marker='o', linestyle='-')
  plt.plot(epochs, test_accuracies, label='Testing Accuracy', marker='s', linestyle='--')
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.title("Training and Testing Accuracy vs. Epochs")
  plt.legend()
  plt.grid(True)
  plt.ylim(0, 1)  # Set y-axis limits to 0 and 1
  plt.show()

# Simulated training data
epochs = range(1, 51)  # Example: 50 epochs
train_accuracies = [0.2,   0.3,   0.4,   0.5,   0.6,   0.65,  0.7,   0.75,  0.8,   0.82,
                    0.85,  0.87,  0.88,  0.89,  0.9,   0.91,  0.92,  0.93,  0.94,  0.945,
                    0.95,  0.952, 0.955, 0.958, 0.96,  0.962, 0.964, 0.965, 0.966, 0.967,
                    0.968, 0.969, 0.97,  0.971, 0.972, 0.973, 0.974, 0.975, 0.976, 0.977,
                    0.978, 0.979, 0.98,  0.981, 0.982, 0.983, 0.984, 0.985, 0.985, 0.985]

# Simulate testing accuracy (example with slight overfitting)
test_accuracies = [0.15, 0.25, 0.35, 0.45, 0.55, 0.6,  0.65, 0.7,  0.75, 0.78,
                   0.8,  0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9,
                   0.9,  0.9,  0.9,  0.9,  0.9,  0.9,  0.89, 0.88, 0.87, 0.86,
                   0.85, 0.84, 0.83, 0.82, 0.81, 0.8,  0.79, 0.78, 0.77, 0.76,
                   0.75, 0.74, 0.73, 0.72, 0.71, 0.7,  0.68, 0.66, 0.64, 0.62]

# Plot the training history
plot_training_history(epochs, train_accuracies, test_accuracies)