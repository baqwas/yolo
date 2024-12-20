#!/usr/bin/env python3
"""
    learningrateaccuracy.py
    Illustrate impact of hyperparameter on model accuracy
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
        learningrateaccuracy.py
"""
import matplotlib.pyplot as plt

def plot_hyperparameter_impact(hyperparameter_values, accuracies, xlabel):
  """
  Plots the impact of a hyperparameter on model accuracy.

  Args:
    hyperparameter_values: A list of values for the hyperparameter.
    accuracies: A list of corresponding accuracies for each hyperparameter value.
  """
  plt.figure(figsize=(8, 6))
  plt.plot(hyperparameter_values, accuracies, marker='o', linestyle='-')
  plt.xlabel(xlabel)
  plt.ylabel("Accuracy")
  plt.title("Impact of Hyperparameter on Accuracy")
  plt.grid(True)
  plt.show()

                                        # Impact of Learning Rate on Accuracy
learning_rates = [0., 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
accuracies = [0., 0.75, 0.85, 0.9, 0.82, 0.7, 0.5]  # Simulated accuracies

                                        # Plot the impact of learning rate
plot_hyperparameter_impact(learning_rates, accuracies, "Learning Rate")


                                        # Impact of Number of Neurons in a Layer
num_neurons = [0, 32, 64, 128, 256, 512]
accuracies = [0., 0.78, 0.85, 0.9, 0.8, 0.7]  # Simulated accuracies

                                        # Plot the impact of number of neurons
plot_hyperparameter_impact(num_neurons, accuracies, "Number of Neurons")
