#!/usr/bin/env python3
"""
    epochvsbatchvsaccuracy.py
    Illustrate epoch versus batch size versus accuracy
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
        epochvsbatchvsaccuracy.py
"""
import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace with your actual results)
epochs = [1, 2, 3, 4, 5]  # List of epochs
batch_sizes = [16, 32, 64, 128]  # List of batch sizes
accuracies = np.array([[0.7, 0.75, 0.8, 0.82, 0.83],  # Accuracy for batch size 16
                      [0.72, 0.78, 0.85, 0.87, 0.88],  # Accuracy for batch size 32
                      [0.68, 0.75, 0.82, 0.86, 0.89],  # Accuracy for batch size 64
                      [0.65, 0.7, 0.78, 0.84, 0.87]])  # Accuracy for batch size 128

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for epochs and batch sizes
X, Y = np.meshgrid(epochs, batch_sizes)

# Plot the accuracy surface
ax.plot_surface(X, Y, accuracies, cmap='viridis')

# Set labels and title
ax.set_xlabel('Epochs')
ax.set_ylabel('Batch Size')
ax.set_zlabel('Accuracy')
ax.set_title('Epoch vs Batch Size vs Accuracy')

# Show the plot
plt.show()