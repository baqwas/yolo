#!/usr/bin/env python3
"""
    epochvsbatch.py
    Illustrate epoch versus batch size
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
        epochvsbatch.py
"""
import matplotlib.pyplot as plt

# Define the data (replace with your actual training results)
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Example epoch values
batch_sizes = [16, 32, 64, 128, 256]  # Example batch sizes
accuracies = [
    [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.87, 0.88, 0.9],  # Accuracy for batch size 16
    [0.45, 0.65, 0.72, 0.78, 0.83, 0.86, 0.88, 0.9, 0.91, 0.92],  # Accuracy for batch size 32
    [0.4, 0.58, 0.68, 0.75, 0.8, 0.85, 0.88, 0.9, 0.91, 0.92],  # Accuracy for batch size 64
    [0.35, 0.55, 0.65, 0.72, 0.78, 0.82, 0.85, 0.88, 0.9, 0.91],  # Accuracy for batch size 128
    [0.3, 0.5, 0.6, 0.68, 0.75, 0.8, 0.83, 0.86, 0.88, 0.9]   # Accuracy for batch size 256
]

# Create a plot
plt.figure(figsize=(10, 6))

# Plot accuracy curves for each batch size
for i, acc in enumerate(accuracies):
    plt.plot(epochs, acc, label=f"Batch Size: {batch_sizes[i]}")

# Set plot labels and title
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Epoch vs. Batch Size")
plt.legend()

# Show the plot
plt.grid(True)
plt.show()