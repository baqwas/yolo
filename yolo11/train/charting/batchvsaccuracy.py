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
"""
import matplotlib.pyplot as plt

# Sample data (replace with your actual training data)
batch_sizes = [16, 32, 64, 128, 256]  # Example batch sizes
accuracies = [0.85, 0.88, 0.90, 0.87, 0.84]  # Example accuracies

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(batch_sizes, accuracies,
         marker='s',  # Square marker
         linestyle='--',  # Dashed line
         color='blue',
         linewidth=2,
         markersize=8)

plt.title("Batch Size vs. Accuracy", fontsize=16)  # Increase title font size
plt.xlabel("Batch Size", fontsize=14)  # Increase x-axis label font size
plt.ylabel("Accuracy", fontsize=14)  # Increase y-axis label font size
plt.grid(True)

plt.show()