#!/usr/bin/env python3
"""
    learningrateaccuracy.py
    Train settings parametric trends
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

# Sample data (replace with your actual results)
batch_sizes = [16, 32, 64, 128]
accuracies_batch = [0.85, 0.90, 0.92, 0.88]

weight_decays = [0.0, 0.0001, 0.001, 0.01]
accuracies_weight_decay = [0.88, 0.91, 0.90, 0.85]

optimizers = ['Adam', 'SGD', 'RMSprop']
accuracies_optimizer = [0.92, 0.89, 0.87]

loss_functions = ['categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy']
accuracies_loss = [0.90, 0.88, 0.92]

# Create subplots
plt.figure(figsize=(8, 6))

# Plot 1: Batch Size vs. Accuracy
plt.plot(batch_sizes, accuracies_batch, marker='o', linestyle='-', color='b')
plt.title("Accuracy vs. Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("Accuracy")
plt.show()
plt.clf()
plt.cla()

# Plot 2: Weight Decay vs. Accuracy
plt.plot(weight_decays, accuracies_weight_decay, marker='o', linestyle='-', color='r')
plt.title("Accuracy vs. Weight Decay")
plt.xlabel("Weight Decay")
plt.ylabel("Accuracy")
plt.show()
plt.clf()
plt.cla()

# Plot 3: Optimizer vs. Accuracy
plt.bar(optimizers, accuracies_optimizer)
plt.title("Accuracy vs. Optimizer")
plt.xlabel("Optimizer")
plt.ylabel("Accuracy")
plt.show()
plt.clf()
plt.cla()

# Plot 4: Loss Function vs. Accuracy
plt.bar(loss_functions, accuracies_loss)
plt.title("Accuracy vs. Loss Function")
plt.xlabel("Loss Function")
plt.ylabel("Accuracy")

#plt.tight_layout()
plt.show()