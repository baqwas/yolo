#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Similarity Image Search using Ultralytics Python package
Author: Matha Goram
Date: 2024-10-10
Description: This script demonstrates how to perform similarity image search using the Ultralytics Python package.
It loads a pre-trained model, processes a query image, and searches for similar images in a specified directory.
@license: MIT
@version: 1.0
@status: Development
@repository: GitHub - baqwas
@maintainer: Matha Goram
@contact: armw
@see: https://docs.ultralytics.com/guides/similarity-search/#how-it-works
Keywords: similarity image search, ultralytics, python, computer vision, image retrieval
Usage: python similarity_image_search.py
Requirements: ultralytics package
Example: python similarity_image_search.py --data path/to/img/directory --device cpu
    * Loads or builds an index from local images
    * Extracts image and text embeddings using CLIP
    * Performs similarity search using cosine similarity

"""
from ultralytics import solutions

searcher = solutions.VisualAISearch(
    # data = "path/to/img/directory" # Optional, build search engine with your own images
    device="cpu"  # configure the device for processing i.e "cpu" or "cuda"
)

results = searcher("a dog sitting on a bench")

# Ranked Results:
#     - 000000546829.jpg | Similarity: 0.3269
#     - 000000549220.jpg | Similarity: 0.2899
#     - 000000517069.jpg | Similarity: 0.2761
#     - 000000029393.jpg | Similarity: 0.2742
#     - 000000534270.jpg | Similarity: 0.2680