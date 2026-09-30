#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Image Search using Ultralytics Python package
Author: Matha Goram
Date: 2024-10-10
Description: This script demonstrates how to perform semantic image search using the Ultralytics Python package. It loads a pre-trained model, processes a query image, and searches for similar images in a specified directory.
@license: MIT
@version: 1.0
@status: Development
@repository: GitHub - baqwas
@maintainer: Matha Goram
@contact: armw
@see: https://docs.ultralytics.com/guides/similarity-search/#how-it-works
Keywords: semantic image search, ultralytics, python, computer vision, image retrieval
Usage: python semantic_image_search.py
Requirements: ultralytics package
Example: python semantic_image_search.py --data path/to/img/directory --device cpu
    * Vision encoder, ResNet or ViT
    * Build an index of the image embeddings and enables fast, scalable retrieval of the closest vectors to a given query
    * Provide a simple web interface to submit natural language queries and display semantically
        matched images from the index
            
"""
from ultralytics import solutions

app = solutions.SearchApp(
    # data = "path/to/img/directory" # Optional, build search engine with your own images
    device="cpu"  # configure the device for processing i.e "cpu" or "cuda"
)

app.run(debug=False)  # You can also use `debug=True` argument for testing