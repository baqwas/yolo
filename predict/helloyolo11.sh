#!/usr/bin/env bash
# Script: helloyolo11.sh
# Purpose: Run a YOLO prediction on an image
# History:
# 2024-10-10 0.1 armw Initial DRAFT
# Description
# This script runs YOLO object detection inference from the command line.
# The basic syntax for the command line is:
# YOLO task mode args
# where
# task is optional from (detect, segment, classify, pose, obb)
# mode is mandatory from (train, val, predict, export, track, benchmark)
# args is optional expressed as key=value pairs
#
echo "Running $0"
task=detect # choose from {detect, segment, classify, pose, obb}
mode=predict # choose from {train, val, predict, export, track, benchmark}
models=("n" "s" "m" "l" "x") # nano, small, medium, large & experimental
image="https://ultralytics.com/images/bus.jpg" # benchmark Ultralytics image
for str in "${models[@]}"; do # iterate over all models
  model="yolo11${str}.pt" # synthesize model name as string
  echo "${mode} ${task} ${model} ${image}"
  yolo ${mode} ${task} model=${model} source=${image} # invoke CLI
done
echo "All done!"
