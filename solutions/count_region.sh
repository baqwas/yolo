#!/usr/bin/env bash

echo "count objects in a region of an image"

# Default value for the first parameter
default_source="/home/reza/Pictures/Travels/Airports/KDFW\ D24\ queue.jpg"
# Check if the source image is provided
if [[ -z "$1" ]]; then
  source_image="${default_source}"
else
  source_image="$1"
fi
echo "Using ${source_image} as source image"

# Default value for the second parameter
default_region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]"
# Check if the image region is provided
if [[ -z "$2" ]]; then
  image_region="${default_region}"
else
  image_region="$2"
fi
echo "Using ${image_region} as the region for counting"

# Run a counting example
# yolo solutions count show=True

# Pass a source video
yolo solutions count source=${source_image}

# Pass region coordinates
# yolo solutions count region=${image_region}