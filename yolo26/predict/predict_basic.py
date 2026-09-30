#!/usr/bin/env python3
"""
================================================================================
Document Name: Batch YOLO Inference Script with TOML Configuration
Purpose: Automatically loads image paths from a companion TOML configuration file,
         performs batched object detection inference using the Ultralytics YOLO framework,
         and saves/displays the annotated output results.
Version: 1.0.0
Update History:
    - 2026-09-27: Initial implementation with dynamic TOML path resolution and pathlib globbing.
Author: Matha Goram
Copyright: (c) 2026 ParkCircus Productions
License: MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, including without limitation the rights to use, copy, modify, 
merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to 
the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Prerequisites:
    - Python 3.8+
    - ultralytics package (`pip install ultralytics`)
    - tomli package (for Python < 3.11, or use built-in tomllib for Python 3.11+)

User Interface Guide:
    - Headless/CLI execution model. Configure input parameters in the companion .toml file.
    - Displays processing feedback via standard output and invokes visual windows via YOLO.

Processing Workflow:
    1. Resolve companion TOML configuration file based on the executing script's filename.
    2. Parse configuration parameters (target image directory, file extension, model weights).
    3. Traverse target directory using glob patterns to assemble a list of image file paths.
    4. Execute batched tensor inference via the YOLO neural network model.
    5. Iterate through results to extract bounding boxes, masks, keypoints, and persist output visualizations.

Algorithms:
    - Deep convolutional neural network object detection architecture (Ultralytics YOLO inference engine).
    - File system pattern matching via glob path filters.

References:
    - Ultralytics YOLO Documentation: https://docs.ultralytics.com/
    - Python Software Foundation: pathlib and configuration parsing modules.

Notes:
    - Ensure the companion TOML file exists in the identical directory with matching stem name.
================================================================================
"""

from pathlib import Path
import sys

# Handle TOML compatibility across Python versions
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("Error: A TOML parser is required. Please install 'tomli' for Python < 3.11 (`pip install tomli`).")
        sys.exit(1)

from ultralytics import YOLO


def load_config() -> dict:
    """Loads configuration parameters from a TOML file sharing the script's stem name."""
    script_path = Path(__file__).resolve()
    toml_path = script_path.with_suffix(".toml")

    if not toml_path.exists():
        print(f"Error: Companion configuration file '{toml_path.name}' not found.")
        sys.exit(1)

    with open(toml_path, "rb") as f:
        config = tomllib.load(f)
    return config


def main() -> None:
    # 1. Load configuration parameters from companion TOML
    config = load_config()

    model_name = config.get("model", "yolo26n.pt")
    image_dir = Path(config.get("image_directory", "."))
    file_extension = config.get("file_extension", "*.jpg")
    output_dir = Path(config.get("output_directory", "./output"))

    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Initialize the YOLO model
    model = YOLO(model_name)

    # 3. Gather all matching files into a list of strings using pathlib globbing
    image_files = [str(file) for file in image_dir.glob(file_extension)]

    # 4. Execute inference workflow
    if image_files:
        print(f"Found {len(image_files)} images matching '{file_extension}' in '{image_dir}'. Running inference...")
        results = model(image_files)

        # 5. Process and persist results
        for idx, result in enumerate(results):
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs

            # Save individual results securely
            out_filename = output_dir / f"result_{idx}.jpg"
            result.save(filename=str(out_filename))
            print(f"Saved: {out_filename}")
    else:
        print(f"No files matching pattern '{file_extension}' found in directory '{image_dir}'.")


if __name__ == "__main__":
    main()