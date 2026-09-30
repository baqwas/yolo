# 📑 System Architecture & Technical Specification
## 🚗 Automatic Number Plate Recognition (ANPR) Module

---

### 📋 Document Control

| 🏷️ Attribute | 📌 Value |
| :--- | :--- |
| **📄 Document Title** | Technical Specification & Operation Guide: `anpr_basic.py` |
| **🏷️ System Version** | `v1.3.0-Production` 🟢 |
| **🟢 Status** | Approved / Production Release |
| **🐍 Target Runtime** | Python 3.10+ |
| **⚡ Hardware Targets** | NVIDIA Jetson / x86_64 Workstation (CUDA enabled or CPU fallback) |

---

### 🎯 1. Purpose & Scope

The `ANPR` module provides a high-throughput, edge-capable Automatic Number Plate Recognition framework. It unifies object detection and optical character recognition (OCR) into a streamlined, low-latency processing pipeline.

```
                  ┌─────────────────────────────────────────┐
                  │   🚗 ANPR System Pipeline Architecture  │
                  └────────────────────┬────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌───────────────────────┐                             ┌───────────────────────┐
│ 🔍 Spatial Detection  │                             │ 🔤 Sequence Recognition│
│   Ultralytics YOLO    │                             │      EasyOCR / CRNN   │
└───────────────────────┘                             └───────────────────────┘
```

#### 🛡️ Primary Deployment Scenarios
* 🛑 **Access Control**: Automated gate validation for corporate and residential facilities.
* 🅿️ **Parking Telemetry**: Entry/exit timestamps and automated billing processing.
* 🛣️ **Traffic Monitoring**: Real-time vehicle density tracking and speed enforcement analytics.
* 📹 **Surveillance Feeds**: Multi-camera feed indexing and license plate query logging.

---

### 📜 2. Version & Update History

| 🏷️ Version | 📅 Date | 🟢 Status | 📝 Description of Changes |
| :--- | :--- | :--- | :--- |
| `v1.0.0` | 2026-01-15 | 🟡 Deprecated | 🚀 **Initial Release**: Baseline implementation combining YOLO plate detection and EasyOCR text extraction. |
| `v1.1.0` | 2026-02-10 | 🟡 Deprecated | 🛠️ **Patch**: Resolved `Literal[0]` and `None` static analysis warnings by adding `typing.Union` and `typing.Optional` annotations. |
| `v1.2.0` | 2026-03-01 | 🟡 Deprecated | 🔧 **Technical Patch**: Addressed PyCharm iterator inspection flags by explicitly casting prediction streams to `list` structures. |
| `v1.3.0` | 2026-03-24 | 🟢 Current | 🛡️ **Type-Safe Refactor**: Replaced fragile `getattr` chaining with explicit `typing.cast(Any, ...)` to bypass incomplete third-party stubs. Resolved `cv2.VideoCapture` type-branching warnings and eliminated all static IDE type flags. |

---

### 📦 3. Prerequisites & Dependencies

#### 3.1 💻 Core System Environment
* 🐍 **Python**: `version >= 3.10`
* ⚡ **CUDA Toolkit** *(Optional, recommended)*: `v11.8` or `v12.x` (for hardware-accelerated GPU inference)

#### 3.2 📚 Python Package Requirements
```text
ultralytics>=8.1.0   # 🎯 YOLO Object Detection Engine
easyocr>=1.7.0       # 🔤 CRNN Sequence Text Recognition
opencv-python>=4.8.0 # 🖼️ Frame Capture & Spatial Transforms
torch>=2.0.0         # 🧠 Deep Learning Execution Framework
numpy>=1.24.0        # 🔢 High-Performance Matrix Operations
```

#### 3.3 🖥️ Target Hardware Specifications
* 🟢 **GPU Execution (Accelerated)**: NVIDIA GPU with Compute Capability $\ge 6.0$, minimum 4 GB VRAM.
* 🔵 **CPU Execution (Fallback)**: x86_64 or ARM64 multi-core processor with AVX2 / NEON vector instruction sets.

---

### 🔌 4. User Interface & API Reference

The module exposes an object-oriented Python API via the `ANPR` class, designed for seamless integration into edge orchestration workflows or direct execution via CLI.

#### 4.1 🏗️ Class Instantiation
```python
from anpr_basic import ANPR

# 🚀 Initialize with custom trained license plate model
anpr = ANPR(model_path="models/number-plate-yolo26s.pt")
```

#### 4.2 ⚙️ Method Specifications

##### 🔹 `ANPR.__init__(model_path: str = "yolo26n.pt")`
Instantiates the inference pipeline, detects available CUDA hardware, loads the YOLO model into VRAM/RAM, and initializes the EasyOCR engine for the English language (`"en"`).

##### 🔹 `ANPR.detect_plates(im0: np.ndarray) -> np.ndarray`
Executes spatial detection on a single image frame.
* 📥 **Input**: `im0` — BGR image frame formatted as a NumPy array (`H x W x C`).
* 📤 **Returns**: Array of bounding boxes with format `[[x1, y1, x2, y2], ...]`. Returns an empty list `[]` if no plates are detected.

##### 🔹 `ANPR.extract_text(im0: np.ndarray, bbox: np.ndarray) -> str`
Crops the specified Region of Interest (ROI), applies grayscale transformations, and executes OCR inference.
* 📥 **Inputs**: 
  * `im0`: Source BGR frame (`np.ndarray`).
  * `bbox`: Bounding box array `[x1, y1, x2, y2]`.
* 📤 **Returns**: Stripped text string containing detected alphanumeric characters.

##### 🔹 `ANPR.infer_video(source: Union[str, int] = 0, output_path: Optional[str] = None, display: bool = True) -> None`
Runs real-time ANPR against a local video file, RTSP stream URL, or USB camera device index.
* 📥 **Parameters**:
  * `source`: Video file path (`str`) or USB camera device index (`int`). Default: `0`.
  * `output_path`: Optional target file path (`str`) to save encoded `.mp4` output video.
  * `display`: Boolean flag enabling local OpenCV GUI window display (`cv2.imshow`).

---

### 🔄 5. System Processing Workflow

The sequence below illustrates frame progression from ingestion to annotated visualization:

```
📹 [ Input Stream / File ]
            │
            ▼
   ( cv2.VideoCapture )
            │
            ▼
   🖼️ [ Raw BGR Frame (im0) ]
            │
            ▼
 🎯 [ detect_plates(im0) ] ──────► 🧠 YOLO Neural Model
            │
            ▼
 🔲 [ Bounding Boxes (xyxy) ]
            │
            ├───► 🎨 [ Annotator Box Drawing ] ───────────────┐
            │                                                 │
            ▼                                                 ▼
 🔤 [ extract_text() ] ───► ✂️ Crop ROI ───► 🎨 Gray ───► 🔤 EasyOCR
            │                                                 │
            └───────────────────────┬─────────────────────────┘
                                    │
                                    ▼
                     🏷️ [ Annotated Output Frame ]
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
💻 [ OpenCV Display Window ]                   💾 [ VideoWriter (.mp4) ]
```

---

### 🧮 6. Core Algorithms & Mathematical Formulations

#### 6.1 🎯 Spatial Detection (YOLO)
The object detector outputs bounding box tensor representations defined by top-left and bottom-right corner coordinates:

$$\mathbf{B} = \{ (x_1, y_1, x_2, y_2, c, p) \mid p \ge p_{\text{threshold}} \}$$

Where $c$ represents class identity and $p$ denotes confidence score. Non-Maximum Suppression (NMS) removes redundant overlapping bounding boxes based on Intersection over Union (IoU):

$$\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}$$

#### 6.2 🎨 Region of Interest (ROI) Transformation
Once a bounding box is isolated, the crop matrix is converted to single-channel luminance space using standard ITU-R Recommendation BT.601 color weights:

$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

This transformation eliminates chromatic noise while maximizing structural character edge contrast for the downstream sequence recognition engine.

#### 6.3 🧠 Sequence Recognition (CRNN + CTC)
EasyOCR processes the preprocessed ROI $Y$ using a Convolutional Recurrent Neural Network (CRNN):
1. 🔍 **Feature Extraction**: Deep CNN layers extract spatial feature maps from the cropped image tensor.
2. 🔄 **Sequence Recurrent Network**: Bidirectional LSTM layers transcribe frame-by-frame feature vectors across horizontal spatial sequences.
3. 🔤 **CTC Transcription**: Connectionist Temporal Classification (CTC) decodes sequence probabilities into a final character string without requiring character-level alignment annotations.

---

### 🚨 7. Exception & Error Response Matrix

| 🛑 Operational Failure | 🔍 Root Cause | ⚠️ System Behavior | 💡 Remediation Strategy |
| :--- | :--- | :--- | :--- |
| `ValueError: Cannot open video source: <source>` | File path does not exist, camera index invalid, or stream offline. | 💥 Exception raised; execution terminates safely. | 📌 Verify file path spelling, USB index (`0, 1...`), or RTSP stream network accessibility. |
| `torch.cuda.OutOfMemoryError` | VRAM exhausted due to high resolution or parallel model instances. | 💥 Application crashes during tensor operation. | 📌 Reduce input video frame dimensions or force CPU fallback by passing `gpu=False` to EasyOCR. |
| 🟡 Empty string output from `extract_text()` | ROI resolution too small, low contrast, or severe motion blur. | ⚠️ Returns `""`. Bounding box rendered without label. | 📌 Implement minimum pixel area filtering prior to crop, or apply contrast stretching (`CLAHE`). |
| ⚠️ PyCharm static inspection flags | Incomplete typing stubs in Ultralytics or OpenCV C++ bindings. | 🟡 IDE flags false positive warnings (`Any \| None`, `Union`). | 🟢 Addressed in `v1.3.0` via `typing.cast(Any, ...)` and explicit type branching. |

---

### 💻 8. Implementation Listing

Below is the complete, production-ready Python implementation (`anpr_basic.py`) engineered with static type-safety patterns:

```python
# ==================================
# Ultralytics YOLO26 + EasyOCR
# Automatic Number Plate Recognition
# ==================================

import cv2
import torch
import easyocr
import numpy as np
from typing import Union, Optional, Any, cast
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


class ANPR:
    """Automatic Number Plate Recognition using Ultralytics YOLO and EasyOCR.

    This class handles license plate detection using a YOLO model and text extraction
    using EasyOCR. It supports both image and video streams for real-time inference.

    Attributes:
        model (YOLO): The YOLO model for license plate detection.
        reader (easyocr.Reader): The OCR reader instance for text recognition.
        device (torch.device): Computation device (CPU or CUDA).
    """

    def __init__(self, model_path: str = "yolo26n.pt"):
        """Initializes the ANPR system."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    def detect_plates(self, im0: np.ndarray):
        """Detects license plates in an image."""
        results = list(self.model.predict(im0, verbose=False))
        if not results:
            return []

        # Casting to Any bypasses PyCharm's incomplete third-party type stubs
        res = cast(Any, results[0])
        if res.boxes is not None:
            return res.boxes.xyxy.cpu().numpy()

        return []

    def extract_text(self, im0: np.ndarray, bbox: np.ndarray):
        """Performs OCR on the cropped license plate region."""
        x1, y1, x2, y2 = map(int, bbox)
        roi = im0[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        text = self.reader.readtext(gray, detail=0, paragraph=True)
        return " ".join(text).strip() if text else ""

    def infer_video(self, source: Union[str, int] = 0, output_path: Optional[str] = None, display: bool = True):
        """Performs real-time ANPR on a video stream."""

        # Explicit type branching to satisfy cv2.VideoCapture overloads
        if isinstance(source, int):
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("🚀 Starting ANPR video inference... Press 'q' to quit.")

        while True:
            ret, im0 = cap.read()
            if not ret:
                break

            boxes = self.detect_plates(im0)
            ann = Annotator(im0, line_width=4)
            for bbox in boxes:
                text = self.extract_text(im0, bbox)
                ann.box_label(bbox, label=text, color=colors(17, True))

            if display:
                cv2.imshow("ANPR (Press 'q' to exit)", im0)
            if writer:
                writer.write(im0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    anpr = ANPR(model_path="number-plate-yolo26s.pt")  # Use trained YOLO license plate model
    anpr.infer_video(source="acar-3.mp4", output_path="anpr_output.mp4", display=True)
```

---

### 💡 9. Engineering Notes & Operational Best Practices

* ⚡ **Frame Skipping Optimization**: Running EasyOCR on every frame can create processing bottlenecks on embedded devices. For high-FPS streams, implement object tracking (e.g., DeepSORT or ByteTRACK) and perform OCR once every 5-10 frames per unique track ID.
* 🌙 **Low-Light / Night Enhancement**: For nighttime or low-contrast environments, applying Contrast Limited Adaptive Histogram Equalization (`cv2.createCLAHE()`) to the cropped grayscale ROI significantly boosts OCR accuracy.
* 🛡️ **IDE Static Type Analysis**: Deep learning frameworks wrapped over C++ bindings often expose dynamic runtime structures. Using `typing.cast(Any, ...)` is the standard, zero-overhead strategy to satisfy IDE static checkers without adding runtime overhead.

---

### 📚 10. Technical References

* 🎯 **Ultralytics YOLO Architecture & Type Annotations**
* 🔤 **EasyOCR: Ready-to-Use Optical Character Recognition Framework**
* 🖼️ **OpenCV Video I/O & Image Processing Modules**
* 🐍 **PEP 484 – Type Hints & PEP 526 – Syntax for Variable Annotations**