# ==================================
# Ultralytics YOLO26 + EasyOCR
# Automatic Number Plate Recognition
# ==================================

import sys
import cv2
import torch
import easyocr
import numpy as np
from pathlib import Path
from typing import Union, Optional, Any, cast
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

try:
    import tomllib  # Native in Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for Python < 3.11


class ANPR:
    """Automatic Number Plate Recognition using Ultralytics YOLO and EasyOCR.

    This class handles license plate detection using a YOLO model and text extraction
    using EasyOCR. It supports both image and video streams for real-time inference.
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

        if isinstance(source, int):
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"❌ Error: Unable to open video capture source '{source}'. Exiting.")
            sys.exit(1)

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


def load_config() -> dict[str, Any]:
    """Loads configuration values from a TOML file matching the script's filename."""
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    cwd_dir = Path.cwd()
    config_path = script_path.with_suffix(".toml")

    defaults: dict[str, Any] = {
        "model_path": "yolo26n.pt",
        "source": "acar-3.mp4",
        "output_path": "anpr_output.mp4",
        "display": True,
    }

    if not config_path.exists():
        print(f"⚠️ Warning: Configuration file '{config_path.name}' not found. Using default settings.")
        raw_cfg = defaults
    else:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        raw_cfg = data.get("anpr", data)

    official_models = {"yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "yolo11s.pt", "yolo26n.pt", "yolo26s.pt"}

    def resolve_model_path(val: str) -> str:
        if val in official_models:
            return val

        p = Path(val)
        if p.is_absolute():
            if p.exists():
                return str(p)
        else:
            for search_dir in (script_dir, cwd_dir):
                candidate = search_dir / p
                if candidate.exists():
                    return str(candidate)

        print(f"❌ Error: Model weights file '{val}' was not found in '{script_dir}' or '{cwd_dir}'.")
        print("   Please check the model path in 'anpr_basic.toml' or copy the file into the working directory.")
        sys.exit(1)

    def resolve_source_path(val: Union[str, int]) -> Union[str, int]:
        if isinstance(val, int) or not isinstance(val, str):
            return val
        if val.startswith(("rtsp://", "http://", "https://")):
            return val

        p = Path(val)
        if p.is_absolute():
            if p.exists():
                return str(p)
        else:
            for search_dir in (script_dir, cwd_dir):
                candidate = search_dir / p
                if candidate.exists():
                    return str(candidate)

        print(f"❌ Error: Input video file '{val}' was not found in '{script_dir}' or '{cwd_dir}'.")
        print("   Please check the source path in 'anpr_basic.toml' or verify file availability.")
        sys.exit(1)

    def resolve_output_path(val: Optional[str]) -> Optional[str]:
        if not val:
            return None
        p = Path(val)
        return str(p if p.is_absolute() else script_dir / p)

    return {
        "model_path": resolve_model_path(raw_cfg.get("model_path", defaults["model_path"])),
        "source": resolve_source_path(raw_cfg.get("source", defaults["source"])),
        "output_path": resolve_output_path(raw_cfg.get("output_path", defaults["output_path"])),
        "display": raw_cfg.get("display", defaults["display"]),
    }


if __name__ == "__main__":
    cfg = load_config()

    anpr = ANPR(model_path=str(cfg["model_path"]))
    anpr.infer_video(
        source=cfg["source"],
        output_path=cfg["output_path"],
        display=bool(cfg["display"]),
    )