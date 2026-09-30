# YOLO26

## Notes from [website](https://docs.ultralytics.com/models/yolo26):
The YOLO26 model family is built around four design areas: 
* **Native end-to-end inference**: The optional one-to-one detection head produces predictions without non-maximum suppression (NMS), simplifying deployment and reducing post-processing. 
* **Lighter box regression**: YOLO26 removes Distribution Focal Loss (DFL), reducing detection-head complexity while preserving an unconstrained regression range. 
* **Training recipe updates**: The training pipeline combines MuSGD (a hybrid Muon + SGD optimizer), Progressive Loss, and STAL (Small-Target-Aware Label Assignment) to improve optimization, shift supervision toward the inference-time head, and maintain positive label coverage for small objects. The full hyperparameters behind the released checkpoints are documented in the YOLO26 Training Recipe guide. 
* **Task-specific heads and losses** : YOLO26 adds targeted designs for instance segmentation, semantic segmentation variants, pose estimation, and oriented detection while keeping a single model pipeline across tasks.

Together, these updates improve the accuracy-latency tradeoff across model scales and deployment targets.

## Key Features 


#### Citation
@misc{jocher2026ultralyticsyolo26unifiedrealtime,
  title = {Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models},
  author = {Glenn Jocher and Jing Qiu and Mengyu Liu and Shuai Lyu and Fatih Cagatay Akyon and Muhammet Esat Kalfaoglu},
  year = {2026},
  eprint = {2606.03748},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.48550/arXiv.2606.03748},
  url = {https://arxiv.org/abs/2606.03748},
}