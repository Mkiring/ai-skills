# ONNX to CVIMODEL Conversion Skill

> **Maintained by**: [LynnL4](https://github.com/LynnL4)
> **License**: MIT

Expert guidance skill for converting ONNX models to CVIMODEL format for Sophgo CV181x TPU platforms (reCamera, SG200x).

## Overview

This skill provides practical scripts and tested configurations for:
- YOLO11: detection, pose, segmentation, classification
- YOLO26: detection, classification
- BiSeNetv2: semantic segmentation (Cityscapes)
- ION memory optimization (--quant_output)
- Hybrid quantization with qtable support
- ONNX vs CVIMODEL validation workflow
- Ready-to-use conversion scripts

## Version

Current Version: **v2.0.0**

## What's Supported

| Model | Detection | Pose | Segmentation | Classification |
|-------|-----------|------|--------------|----------------|
| YOLO11 | OK | OK | OK | OK |
| YOLO26 | OK | -- (Mod not supported) | -- (Mod not supported) | OK |
| BiSeNetv2 | - | - | Semantic Seg (19-class Cityscapes) | - |

## Structure

```
onnx-to-cvimodel/
+-- SKILL.md              # Main skill definition
+-- LICENSE.txt           # MIT License
+-- README.md             # This file
+-- .skillrc              # Skill configuration
+-- scripts/
|   +-- convert_to_cvimodel.sh      # Universal conversion script
|   +-- batch_convert_all.sh        # Batch conversion
|   +-- convert_yolo11_detect.sh    # YOLO11 detection
|   +-- convert_yolo11_pose.sh      # YOLO11 pose
|   +-- convert_yolo11_seg.sh       # YOLO11 segmentation
|   +-- convert_yolo11_cls.sh       # YOLO11 classification
|   +-- convert_yolo26_detect.sh    # YOLO26 detection
|   +-- convert_yolo26_cls.sh       # YOLO26 classification
|   +-- convert_bisenetv2.sh        # BiSeNetv2 semantic segmentation
|   +-- validate_conversion.py      # ONNX vs CVIMODEL validation
+-- assets/
    +-- yolo11n_pose_qtable         # Pose hybrid quantization
    +-- yolo11n_seg                 # Segmentation hybrid quantization
```

## Quick Start

### 1. Export YOLO to ONNX

```python
from ultralytics import YOLO
YOLO('yolo11n.pt').export(format='onnx', imgsz=640, simplify=False, opset=12)
YOLO('yolo11n-pose.pt').export(format='onnx', imgsz=640, simplify=False, opset=12)
YOLO('yolo11n-seg.pt').export(format='onnx', imgsz=640, simplify=False, opset=12)
```

### 2. Convert to CVIMODEL

```bash
# YOLO models
./scripts/convert_to_cvimodel.sh model.onnx dataset/

# BiSeNetv2 (produces INT8 + BF16 + INT8_quant_output)
./scripts/convert_bisenetv2.sh bisenetv2.onnx dataset/
```

### 3. Validate Conversion

```bash
# Inside TPU-MLIR Docker
docker run --rm \
    -v /path/to/work:/work \
    -v /path/to/tpu-mlir:/workspace/tpu-mlir \
    sophgo/tpuc_dev:v3.1 python3 /work/scripts/validate_conversion.py \
    /work/model.onnx /work/model_int8_qout.cvimodel /work/test_images \
    --output_dir /work/validation_results
```

Outputs per-image pixel agreement, per-class IoU, comparison images, and summary CSV.

## ION Memory Optimization

CV181x ION memory is shared (~60MB total). Use `--quant_output` to keep int8 output:

| BiSeNetv2 Variant | Model Size | ION Memory |
|-------------------|-----------|------------|
| BF16 | 14 MB | 87.59 MB |
| INT8 (float32 output) | 6.0 MB | 62.89 MB |
| **INT8 (--quant_output)** | **5.9 MB** | **34.27 MB** |

## BiSeNetv2 Validation Results

| Metric | Value |
|--------|-------|
| Pixel agreement (image 1) | 98.76% |
| Pixel agreement (image 2) | 99.00% |
| road IoU | 0.9982 |
| building IoU | 0.9821 |
| car IoU | 0.9161 |
| Device inference (CV181x) | **436ms** (pre:2 + infer:276 + post:158) |

## Key Pitfalls (Learned from Production)

1. **Mount local tpu-mlir** - Docker image has empty /workspace/tpu-mlir
2. **Input is uint8 RGB NHWC** - fuse_preprocess handles normalization, NOT float32 NCHW
3. **Use --quant_output** for ION-constrained devices
4. **int8 argmax is valid** - uniform quantization preserves relative ordering
5. **Validate after conversion** - always check pixel agreement

## Requirements

- Docker with `sophgo/tpuc_dev:v3.1` image
- Local tpu-mlir installation
- Calibration images (100+ recommended for production)
- ONNX model file

## License

This skill is licensed under the [MIT License](LICENSE.txt).
