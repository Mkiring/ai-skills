---
name: onnx-to-cvimodel
description: "Expert guide for converting ONNX models to CVIMODEL format for Sophgo CV181x TPU. Supports YOLO11/YOLO26 (detect, pose, seg, cls) and BiSeNetv2 (semantic segmentation). Includes tested conversion scripts, quantization tables (qtables), ION memory optimization (--quant_output), validation workflow, and complete documentation. Use when user needs to convert ONNX models to CVIMODEL, set up TPU-MLIR conversion pipeline, configure output names and quantization, optimize ION memory usage, validate conversion accuracy, or troubleshoot conversion issues."
compatibility: Requires Docker, the sophgo/tpuc_dev:v3.1 image, a local TPU-MLIR installation, an ONNX model, and a calibration dataset.
license: Complete terms in LICENSE.txt
---

# ONNX to CVIMODEL Conversion Guide

This skill provides scripts, configurations, and guidance for converting models from ONNX to CVIMODEL format for Sophgo CV181x TPU (reCamera, SG200x).

## Instructions

### Step 1: Confirm the required environment

Before suggesting or running conversion commands, require:
- Docker
- The `sophgo/tpuc_dev:v3.1` image
- A local TPU-MLIR installation mounted into the container
- An ONNX model file
- A calibration dataset

If a required dependency or input is missing:
- Stop before claiming conversion can proceed
- State exactly what is missing
- Tell the user the skill may be installed, but the current task is blocked until that requirement is provided

### Step 2: Treat missing validation as unresolved

If conversion completes but ONNX vs CVIMODEL validation was not run, do not describe the model as verified.

Supported models:
- **YOLO11/YOLO26**: detection, pose, segmentation, classification
- **BiSeNetv2**: semantic segmentation (Cityscapes)

The conversion process uses **TPU-MLIR** Docker environment and requires:
- ONNX model file
- Calibration dataset (images)
- Correct output names for each model/task combination

## Quick Start - One Command Conversion

```bash
# Export YOLO11n to ONNX and convert to CVIMODEL in one go
python3 export_and_convert.py --model yolo11n --task detect

# Or use existing ONNX
./convert_yolo11_detect.sh yolo11n.onnx dataset/
```

## Available Scripts

### 1. Universal Conversion Script
```bash
./convert_to_cvimodel.sh <onnx_file> <dataset_dir>
```

### 2. Task-Specific Scripts
```bash
./convert_yolo11_detect.sh <onnx> <dataset>     # YOLO11 detection
./convert_yolo11_pose.sh <onnx> <dataset>      # YOLO11 pose (needs qtable)
./convert_yolo11_seg.sh <onnx> <dataset>       # YOLO11 segmentation (needs qtable)
./convert_yolo11_cls.sh <onnx> <dataset>       # YOLO11 classification
./convert_yolo26_detect.sh <onnx> <dataset>     # YOLO26 detection
./convert_yolo26_cls.sh <onnx> <dataset>       # YOLO26 classification
```

### 3. BiSeNetv2 Semantic Segmentation
```bash
./convert_bisenetv2.sh <onnx> <dataset>              # BiSeNetv2 (INT8 + BF16 + INT8_quant_output)
```

### 4. Batch Conversion
```bash
./batch_convert_all.sh    # Convert all ONNX files in current directory
```

## Critical: ION Memory Optimization (--quant_output)

**Problem**: CV181x ION memory is shared between TPU, VPSS, VENC. Total ~60MB.
INT8 models with float32 output dequantize on-chip, consuming massive ION memory.

**Solution**: Use `--quant_output` flag to keep int8 output, avoiding dequantization.

```bash
# Without --quant_output (INT8 model, float32 output) → 62.89MB ION ❌
model_deploy.py ... --quantize INT8 --model model_int8.cvimodel

# With --quant_output (INT8 model, int8 output) → 34.27MB ION ✅
model_deploy.py ... --quantize INT8 --quant_output --model model_int8_qout.cvimodel
```

**When to use `--quant_output`**:
- When ION memory is constrained (< 60MB budget)
- When the model output is used for argmax/classification (int8 argmax is equivalent for uniform quantization)
- BiSeNetV2, classification models, any model where post-processing only needs relative ordering

**When NOT to use `--quant_output`**:
- When absolute float32 values matter (e.g., confidence thresholds at specific values)
- When downstream processing requires float32 precision

| BiSeNetv2 Variant | Model Size | ION Memory | Output Type |
|-------------------|-----------|------------|-------------|
| BF16 | 14 MB | 87.59 MB | float32 |
| INT8 (no --quant_output) | 6.0 MB | 62.89 MB | float32 |
| **INT8 (--quant_output)** | **5.9 MB** | **34.27 MB** | **int8** |

## Critical: Docker tpu-mlir Mount

The Docker image `sophgo/tpuc_dev:v3.1` ships with `/workspace/tpu-mlir` empty.
You MUST mount your local tpu-mlir installation:

```bash
# Correct: mount local tpu-mlir
docker run --rm \
    -v /path/to/local/tpu-mlir:/workspace/tpu-mlir \
    -v /path/to/work:/work \
    sophgo/tpuc_dev:v3.1 bash -c \
    'source /workspace/tpu-mlir/envsetup.sh && ...'

# Wrong: using the empty /workspace/tpu-mlir inside Docker
```

## Critical: CVIMODEL Input Format

Models converted with `--fuse_preprocess` expect **uint8 RGB NHWC** input, NOT float32 NCHW:

```python
# CORRECT: feed raw uint8 RGB data
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (MODEL_W, MODEL_H))
input_data = np.ascontiguousarray(img_resized, dtype=np.uint8)
# Flatten to (1, 1, 1, H*W*3) for input_image_raw
model.inputs[0].data[:] = input_data.reshape(model.inputs[0].data.shape)

# WRONG: feeding float32 NCHW normalized data
```

## Validation Workflow

After conversion, always validate ONNX vs CVIMODEL output consistency:

```python
# 1. ONNX inference (standard float32 pipeline)
from ultralytics import YOLO
import onnxruntime as ort, numpy as np, cv2

sess = ort.InferenceSession("model.onnx")
img = cv2.imread("test.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_float = cv2.resize(img_rgb, (W, H)).astype(np.float32) / 255.0
img_norm = (img_float - mean) / std
img_nchw = np.transpose(img_norm, (2, 0, 1))[np.newaxis].astype(np.float32)
onnx_out = sess.run(None, {input_name: img_nchw})[0]
label_onnx = np.argmax(onnx_out[0], axis=0).astype(np.uint8)

# 2. CVIMODEL inference (uint8 RGB, fuse_preprocess handles normalization)
import pyruntime_cvi as cvi
model = cvi.Model("model_cv181x_int8_qout.cvimodel", output_all_tensors=True)
img_rgb2 = cv2.cvtColor(cv2.imread("test.jpg"), cv2.COLOR_BGR2RGB)
input_data = np.ascontiguousarray(cv2.resize(img_rgb2, (W, H)), dtype=np.uint8)
model.inputs[0].data[:] = input_data.reshape(model.inputs[0].data.shape)
model.forward()
# Find int8 output (name contains "preds" and dtype is int8)
for out in model.outputs:
    if "preds" in out.name:
        cvi_out = out.data
label_cvi = np.argmax(cvi_out[0].astype(np.int32), axis=0).astype(np.uint8)

# 3. Compare
agreement = np.sum(label_onnx == label_cvi) / label_onnx.size * 100
print(f"Pixel agreement: {agreement:.2f}%")
```

## Tested & Validated Conversions

### BiSeNetv2 Cityscapes (Full Validation)

| Metric | Value |
|--------|-------|
| ONNX → CVIMODEL pixel agreement | **98.76%** (image 1), **99.00%** (image 2) |
| road IoU | 0.9982 |
| sidewalk IoU | 0.9896 |
| building IoU | 0.9821 |
| car IoU | 0.9161 / 0.9565 |
| vegetation IoU | 0.9509 / 0.9440 |
| Device inference (CV181x TPU) | pre: 2ms, infer: 276ms, post: 158ms = **436ms total** |
| ION memory (int8_qout) | **34.27 MB** |

Quantization errors are concentrated at:
- Small object edges (pole, traffic sign) — expected INT8 precision loss
- Class boundaries — normal quantization artifacts

### YOLO Series

| Model | ONNX Size | CVIMODEL Size | Command |
|-------|-----------|---------------|---------|
| YOLO11n detect | 10.7 MB | 3.0 MB | `./convert_yolo11_detect.sh yolo11n.onnx dataset/` |
| YOLO11n pose | 11.3 MB | 3.6 MB | `./convert_yolo11_pose.sh yolo11n-pose.onnx dataset/` |
| YOLO11n seg | 11.7 MB | 4.0 MB | `./convert_yolo11_seg.sh yolo11n-seg.onnx dataset/` |
| YOLO11n cls | 10.8 MB | 3.0 MB | `./convert_yolo11_cls.sh yolo11n-cls.onnx dataset/` |
| YOLO26n detect | 9.4 MB | 2.9 MB | `./convert_yolo26_detect.sh yolo26n.onnx dataset/` |
| YOLO26n cls | 11.3 MB | 3.0 MB | `./convert_yolo26_cls.sh yolo26n-cls.onnx dataset/` |
| BiSeNetv2 seg | 13 MB | 5.9 MB (INT8_qout) | `./convert_bisenetv2.sh bisenetv2.onnx dataset/` |

## Model-Specific Output Names (Copy & Paste)

### YOLO11 Detection
```bash
--output_names /model.23/cv2.0/cv2.0.2/Conv_output_0,/model.23/cv3.0/cv3.0.2/Conv_output_0,/model.23/cv2.1/cv2.1.2/Conv_output_0,/model.23/cv3.1/cv3.1.2/Conv_output_0,/model.23/cv2.2/cv2.2.2/Conv_output_0,/model.23/cv3.2/cv3.2.2/Conv_output_0
```

### YOLO26 Detection
```bash
--output_names /model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0,/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0,/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0,/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0,/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0,/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0
```

### YOLO11 Segmentation
```bash
--output_names output0,output1
```

### YOLO11 Pose
```bash
--output_names output0
--quantize_table yolo11n_pose_qtable
```

### BiSeNetv2 Segmentation
```bash
--output_names preds
--input_shapes [[1,3,512,1024]]
--mean 123.675,116.28,103.53 --scale 0.01712475,0.01750700,0.01742919
```

## BiSeNetv2 Conversion Details

BiSeNetv2 is a lightweight semantic segmentation model. Key differences from YOLO:

- **Input shape**: Non-square `[1,3,512,1024]` (H, W configurable via env vars)
- **Preprocessing**: ImageNet mean/scale (`123.675,116.28,103.53` / `0.01712,0.01751,0.01743`)
- **No `--keep_aspect_ratio`**: Fixed resolution input
- **Triple output**: Produces INT8, BF16, and INT8+quant_output CVIMODEL files
- **Recommended**: Use INT8+quant_output variant for ION-constrained devices

### Usage
```bash
# Basic conversion (produces 3 variants: int8, bf16, int8_qout)
./scripts/convert_bisenetv2.sh bisenetv2.onnx ./dataset

# Custom resolution
INPUT_H=256 INPUT_W=512 ./scripts/convert_bisenetv2.sh bisenetv2.onnx ./dataset

# Custom model name
MODEL_NAME=bisenetv2_custom ./scripts/convert_bisenetv2.sh bisenetv2.onnx ./dataset
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `bisenetv2_cityscapes` | Model name used in output filenames |
| `CHIP` | `cv181x` | Target chip |
| `INPUT_H` | `512` | Input height |
| `INPUT_W` | `1024` | Input width |
| `CALIBRATION_EPOCHS` | `20` | Number of calibration images |
| `OUTPUT_NAME` | `preds` | ONNX output tensor name |
| `MEAN` | `123.675,116.28,103.53` | ImageNet mean |
| `SCALE` | `0.01712475,0.01750700,0.01742919` | ImageNet scale |
| `QUANT_OUTPUT` | `true` | Whether to generate --quant_output variant |

## Docker Command Template (Manual)

```bash
docker run --privileged --rm \
    -v /path/to/local/tpu-mlir:/workspace/tpu-mlir \
    -v $PWD:/work \
    -w /workspace sophgo/tpuc_dev:v3.1 bash -c "
source /workspace/tpu-mlir/envsetup.sh
mkdir -p workspace && cd workspace
cp ../model.onnx .
model_transform.py --model_name MODEL --model_def model.onnx \
  --input_shapes '[[1,3,640,640]]' --mean 0.0,0.0,0.0 --scale 0.0039216,0.0039216,0.0039216 \
  --keep_aspect_ratio --pixel_format rgb --output_names 'OUTPUT_NAMES' \
  --test_input test.jpg --test_result top.npz --mlir model.mlir
run_calibration.py model.mlir --dataset ../dataset --input_num 100 -o calib_table
model_deploy.py --mlir model.mlir --quantize INT8 --quant_input \
  --processor cv181x --calibration_table calib_table \
  --test_input test.jpg --test_reference top.npz \
  --customization_format RGB_PACKED --fuse_preprocess --aligned_input \
  --model model_cv181x_int8.cvimodel
"
```

## Quick ONNX Export Commands

```python
from ultralytics import YOLO
YOLO('yolo11n.pt').export(format='onnx', imgsz=640, simplify=False, opset=12)     # Detection
YOLO('yolo11n-cls.pt').export(format='onnx', imgsz=224, simplify=False, opset=12)   # Classification
YOLO('yolo11n-pose.pt').export(format='onnx', imgsz=640, simplify=False, opset=12)  # Pose
YOLO('yolo11n-seg.pt').export(format='onnx', imgsz=640, simplify=False, opset=12)   # Segmentation
```

## Check ONNX Outputs

```python
import onnx
from onnx import shape_inference

model = onnx.load('model.onnx')
model = shape_inference.infer_shapes(model)

for output in model.graph.output:
    shape = [str(d.dim_value) if d.dim_value else '?'
             for d in output.type.tensor_type.shape.dim]
    print(f"{output.name}: [{', '.join(shape)}]")
```

## Hybrid Quantization (qtable)

For pose/segmentation, use qtable for better accuracy:

```bash
# Copy qtables from skill assets
cp assets/yolo11n_pose_qtable ./
cp assets/yolo11n_seg ./

# Use in deployment
model_deploy.py ... --quantize_table yolo11n_pose_qtable \
  --model yolo11n-pose_cv181x_mix.cvimodel
```

## Not Supported

| Model | Reason | Alternative |
|-------|--------|-------------|
| YOLO26n pose | Mod operation not supported | Use YOLO11n-pose |
| YOLO26n seg | Mod operation not supported | Use YOLO11n-seg |
| ONNX TopK/Argmax ops | Not supported in TPU-MLIR | Post-process on CPU instead |

## Troubleshooting

Error: `Op not support: Mod`
Cause: YOLO26 pose or segmentation uses the `Mod` operator, which is not supported here.
Solution: Use YOLO11 for pose or segmentation instead.

Error: `model_transform: command not found`
Cause: TPU-MLIR environment was not initialized.
Solution: Run `source /workspace/tpu-mlir/envsetup.sh` inside the container.

Error: Wrong output order or unusable detection results
Cause: Detection outputs are not provided in the expected alternating order.
Solution: For detection models, ensure outputs are alternating `Box, Class, Box, Class, Box, Class`.

Error: ION memory exceeds budget
Cause: INT8 output was dequantized to float32 or the chosen variant uses too much memory.
Solution: Add `--quant_output` when downstream logic only needs relative ordering.

Error: `pymlir` import fails
Cause: Host Python environment does not match the expected TPU-MLIR runtime.
Solution: Run the workflow inside the Docker container instead of on the host.

Error: `/workspace/tpu-mlir` is empty inside Docker
Cause: Local TPU-MLIR was not mounted into the container.
Solution: Mount it explicitly with `-v /path/to/tpu-mlir:/workspace/tpu-mlir`.

Error: CVIMODEL results are completely wrong
Cause: Input preprocessing is mismatched.
Solution: Check that `fuse_preprocess` is receiving uint8 RGB NHWC input, not float32 NCHW.

## Key Points

1. **YOLO26 pose/seg NOT supported** - Use YOLO11
2. **Detection needs 6 alternating outputs** - Box, Class, Box, Class, Box, Class
3. **BiSeNetv2 uses ImageNet preprocessing** - Different mean/scale from YOLO
4. **Use `--quant_output` for ION-constrained devices** - Saves ~28MB, int8 argmax is equivalent
5. **Mount local tpu-mlir into Docker** - Docker image has empty /workspace/tpu-mlir
6. **CVIMODEL with fuse_preprocess expects uint8 RGB NHWC** - Not float32 NCHW
7. **qtable for YOLO pose/seg** - Better accuracy with hybrid quantization
8. **100+ calibration images** for production
9. **Always validate** ONNX vs CVIMODEL pixel agreement after conversion
