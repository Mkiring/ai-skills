---
name: onnx-to-cvimodel
description: "Expert guide for converting ONNX models to CVIMODEL format for Sophgo CV181x TPU. Supports YOLO11/YOLO26 (detect, pose, seg, cls) and BiSeNetv2 (semantic segmentation). Includes tested conversion scripts, quantization tables (qtables), and complete workflow documentation. Use when user needs to convert ONNX models to CVIMODEL, set up TPU-MLIR conversion pipeline, configure output names and quantization, or troubleshoot conversion issues."
license: Complete terms in LICENSE.txt
---

# ONNX to CVIMODEL Conversion Guide

This skill provides scripts, configurations, and guidance for converting models from ONNX to CVIMODEL format for Sophgo CV181x TPU (reCamera, SG200x).

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
./convert_bisenetv2.sh <onnx> <dataset>      # BiSeNetv2 (INT8 + BF16)
```

### 4. Batch Conversion
```bash
./batch_convert_all.sh    # Convert all ONNX files in current directory
```

## Universal Conversion Script (Full Code)

```bash
#!/bin/bash
# Universal ONNX to CVIMODEL Conversion Script
set -e

MODEL_TYPE="${MODEL_TYPE:-yolo11}"
TASK="${TASK:-detect}"
MODEL_SIZE="${MODEL_SIZE:-n}"
CHIP="${CHIP:-cv181x}"
CALIBRATION_EPOCHS="${CALIBRATION_EPOCHS:-100}"
WORK_DIR="${WORK_DIR:-./work_dir}"

# Get output names based on model type and task
get_output_names() {
    case "$TASK" in
        detect)
            if [[ "$MODEL_TYPE" == "yolo26" ]]; then
                echo "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0,/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0,/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0,/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0,/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0,/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0"
            else
                echo "/model.23/cv2.0/cv2.0.2/Conv_output_0,/model.23/cv3.0/cv3.0.2/Conv_output_0,/model.23/cv2.1/cv2.1.2/Conv_output_0,/model.23/cv3.1/cv3.1.2/Conv_output_0,/model.23/cv2.2/cv2.2.2/Conv_output_0,/model.23/cv3.2/cv3.2.2/Conv_output_0"
            fi
            ;;
        seg) echo "output0,output1" ;;
        pose|cls) echo "output0" ;;
    esac
}

get_input_size() {
    [[ "$TASK" == "cls" ]] && echo "224" || echo "640"
}

check_model_support() {
    if [[ "$MODEL_TYPE" == "yolo26" ]] && [[ "$TASK" =~ ^(pose|seg)$ ]]; then
        echo "ERROR: YOLO26 $TASK is NOT supported (uses Mod operation)"
        echo "Use YOLO11 instead"
        exit 1
    fi
}

convert_model() {
    local onnx="$1" dataset="$2"
    local model_name="${MODEL_TYPE}${MODEL_SIZE}"
    [[ "$TASK" != "detect" ]] && model_name="${model_name}-${TASK}"
    local input_size=$(get_input_size)
    local output_names=$(get_output_names)

    check_model_support

    mkdir -p "$WORK_DIR/$model_name/workspace"
    cp "$onnx" "$dataset" "$WORK_DIR/$model_name/model/"

    docker run --rm -v "$PWD/$WORK_DIR:/work" -w "/work/$model_name" \
        sophgo/tpuc_dev:v3.1 bash -c "
        source /workspace/tpu-mlir/envsetup.sh
        cd workspace && cp ../model/* .

        model_transform.py --model_name $model_name --model_def *.onnx \
            --input_shapes '[[1,3,$input_size,$input_size]]' \
            --mean 0.0,0.0,0.0 --scale 0.0039216,0.0039216,0.0039216 \
            --keep_aspect_ratio --pixel_format rgb \
            --output_names '$output_names' \
            --test_input /work/model/*.jpg --test_result top.npz --mlir $model_name.mlir

        run_calibration.py $model_name.mlir --dataset /work/dataset --input_num $CALIBRATION_EPOCHS -o calib_table

        model_deploy.py --mlir $model_name.mlir --quantize INT8 --quant_input \
            --processor $CHIP --calibration_table calib_table \
            --test_input /work/model/*.jpg --test_reference top.npz \
            --customization_format RGB_PACKED --fuse_preprocess --aligned_input \
            --model ${model_name}_${CHIP}_int8.cvimodel
        "

    cp "$WORK_DIR/$model_name/workspace"/*.cvimodel ./
}

convert_model "$@"
```

## Python Export & Convert Script

```python
#!/usr/bin/env python3
"""
Export YOLO model to ONNX and convert to CVIMODEL in one go.
"""
import os
import subprocess
import argparse

def export_yolo_to_cvimodel(model_name="yolo11n", task="detect", imgsz=640):
    """Export YOLO model and convert to CVIMODEL."""

    # Export to ONNX
    from ultralytics import YOLO
    model = YOLO(f"{model_name}.pt")
    onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=False, opset=12)

    # Run conversion script
    task_map = {"detect": "detect", "pose": "pose", "seg": "seg", "cls": "cls"}
    script = f"convert_{model_name.split('n')[0]}_{task_map.get(task, 'detect')}.sh"

    if os.path.exists(script):
        subprocess.run([script, onnx_path, "dataset/"])
    else:
        print(f"Script {script} not found, using universal script...")
        subprocess.run([
            "bash", "convert_to_cvimodel.sh", onnx_path, "dataset/",
            f"MODEL_TYPE={model_name.removesuffix(model_name[-1])}",
            f"TASK={task}"
        ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n")
    parser.add_argument("--task", default="detect", choices=["detect", "pose", "seg", "cls"])
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    export_yolo_to_cvimodel(args.model, args.task, args.imgsz)
```

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

## Tested & Working Conversions

| Model | ONNX Size | CVIMODEL Size | Command |
|-------|-----------|---------------|---------|
| YOLO11n detect | 10.7 MB | 3.0 MB | `./convert_yolo11_detect.sh yolo11n.onnx dataset/` |
| YOLO11n pose | 11.3 MB | 3.6 MB | `./convert_yolo11_pose.sh yolo11n-pose.onnx dataset/` |
| YOLO11n seg | 11.7 MB | 4.0 MB | `./convert_yolo11_seg.sh yolo11n-seg.onnx dataset/` |
| YOLO11n cls | 10.8 MB | 3.0 MB | `./convert_yolo11_cls.sh yolo11n-cls.onnx dataset/` |
| YOLO26n detect | 9.4 MB | 2.9 MB | `./convert_yolo26_detect.sh yolo26n.onnx dataset/` |
| YOLO26n cls | 11.3 MB | 3.0 MB | `./convert_yolo26_cls.sh yolo26n-cls.onnx dataset/` |
| BiSeNetv2 seg | 13 MB | 6.0 MB (INT8) / 14 MB (BF16) | `./convert_bisenetv2.sh bisenetv2.onnx dataset/` |

## Not Supported

| Model | Reason | Alternative |
|-------|--------|-------------|
| YOLO26n pose | Mod operation not supported | Use YOLO11n-pose |
| YOLO26n seg | Mod operation not supported | Use YOLO11n-seg |

## BiSeNetv2 Conversion

BiSeNetv2 is a lightweight segmentation model. It uses different preprocessing from YOLO models (ImageNet mean/scale instead of 0-1 normalization).

### Key Differences from YOLO
- **Input shape**: Non-square `[1,3,512,1024]` (H, W configurable via env vars)
- **Preprocessing**: ImageNet mean/scale (`123.675,116.28,103.53` / `0.01712,0.01751,0.01743`)
- **No `--keep_aspect_ratio`**: Fixed resolution input
- **Dual output**: Produces both INT8 and BF16 CVIMODEL files

### Usage
```bash
# Basic conversion
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

## Docker Command Template (Manual)

```bash
docker run --privileged --rm -v $PWD:/workspace -w /workspace sophgo/tpuc_dev:v3.1 bash -c "
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

## Quick Onnx Export Commands

```python
# Detection
from ultralytics import YOLO
YOLO('yolo11n.pt').export(format='onnx', imgsz=640, simplify=False, opset=12)

# Classification
YOLO('yolo11n-cls.pt').export(format='onnx', imgsz=224, simplify=False, opset=12)

# Pose
YOLO('yolo11n-pose.pt').export(format='onnx', imgsz=640, simplify=False, opset=12)

# Segmentation
YOLO('yolo11n-seg.pt').export(format='onnx', imgsz=640, simplify=False, opset=12)
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

For pose/segmentation, use qtable for better accuracy. Qtables are included in this skill:

```bash
# Copy qtables from skill assets to working directory
cp ~/.claude/skills/skills/onnx-to-cvimodel/assets/yolo11n_pose_qtable ./
cp ~/.claude/skills/skills/onnx-to-cvimodel/assets/yolo11n_seg ./

# Use in deployment
model_deploy.py ... --quantize_table yolo11n_pose_qtable \
  --model yolo11n-pose_cv181x_mix.cvimodel
```

## Common Issues

### "Op not support: Mod"
**Problem**: YOLO26 pose/seg uses Mod operation
**Solution**: Use YOLO11 instead

### "model_transform: command not found"
**Solution**: `source /workspace/tpu-mlir/envsetup.sh && ./build.sh`

### Wrong output order
**Solution**: For detection, outputs MUST be alternating (Box, Class, Box, Class...)

## When to Use This Skill

- User wants to convert YOLO or BiSeNetv2 models to CVIMODEL
- User mentions reCamera, SG200x, CV181x
- User needs quick conversion scripts
- User asks about YOLO11 vs YOLO26 differences
- User asks about semantic segmentation on CV181x

## Key Points

1. **YOLO26 pose/seg NOT supported** - Use YOLO11
2. **Detection needs 6 alternating outputs** - Box, Class, Box, Class, Box, Class
3. **BiSeNetv2 uses ImageNet preprocessing** - Different mean/scale from YOLO
4. **Use scripts for conversion** - Don't manually run Docker commands
5. **qtable for YOLO pose/seg** - Better accuracy with hybrid quantization
6. **100+ calibration images** for production
