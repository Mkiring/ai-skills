#!/usr/bin/env python3
"""
ONNX vs CVIMODEL Conversion Validation Script
==============================================
Compares pixel-level agreement between ONNX (float32) and CVIMODEL (int8) inference.

Usage:
    python3 validate_conversion.py <onnx_model> <cvimodel> <test_images_dir> [options]

    # Run inside TPU-MLIR Docker:
    docker run --rm \
        -v /path/to/work:/work \
        -v /path/to/tpu-mlir:/workspace/tpu-mlir \
        sophgo/tpuc_dev:v3.1 python3 /work/validate_conversion.py \
        /work/model.onnx /work/model.cvimodel /work/test_images

Options:
    --task            Task type: detect, cls, semantic_seg (default: auto-detect)
    --input_h         Input height (default: from model)
    --input_w         Input width (default: from model)
    --output_name     ONNX output name (default: auto)
    --mean            Preprocessing mean (default: auto)
    --scale           Preprocessing scale (default: auto)
    --threshold       IoU threshold for PASS/FAIL (default: 0.85)
    --save_diff       Save diff images to output dir (default: True)
    --output_dir      Output directory for results (default: ./validation_results)

For semantic segmentation, outputs:
    - Per-image pixel agreement %
    - Per-class IoU
    - Side-by-side comparison images
    - Diff maps (gray=agree, red=disagree)
    - Summary CSV with all metrics
"""

import os
import sys
import csv
import argparse
import numpy as np
import cv2
from collections import defaultdict

# Cityscapes colors and labels
CS_COLORS = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
]
CS_LABELS = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]


def colorize(label, colors=None):
    if colors is None:
        colors = CS_COLORS
    h, w = label.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in enumerate(colors):
        if cid < len(colors):
            mask = label == cid
            color_img[mask] = color
    return color_img


def detect_task(onnx_path, cvimodel_path):
    """Auto-detect task type from model metadata."""
    import onnx
    model = onnx.load(onnx_path)
    output_shapes = []
    for out in model.graph.output:
        shape = []
        for d in out.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value else 0)
        output_shapes.append((out.name, shape))

    num_outputs = len(output_shapes)
    first_shape = output_shapes[0][1] if output_shapes else []

    if num_outputs == 1:
        dims = [d for d in first_shape if d > 0]
        if len(dims) == 3 and dims[0] > 2 and dims[1] >= 32 and dims[2] >= 32:
            return "semantic_seg", output_shapes
        elif len(dims) == 2 or (len(dims) == 1 and dims[0] > 2):
            return "cls", output_shapes
    elif num_outputs == 2:
        return "instance_seg", output_shapes
    elif num_outputs >= 6:
        return "detect", output_shapes

    return "unknown", output_shapes


def validate_semantic_seg(onnx_path, cvimodel_path, test_dir, args):
    """Validate semantic segmentation model."""
    import onnxruntime as ort
    import pyruntime_cvi as cvi

    # Setup ONNX
    sess = ort.InferenceSession(onnx_path)
    inp_name = sess.get_inputs()[0].name
    onnx_out_name = sess.get_outputs()[0].name
    onnx_input_shape = sess.get_inputs()[0].shape

    # Determine input size
    if args.input_h and args.input_w:
        model_h, model_w = args.input_h, args.input_w
    else:
        model_h = onnx_input_shape[2] if isinstance(onnx_input_shape[2], int) else 512
        model_w = onnx_input_shape[3] if isinstance(onnx_input_shape[3], int) else 1024

    # Determine preprocessing
    if args.mean and args.scale:
        mean = np.array([float(x) for x in args.mean.split(",")])
        scale = np.array([float(x) for x in args.scale.split(",")])
    else:
        mean = np.array([0.485, 0.456, 0.406])
        scale = np.array([0.229, 0.224, 0.225])

    print(f"  Input size: {model_h}x{model_w}")
    print(f"  Preprocessing: mean={mean.tolist()}, std={scale.tolist()}")

    # Setup CVIMODEL
    cvi_model = cvi.Model(cvimodel_path, output_all_tensors=True)
    print("  CVIMODEL inputs:")
    for inp in cvi_model.inputs:
        print(f"    {inp.name}: shape={inp.data.shape}, dtype={inp.data.dtype}")
    print("  CVIMODEL outputs:")
    for out in cvi_model.outputs:
        print(f"    {out.name}: shape={out.data.shape}, dtype={out.data.dtype}")

    # Find cvimodel output (prefer int8, fallback to f32)
    cvi_out_name = None
    for out in cvi_model.outputs:
        if "preds" in out.name and out.data.dtype == np.int8:
            cvi_out_name = out.name
            break
    if cvi_out_name is None:
        for out in cvi_model.outputs:
            if "f32" in out.name and out.data.dtype == np.float32:
                cvi_out_name = out.name
                break
    if cvi_out_name is None:
        cvi_out_name = cvi_model.outputs[-1].name
    print(f"  Using cvimodel output: {cvi_out_name}")

    # Find test images
    img_files = sorted(
        [os.path.join(test_dir, f) for f in os.listdir(test_dir)
         if f.lower().endswith(('.png', '.jpg', '.jpeg'))
         and "result" not in f.lower() and "diff" not in f.lower()]
    )
    print(f"  Test images: {len(img_files)}")

    if not img_files:
        print("  ERROR: No test images found!")
        return {}

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    class_iou_sum = defaultdict(float)
    class_count = 0

    for img_path in img_files:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n  --- {os.path.basename(img_path)} ---")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"    SKIP: cannot read")
            continue
        orig_h, orig_w = img_bgr.shape[:2]

        # ONNX inference
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (model_w, model_h))
        img_float = img_resized.astype(np.float32) / 255.0
        img_norm = (img_float - mean) / scale
        img_nchw = np.ascontiguousarray(np.transpose(img_norm, (2, 0, 1))[np.newaxis].astype(np.float32))
        onnx_out = sess.run([onnx_out_name], {inp_name: img_nchw})[0]
        label_onnx = np.argmax(onnx_out[0], axis=0).astype(np.uint8)

        # CVIMODEL inference
        img_rgb2 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized2 = cv2.resize(img_rgb2, (model_w, model_h))
        input_data = np.ascontiguousarray(img_resized2, dtype=np.uint8)
        cvi_model.inputs[0].data[:] = input_data.reshape(cvi_model.inputs[0].data.shape)
        cvi_model.forward()
        cvi_out = None
        for out in cvi_model.outputs:
            if out.name == cvi_out_name:
                cvi_out = out.data
                break
        pred = cvi_out[0]
        if pred.dtype == np.int8:
            pred = pred.astype(np.int32)
        label_cvi = np.argmax(pred, axis=0).astype(np.uint8)

        # Resize to original size
        label_onnx_orig = cv2.resize(label_onnx, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        label_cvi_orig = cv2.resize(label_cvi, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Pixel comparison
        total = orig_h * orig_w
        match = int(np.sum(label_onnx_orig == label_cvi_orig))
        diff = total - match
        agree_pct = match / total * 100

        print(f"    Agreement: {agree_pct:.2f}% ({match}/{total}), Diff: {diff} pixels")

        # Per-class IoU
        num_classes = max(int(label_onnx_orig.max()), int(label_cvi_orig.max())) + 1
        row = {"image": fname, "total_pixels": total, "match_pixels": match, "agree_pct": agree_pct}

        for c in range(min(num_classes, len(CS_LABELS))):
            onnx_mask = (label_onnx_orig == c)
            cvi_mask = (label_cvi_orig == c)
            onnx_cnt = int(np.sum(onnx_mask))
            cvi_cnt = int(np.sum(cvi_mask))
            onnx_pct = onnx_cnt / total * 100
            cvi_pct = cvi_cnt / total * 100

            if onnx_pct > 0.05 or cvi_pct > 0.05:
                iou_inter = int(np.sum(onnx_mask & cvi_mask))
                iou_union = int(np.sum(onnx_mask | cvi_mask))
                iou = iou_inter / iou_union if iou_union > 0 else 0.0
                label = CS_LABELS[c] if c < len(CS_LABELS) else f"class_{c}"
                print(f"    {label:15s}: ONNX {onnx_pct:5.1f}% ({onnx_cnt:7d})  CVI {cvi_pct:5.1f}% ({cvi_cnt:7d})  IoU={iou:.4f}")
                row[f"iou_{label}"] = iou
                class_iou_sum[label] += iou
                class_count += 1

        all_results.append(row)

        # Save visual comparison
        if args.save_diff:
            diff_map = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            diff_pixels = (label_onnx_orig != label_cvi_orig)
            diff_map[~diff_pixels] = [200, 200, 200]
            diff_map[diff_pixels] = [0, 0, 255]

            seg_onnx = colorize(label_onnx_orig)
            seg_cvi = colorize(label_cvi_orig)
            combined = np.hstack([seg_onnx, seg_cvi, diff_map])
            cv2.imwrite(f"{args.output_dir}/{fname}_compare.png", combined)
            cv2.imwrite(f"{args.output_dir}/{fname}_diff_map.png", diff_map)
            print(f"    Saved: {args.output_dir}/{fname}_compare.png")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    avg_agree = np.mean([r["agree_pct"] for r in all_results])
    min_agree = np.min([r["agree_pct"] for r in all_results])
    passed = avg_agree >= args.threshold * 100

    print(f"  Images tested:  {len(all_results)}")
    print(f"  Avg agreement:  {avg_agree:.2f}%")
    print(f"  Min agreement:  {min_agree:.2f}%")
    print(f"  Threshold:      {args.threshold * 100:.0f}%")
    print(f"  Result:         {'PASS' if passed else 'FAIL'}")

    if class_iou_sum:
        print(f"\n  Average per-class IoU:")
        for label, iou_sum in sorted(class_iou_sum.items()):
            avg_iou = iou_sum / (len(all_results))
            status = "OK" if avg_iou >= args.threshold else "LOW"
            print(f"    {label:15s}: {avg_iou:.4f}  [{status}]")

    # Save CSV
    csv_path = f"{args.output_dir}/validation_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n  CSV saved: {csv_path}")

    return {"passed": passed, "avg_agreement": avg_agree, "results": all_results}


def main():
    parser = argparse.ArgumentParser(description="Validate ONNX vs CVIMODEL conversion")
    parser.add_argument("onnx_model", help="Path to ONNX model")
    parser.add_argument("cvimodel", help="Path to CVIMODEL file")
    parser.add_argument("test_dir", help="Directory with test images")
    parser.add_argument("--task", choices=["semantic_seg", "cls", "detect", "auto"], default="auto")
    parser.add_argument("--input_h", type=int, default=0)
    parser.add_argument("--input_w", type=int, default=0)
    parser.add_argument("--output_name", default=None)
    parser.add_argument("--mean", default=None)
    parser.add_argument("--scale", default=None)
    parser.add_argument("--threshold", type=float, default=0.85, help="IoU threshold (default: 0.85)")
    parser.add_argument("--save_diff", action="store_true", default=True)
    parser.add_argument("--no-save-diff", dest="save_diff", action="store_false")
    parser.add_argument("--output_dir", default="./validation_results")
    args = parser.parse_args()

    if not os.path.isfile(args.onnx_model):
        print(f"ERROR: ONNX not found: {args.onnx_model}")
        sys.exit(1)
    if not os.path.isfile(args.cvimodel):
        print(f"ERROR: CVIMODEL not found: {args.cvimodel}")
        sys.exit(1)
    if not os.path.isdir(args.test_dir):
        print(f"ERROR: Test dir not found: {args.test_dir}")
        sys.exit(1)

    print("=" * 60)
    print("ONNX vs CVIMODEL Validation")
    print("=" * 60)
    print(f"  ONNX:     {args.onnx_model}")
    print(f"  CVIMODEL: {args.cvimodel}")
    print(f"  Test dir: {args.test_dir}")

    # Auto-detect task
    if args.task == "auto":
        task, _ = detect_task(args.onnx_model, args.cvimodel)
        print(f"  Detected task: {task}")
    else:
        task = args.task

    if task == "semantic_seg":
        result = validate_semantic_seg(args.onnx_model, args.cvimodel, args.test_dir, args)
        if result and not result["passed"]:
            print("\n  WARNING: Validation did not meet threshold!")
            sys.exit(1)
    else:
        print(f"  Task '{task}' validation not yet implemented")
        sys.exit(1)


if __name__ == "__main__":
    main()
