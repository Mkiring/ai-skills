#!/bin/bash
# PP-LiteSeg ONNX to CVIMODEL Conversion Script
# ==============================================
# PaddlePaddle lightweight semantic segmentation model
# Target: Sophgo CV181x TPU (reCamera, SG200x)
#
# Produces 2 variants:
#   - INT8 (float32 output) → high ION usage
#   - INT8 + --quant_output (int8 output) → LOWEST ION usage, RECOMMENDED
#
# PP-LiteSeg specific handling:
#   1. Removes ArgMax + Cast nodes (output pre-argmax logits)
#   2. Fixes AveragePool count_include_pad: 0 → 1 (cv181x requirement)
#   3. Pre-simplifies ONNX with onnxsim (avoids Docker onnxsim Squeeze bug)
#   4. Uses ImageNet normalization for fuse_preprocess

set -e

# Default values
MODEL_NAME="${MODEL_NAME:-pp_liteseg}"
CHIP="${CHIP:-cv181x}"
INPUT_H="${INPUT_H:-512}"
INPUT_W="${INPUT_W:-1024}"
CALIBRATION_EPOCHS="${CALIBRATION_EPOCHS:-20}"
WORK_DIR="${WORK_DIR:-./work_dir}"
DOCKER_IMAGE="${DOCKER_IMAGE:-sophgo/tpuc_dev:v3.1}"
MEAN="${MEAN:-123.675,116.28,103.53}"
SCALE="${SCALE:-0.01712475,0.01750700,0.01742919}"
QUANT_OUTPUT="${QUANT_OUTPUT:-true}"
TPU_MLIR_DIR="${TPU_MLIR_DIR:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_dependencies() {
    log_info "Checking dependencies..."
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker first."
        exit 1
    fi
    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found. Required for ONNX graph surgery."
        exit 1
    fi
    python3 -c "import onnx, onnxsim" 2>/dev/null || {
        log_error "Missing Python packages. Install: pip install onnx onnxsim"
        exit 1
    }
    if [[ -z "$TPU_MLIR_DIR" ]]; then
        for d in /home/*/recamera/tpu-mlir /home/*/tpu-mlir /opt/tpu-mlir; do
            if [[ -d "$d" ]] && [[ -f "$d/install/python/tools/model_transform.py" ]]; then
                TPU_MLIR_DIR="$d"
                break
            fi
        done
    fi
    if [[ -z "$TPU_MLIR_DIR" ]] || [[ ! -f "$TPU_MLIR_DIR/install/python/tools/model_transform.py" ]]; then
        log_warn "tpu-mlir not found locally. Set TPU_MLIR_DIR=/path/to/tpu-mlir"
    fi
}

print_usage() {
    cat << EOF
PP-LiteSeg ONNX to CVIMODEL Conversion Script
=============================================

PP-LiteSeg requires ONNX graph surgery before conversion:
  1. Remove ArgMax + Cast (keep pre-argmax logits)
  2. Fix AveragePool count_include_pad (0 → 1)
  3. Pre-simplify with onnxsim (avoid Docker onnxsim bug)

Usage:
    $0 <onnx_file> <dataset_dir>

Arguments:
    onnx_file       Path to PP-LiteSeg ONNX model
    dataset_dir     Path to calibration images directory

Options (environment variables):
    MODEL_NAME          Model name (default: pp_liteseg)
    CHIP                Target chip (default: cv181x)
    INPUT_H             Input height (default: 512)
    INPUT_W             Input width (default: 1024)
    CALIBRATION_EPOCHS  Calibration images (default: 20)
    MEAN                ImageNet mean (default: 123.675,116.28,103.53)
    SCALE               ImageNet scale (default: 0.01712475,0.01750700,0.01742919)
    QUANT_OUTPUT        Generate int8 output variant (default: true)
    TPU_MLIR_DIR        Local tpu-mlir path for Docker mount
    WORK_DIR            Working directory (default: ./work_dir)
    DOCKER_IMAGE        TPU-MLIR Docker image (default: sophgo/tpuc_dev:v3.1)

Output:
    \${MODEL_NAME}_int8.cvimodel         (INT8, float32 output)
    \${MODEL_NAME}_int8_qout.cvimodel    (INT8, int8 output) ← RECOMMENDED

Examples:
    $0 pp_liteseg.onnx ./dataset
    INPUT_H=256 INPUT_W=512 $0 pp_liteseg.onnx ./dataset

EOF
}

# ONNX graph surgery: remove ArgMax+Cast, fix AveragePool, simplify
prepare_onnx() {
    local input_onnx="$1"
    local output_onnx="$2"

    log_info "ONNX graph surgery: $input_onnx → $output_onnx"

    python3 << PYEOF
import onnx
import onnxsim
import numpy as np
from onnx import TensorProto, helper, numpy_helper

model = onnx.load("$input_onnx")

# Step 1: Find the last Resize (bilinear_interp) node and its output name
# PP-LiteSeg architecture: ... → Conv → Resize → ArgMax → Cast
graph = model.graph
resize_output = None
for i, node in enumerate(graph.node):
    if node.op_type == 'Resize':
        resize_output = node.output[0]

if not resize_output:
    raise RuntimeError("No Resize node found in model. Is this a PP-LiteSeg model?")

print(f"  Found Resize output: {resize_output}")

# Step 2: Add Resize output as a new graph output (pre-argmax logits [1, C, H, W])
new_out = helper.make_tensor_value_info(resize_output, TensorProto.FLOAT, [1, 19, $INPUT_H, $INPUT_W])
graph.output.insert(0, new_out)
print(f"  Added output: {resize_output} [1, 19, $INPUT_H, $INPUT_W]")

# Step 3: Pre-simplify ONNX with local onnxsim
# IMPORTANT: Do this BEFORE Docker. Docker's onnxsim introduces broken Squeeze ops.
print("  Running onnxsim (local)...")
model_sim, check = onnxsim.simplify(model, test_input_shapes={'x': [1, 3, $INPUT_H, $INPUT_W]})
if check:
    print("  onnxsim check passed")
    model = model_sim
else:
    print("  WARNING: onnxsim check failed, using unsimplified model")

# Step 4: Fix AveragePool count_include_pad (cv181x requires =1)
avgpool_count = 0
for node in model.graph.node:
    if node.op_type == 'AveragePool':
        for attr in node.attribute:
            if attr.name == 'count_include_pad' and attr.i == 0:
                attr.i = 1
                avgpool_count += 1
                print(f"  Fixed AveragePool: count_include_pad 0 → 1")

if avgpool_count == 0:
    print("  No AveragePool patches needed")

onnx.save(model, "$output_onnx")
print(f"  Saved: $output_onnx")

# Verify: run both models and check agreement
import onnxruntime as ort
import cv2

sess_orig = ort.InferenceSession("$input_onnx")
sess_new = ort.InferenceSession("$output_onnx")

inp_name = sess_orig.get_inputs()[0].name
# Create dummy input
dummy = np.random.randn(1, 3, $INPUT_H, $INPUT_W).astype(np.float32)

orig_out = sess_orig.run(None, {inp_name: dummy})
new_out_0 = sess_new.run([sess_new.get_outputs()[0].name], {inp_name: dummy})[0]
new_out_1 = sess_new.run([sess_new.get_outputs()[1].name], {inp_name: dummy})[0]

# Original output is argmaxed label: [1, H, W] int32
# New first output is pre-argmax: [1, 19, H, W] float32
label_orig = orig_out[0].astype(np.int32)
label_new = np.argmax(new_out_0[0], axis=0).astype(np.int32)

agree = np.sum(label_orig == label_new) / label_orig.size * 100
print(f"  Verification: argmax agreement = {agree:.2f}%")
if agree < 99.0:
    print("  WARNING: Low agreement! Check graph surgery.")
else:
    print("  Graph surgery verified OK")
PYEOF
}

convert_model() {
    local onnx_file="$1"
    local dataset_dir="$2"

    if [[ ! -f "$onnx_file" ]]; then log_error "ONNX not found: $onnx_file"; exit 1; fi
    if [[ ! -d "$dataset_dir" ]]; then log_error "Dataset not found: $dataset_dir"; exit 1; fi

    local onnx_basename=$(basename "$onnx_file")
    local onnx_name="${onnx_basename%.*}"
    local workspace="$WORK_DIR/$MODEL_NAME"
    mkdir -p "$workspace"

    log_info "=== PP-LiteSeg Conversion ==="
    log_info "Input: $onnx_file"
    log_info "Model: $MODEL_NAME"
    log_info "Shape: [1, 3, ${INPUT_H}, ${INPUT_W}]"
    log_info "Chip: $CHIP"

    # Step 1: ONNX graph surgery (local, requires onnx + onnxsim)
    local prepared_onnx="$workspace/${MODEL_NAME}_prepared.onnx"
    prepare_onnx "$onnx_file" "$prepared_onnx"

    # Step 2: Determine output name from prepared ONNX
    local output_name=$(python3 -c "
import onnx
m = onnx.load('$prepared_onnx')
print(m.graph.output[0].name)
")
    log_info "Output name: $output_name"

    # Step 3: Build Docker mount args
    local docker_mounts="-v $(cd "$workspace" && pwd):/work"
    if [[ -d "$dataset_dir" ]]; then
        docker_mounts="$docker_mounts -v $(cd "$dataset_dir" && pwd):/work/dataset"
    fi
    if [[ -n "$TPU_MLIR_DIR" ]]; then
        docker_mounts="$docker_mounts -v $TPU_MLIR_DIR:/workspace/tpu-mlir"
        log_info "Mounting tpu-mlir: $TPU_MLIR_DIR"
    fi

    # Step 4: Run model_transform + calibration + deploy in Docker
    local prepared_basename=$(basename "$prepared_onnx")

    local deploy_qout=""
    if [[ "$QUANT_OUTPUT" == "true" ]]; then
        deploy_qout="
        echo '=== Deploy INT8 + quant_output (LOW ION) ==='
        model_deploy.py \\
            --mlir ${MODEL_NAME}.mlir \\
            --quantize INT8 \\
            --calibration_table ${MODEL_NAME}_calib_table \\
            --chip $CHIP \\
            --model ${MODEL_NAME}_int8_qout.cvimodel \\
            --fuse_preprocess \\
            --customization_format RGB_PACKED \\
            --quant_output
        "
    fi

    docker run --privileged --rm --name "${MODEL_NAME}_convert" \
        $docker_mounts \
        -w /work \
        "$DOCKER_IMAGE" bash -c "
        if [[ -f /workspace/tpu-mlir/envsetup.sh ]]; then
            source /workspace/tpu-mlir/envsetup.sh
        else
            export PROJECT_ROOT=/workspace/tpu-mlir
            export BUILD_PATH=\$PROJECT_ROOT/build
            export INSTALL_PATH=\$PROJECT_ROOT/install
            export TPUC_ROOT=\$INSTALL_PATH
            export PYTHONPATH=\$INSTALL_PATH/python:\$PYTHONPATH
            export PATH=\$INSTALL_PATH/bin:\$INSTALL_PATH/python/tools:\$PATH
            export LD_LIBRARY_PATH=\$INSTALL_PATH/lib:\$LD_LIBRARY_PATH
        fi

        # Check tpu-mlir
        which model_transform.py &>/dev/null || { echo 'ERROR: model_transform.py not found'; exit 1; }

        echo '=== Model Transform ==='
        model_transform.py \\
            --model_name $MODEL_NAME \\
            --model_def $prepared_basename \\
            --input_shapes [[1,3,$INPUT_H,$INPUT_W]] \\
            --output_names $output_name \\
            --pixel_format rgb \\
            --mean '$MEAN' \\
            --scale '$SCALE' \\
            --test_input /work/dataset/\$(ls /work/dataset/ | head -1) \\
            --test_result ${MODEL_NAME}_top.npz \\
            --mlir ${MODEL_NAME}.mlir

        echo '=== Calibration ==='
        run_calibration.py ${MODEL_NAME}.mlir \\
            --dataset /work/dataset \\
            --input_num $CALIBRATION_EPOCHS \\
            -o ${MODEL_NAME}_calib_table

        echo '=== Deploy INT8 (float32 output) ==='
        model_deploy.py \\
            --mlir ${MODEL_NAME}.mlir \\
            --quantize INT8 \\
            --calibration_table ${MODEL_NAME}_calib_table \\
            --chip $CHIP \\
            --model ${MODEL_NAME}_int8.cvimodel \\
            --fuse_preprocess \\
            --customization_format RGB_PACKED

        $deploy_qout

        echo '=== Results ==='
        ls -lh *.cvimodel
        "

    # Copy results to current directory
    for cvimodel in "$workspace/"*.cvimodel; do
        if [[ -f "$cvimodel" ]]; then
            cp "$cvimodel" ./
            log_info "Output: $(basename "$cvimodel")"
        fi
    done

    log_info "Done! Use ${MODEL_NAME}_int8_qout.cvimodel for deployment."
}

main() {
    if [[ $# -lt 2 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
        print_usage
        exit 0
    fi
    check_dependencies
    convert_model "$1" "$2"
}

main "$@"
