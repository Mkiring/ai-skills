#!/bin/bash
# BiSeNetv2 ONNX to CVIMODEL Conversion Script
# ==============================================
# Semantic segmentation model for Cityscapes
# Target: Sophgo CV181x TPU (reCamera, SG200x)
#
# Produces 3 variants:
#   - INT8 (float32 output) → high ION usage
#   - BF16 (float32 output) → highest ION usage
#   - INT8 + --quant_output (int8 output) → LOWEST ION usage, RECOMMENDED

set -e

# Default values
MODEL_NAME="${MODEL_NAME:-bisenetv2_cityscapes}"
CHIP="${CHIP:-cv181x}"
INPUT_H="${INPUT_H:-512}"
INPUT_W="${INPUT_W:-1024}"
CALIBRATION_EPOCHS="${CALIBRATION_EPOCHS:-20}"
WORK_DIR="${WORK_DIR:-./work_dir}"
DOCKER_IMAGE="${DOCKER_IMAGE:-sophgo/tpuc_dev:v3.1}"
OUTPUT_NAME="${OUTPUT_NAME:-preds}"
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
    if ! docker images "$DOCKER_IMAGE" | grep -q "${DOCKER_IMAGE%%:*}"; then
        log_info "Pulling TPU-MLIR Docker image..."
        docker pull "$DOCKER_IMAGE"
    fi
    if [[ -z "$TPU_MLIR_DIR" ]]; then
        for d in /home/*/recamera/tpu-mlir /home/*/tpu-mlir /opt/tpu-mlir; do
            if [[ -d "$d" ]] && [[ -f "$d/install/python/tools/model_transform.py" ]]; then
                TPU_MLIR_DIR="$d"
                break
            fi
        done
    fi
    if [[ -z "$TPU_MLIR_DIR" ]] || [[ ! -f "$TPU_MLIR_DIR/install/python/tools/model_transform.py" ]]; then
        log_warn "tpu-mlir not found locally. Docker's /workspace/tpu-mlir may be empty."
        log_warn "Set TPU_MLIR_DIR=/path/to/tpu-mlir to mount it."
    fi
}

print_usage() {
    cat << EOF
BiSeNetv2 ONNX to CVIMODEL Conversion Script
=============================================

Usage:
    $0 <onnx_file> <dataset_dir>

Arguments:
    onnx_file       Path to ONNX model file
    dataset_dir     Path to calibration images directory

Options (environment variables):
    MODEL_NAME          Model name (default: bisenetv2_cityscapes)
    CHIP                Target chip (default: cv181x)
    INPUT_H             Input height (default: 512)
    INPUT_W             Input width (default: 1024)
    CALIBRATION_EPOCHS  Number of calibration images (default: 20)
    OUTPUT_NAME         ONNX output tensor name (default: preds)
    MEAN                ImageNet mean (default: 123.675,116.28,103.53)
    SCALE               ImageNet scale (default: 0.01712475,0.01750700,0.01742919)
    QUANT_OUTPUT        Generate int8 output variant (default: true)
    TPU_MLIR_DIR        Local tpu-mlir path for Docker mount
    WORK_DIR            Working directory (default: ./work_dir)
    DOCKER_IMAGE        TPU-MLIR Docker image (default: sophgo/tpuc_dev:v3.1)

Output:
    \${MODEL_NAME}_\${CHIP}_int8.cvimodel         (INT8, float32 output)
    \${MODEL_NAME}_\${CHIP}_bf16.cvimodel         (BF16, float32 output)
    \${MODEL_NAME}_\${CHIP}_int8_qout.cvimodel    (INT8, int8 output) ← RECOMMENDED

Examples:
    $0 bisenetv2.onnx ./dataset
    INPUT_H=256 INPUT_W=512 $0 bisenetv2.onnx ./dataset
    QUANT_OUTPUT=false $0 bisenetv2.onnx ./dataset

EOF
}

convert_model() {
    local onnx_file="$1"
    local dataset_dir="$2"

    if [[ ! -f "$onnx_file" ]]; then log_error "ONNX not found: $onnx_file"; exit 1; fi
    if [[ ! -d "$dataset_dir" ]]; then log_error "Dataset not found: $dataset_dir"; exit 1; fi

    log_info "Converting: $MODEL_NAME"
    log_info "Input: [1, 3, ${INPUT_H}, ${INPUT_W}]"
    log_info "Output: ${OUTPUT_NAME} [1, 19, ${INPUT_H}, ${INPUT_W}]"
    log_info "Chip: $CHIP"
    log_info "Dataset: $dataset_dir (${CALIBRATION_EPOCHS} calib images)"
    [[ "$QUANT_OUTPUT" == "true" ]] && log_info "Will generate --quant_output variant (LOW ION)"

    mkdir -p "$WORK_DIR/$MODEL_NAME/workspace"
    cp "$onnx_file" "$WORK_DIR/$MODEL_NAME/workspace/"
    local onnx_basename=$(basename "$onnx_file")

    # Build Docker mount args
    local docker_mounts="-v $(pwd)/$WORK_DIR:/work -v $(cd "$dataset_dir" && pwd):/work/dataset"
    if [[ -n "$TPU_MLIR_DIR" ]]; then
        docker_mounts="$docker_mounts -v $TPU_MLIR_DIR:/workspace/tpu-mlir"
        log_info "Mounting tpu-mlir: $TPU_MLIR_DIR"
    fi

    # Build deploy commands
    local deploy_int8_qout=""
    if [[ "$QUANT_OUTPUT" == "true" ]]; then
        deploy_int8_qout="
        echo '=== Step 3c: Deploy INT8 + quant_output (LOW ION) ==='
        model_deploy.py \\
            --mlir ${MODEL_NAME}.mlir \\
            --quantize INT8 \\
            --quant_input \\
            --quant_output \\
            --processor $CHIP \\
            --calibration_table ${MODEL_NAME}_calib_table \\
            --test_input /work/dataset/\$(ls /work/dataset/ | head -1) \\
            --test_reference ${MODEL_NAME}_top_outputs.npz \\
            --customization_format RGB_PACKED \\
            --fuse_preprocess \\
            --aligned_input \\
            --model ${MODEL_NAME}_${CHIP}_int8_qout.cvimodel
        "
    fi

    docker run --privileged --rm --name "${MODEL_NAME}_convert" \
        $docker_mounts \
        -w "/work/$MODEL_NAME/workspace" \
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

        python3 -c 'import pymlir; print(\"tpu-mlir OK:\", pymlir.__file__)' || { echo 'ERROR: pymlir import failed'; exit 1; }

        echo '=== Step 1: Model Transform ==='
        python3 model_transform.py \\
            --model_name $MODEL_NAME \\
            --model_def $onnx_basename \\
            --input_shapes '[[1,3,$INPUT_H,$INPUT_W]]' \\
            --mean '$MEAN' \\
            --scale '$SCALE' \\
            --pixel_format rgb \\
            --output_names '$OUTPUT_NAME' \\
            --test_input /work/dataset/\$(ls /work/dataset/ | head -1) \\
            --test_result ${MODEL_NAME}_top_outputs.npz \\
            --mlir ${MODEL_NAME}.mlir

        echo '=== Step 2: Calibration ==='
        python3 run_calibration.py ${MODEL_NAME}.mlir \\
            --dataset /work/dataset \\
            --input_num $CALIBRATION_EPOCHS \\
            -o ${MODEL_NAME}_calib_table

        echo '=== Step 3a: Deploy INT8 (float32 output) ==='
        python3 model_deploy.py \\
            --mlir ${MODEL_NAME}.mlir \\
            --quantize INT8 \\
            --quant_input \\
            --processor $CHIP \\
            --calibration_table ${MODEL_NAME}_calib_table \\
            --test_input /work/dataset/\$(ls /work/dataset/ | head -1) \\
            --test_reference ${MODEL_NAME}_top_outputs.npz \\
            --customization_format RGB_PACKED \\
            --fuse_preprocess \\
            --aligned_input \\
            --model ${MODEL_NAME}_${CHIP}_int8.cvimodel

        echo '=== Step 3b: Deploy BF16 ==='
        python3 model_deploy.py \\
            --mlir ${MODEL_NAME}.mlir \\
            --quantize BF16 \\
            --processor $CHIP \\
            --test_input /work/dataset/\$(ls /work/dataset/ | head -1) \\
            --test_reference ${MODEL_NAME}_top_outputs.npz \\
            --customization_format RGB_PACKED \\
            --fuse_preprocess \\
            --aligned_input \\
            --model ${MODEL_NAME}_${CHIP}_bf16.cvimodel
        $deploy_int8_qout

        echo '=== Results ==='
        ls -lh *.cvimodel
        echo ''
        echo '=== ION Memory Usage ==='
        for f in *.cvimodel; do
            ion=\$(python3 -c "
import subprocess, re
r = subprocess.run(['\$INSTALL_PATH/bin/cvimodel_info', '\$f'], capture_output=True, text=True)
m = re.search(r'ION_MEM:\s*([\d.]+)\s*MB', r.stdout + r.stderr)
print(m.group(1) if m else 'N/A')
" 2>/dev/null || echo 'N/A')
            echo \"  \$(basename \$f): ION = \${ion} MB\"
        done
        "

    # Copy results
    for cvimodel in "$WORK_DIR/$MODEL_NAME/workspace/"*.cvimodel; do
        if [[ -f "$cvimodel" ]]; then
            cp "$cvimodel" ./
            log_info "Output: $(basename "$cvimodel")"
        fi
    done

    log_info "Done!"
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
