#!/bin/bash
# BiSeNetv2 ONNX to CVIMODEL Conversion Script
# ==============================================
# Semantic segmentation model for Cityscapes
# Target: Sophgo CV181x TPU (reCamera, SG200x)

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

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
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
}

# Print usage
print_usage() {
    cat << EOF
BiSeNetv2 ONNX to CVIMODEL Conversion Script
=============================================

Usage:
    $0 <onnx_file> <dataset_dir> [options]

Arguments:
    onnx_file       Path to ONNX model file
    dataset_dir     Path to calibration images directory

Options (environment variables):
    MODEL_NAME          Model name (default: bisenetv2_cityscapes)
    CHIP                Target chip (default: cv181x)
    INPUT_H             Input height (default: 512)
    INPUT_W             Input width (default: 1024)
    CALIBRATION_EPOCHS  Number of calibration images (default: 20)
    OUTPUT_NAME         ONNX output name (default: preds)
    MEAN                Mean values (default: 123.675,116.28,103.53)
    SCALE               Scale values (default: 0.01712475,0.01750700,0.01742919)
    WORK_DIR            Working directory (default: ./work_dir)
    DOCKER_IMAGE        TPU-MLIR Docker image (default: sophgo/tpuc_dev:v3.1)

Examples:
    # Basic conversion
    $0 bisenetv2.onnx ./dataset

    # Custom resolution
    INPUT_H=256 INPUT_W=512 $0 bisenetv2.onnx ./dataset

    # Custom model name
    MODEL_NAME=bisenetv2_custom $0 bisenetv2.onnx ./dataset

Output:
    \${MODEL_NAME}_\${CHIP}_int8.cvimodel   (INT8 quantized)
    \${MODEL_NAME}_\${CHIP}_bf16.cvimodel   (BF16 quantized)

EOF
}

# Main conversion function
convert_model() {
    local onnx_file="$1"
    local dataset_dir="$2"

    if [[ ! -f "$onnx_file" ]]; then
        log_error "ONNX file not found: $onnx_file"
        exit 1
    fi

    if [[ ! -d "$dataset_dir" ]]; then
        log_error "Dataset directory not found: $dataset_dir"
        exit 1
    fi

    log_info "Converting: $MODEL_NAME"
    log_info "Input size: [1, 3, ${INPUT_H}, ${INPUT_W}]"
    log_info "Output name: $OUTPUT_NAME"
    log_info "Chip: $CHIP"
    log_info "Dataset: $dataset_dir"
    log_info "Calibration epochs: $CALIBRATION_EPOCHS"

    # Create workspace
    mkdir -p "$WORK_DIR/$MODEL_NAME/workspace"
    cp "$onnx_file" "$WORK_DIR/$MODEL_NAME/workspace/"
    local onnx_basename=$(basename "$onnx_file")

    # Run conversion in Docker
    log_info "Starting Docker conversion..."

    docker run --privileged --rm --name "${MODEL_NAME}_convert" \
        -v "$(pwd)/$WORK_DIR:/work" \
        -v "$(cd "$dataset_dir" && pwd):/work/dataset" \
        -w "/work/$MODEL_NAME/workspace" \
        "$DOCKER_IMAGE" bash -c "
        source /workspace/tpu-mlir/envsetup.sh

        echo '=== Step 1: Model Transform ==='
        model_transform.py \\
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
        run_calibration.py ${MODEL_NAME}.mlir \\
            --dataset /work/dataset \\
            --input_num $CALIBRATION_EPOCHS \\
            -o ${MODEL_NAME}_calib_table

        echo '=== Step 3a: Deploy INT8 ==='
        model_deploy.py \\
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
        model_deploy.py \\
            --mlir ${MODEL_NAME}.mlir \\
            --quantize BF16 \\
            --processor $CHIP \\
            --test_input /work/dataset/\$(ls /work/dataset/ | head -1) \\
            --test_reference ${MODEL_NAME}_top_outputs.npz \\
            --customization_format RGB_PACKED \\
            --fuse_preprocess \\
            --aligned_input \\
            --model ${MODEL_NAME}_${CHIP}_bf16.cvimodel

        echo '=== Results ==='
        ls -lh *.cvimodel
        "

    # Copy result to current directory
    for cvimodel in "$WORK_DIR/$MODEL_NAME/workspace/"*.cvimodel; do
        if [[ -f "$cvimodel" ]]; then
            cp "$cvimodel" ./
            log_info "CVIMODEL created: $(basename "$cvimodel")"
        fi
    done
}

# Main
main() {
    if [[ $# -lt 2 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
        print_usage
        exit 0
    fi

    local onnx_file="$1"
    local dataset_dir="$2"

    check_dependencies
    convert_model "$onnx_file" "$dataset_dir"
}

main "$@"
