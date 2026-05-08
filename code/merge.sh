# Merge LoRA adapter weights into base model
# Usage: bash code/merge.sh

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# TODO: set your checkpoint path and base model path
ADAPTERS="${PROJECT_ROOT}/output/sft/MMDD_HH/checkpoint-XXXX"
BASE_MODEL="Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR="${PROJECT_ROOT}/output/sft/MMDD_HH/qwen3vl-merged"

swift export \
    --adapters "${ADAPTERS}" \
    --merge_lora true \
    --model "${BASE_MODEL}" \
    --model_type qwen3_vl \
    --torch_dtype bfloat16 \
    --output_dir "${OUTPUT_DIR}"
