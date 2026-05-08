#!/bin/bash
# Inference script using swift infer + vLLM
# Usage: bash code/test.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/libs:${PYTHONPATH}"

# TODO: set your checkpoint path
CKPT="${PROJECT_ROOT}/output/rl_full/MMDD_HH/checkpoint-XXXX"
RESULT_PATH="${PROJECT_ROOT}/output/results.jsonl"

swift infer \
    --model "${CKPT}" \
    --infer_backend vllm \
    --model_type qwen3_vl \
    --torch_dtype bfloat16 \
    --val_dataset "${PROJECT_ROOT}/Datasets/Test-newpath-balanced.jsonl" \
    --result_path "${RESULT_PATH}" \
    --max_batch_size 6 \
    --max_new_tokens 2048 \
    --temperature 0 \
    --max_length 5500 \
    --vllm_max_model_len 5500 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_tensor_parallel_size 2 \
    --remove_unused_columns False \
    --system "Please answer this question based on the visual content. Provide your thinking process between the <think> and</think> tags, and then give your final answer between the <answer> and </answer> tags. At the end, you must output the final answer in the format: <answer><your_answer_here></answer>"
