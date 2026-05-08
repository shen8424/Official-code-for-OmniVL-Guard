#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/libs:${PYTHONPATH}"

MODE=${1:-rl_full}  # rl_full or rl_lora

deepspeed --include localhost:0,1,2,3 code/strench_train/${MODE}.py
