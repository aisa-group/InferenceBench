#!/bin/bash
#
# Model registry for InferenceBench experiments.
# Source this file to get MODEL_IDS and MODEL_MAX_LEN associative arrays.
#
# Usage:
#   source src/commit_utils/model_registry.sh
#   echo "${MODEL_IDS[qwen3-8b]}"       # Qwen/Qwen3-8B
#   echo "${MODEL_MAX_LEN[mistral-7b]}" # 32768
#

declare -A MODEL_IDS=(
    [qwen3-8b]="Qwen/Qwen3-8B"
    [mistral-7b]="mistralai/Mistral-7B-Instruct-v0.3"
    [deepseek-v2-lite]="deepseek-ai/DeepSeek-V2-Lite"
)

declare -A MODEL_MAX_LEN=(
    [qwen3-8b]="32768"
    [mistral-7b]="32768"
    [deepseek-v2-lite]="32768"
)

# Ordered list of model keys for iteration
MODEL_KEYS=("qwen3-8b" "mistral-7b" "deepseek-v2-lite")
