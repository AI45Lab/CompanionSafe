#!/bin/bash
# CompanionGuard one-shot training pipeline:
#   judged trajectories -> turn-localized boundary data -> group-level 95/5 split
#   -> LoRA SFT on Llama-Guard-3-8B (rank 16, 1 epoch) -> evaluation on the held-out test set.
#
# Required environment variables:
#   TRAJECTORIES  path to your judged-trajectory jsonl (schema in README.md)
#   MODEL_PATH    path to the Llama-Guard-3-8B weights directory
# Optional:
#   NGPU / TP     GPU count override; TP defaults to the largest power of two <= NGPU
#   EPOCHS        training epochs (default 1)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}."

TRAJECTORIES=${TRAJECTORIES:?set TRAJECTORIES to your judged-trajectory jsonl}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the Llama-Guard-3-8B directory}
EPOCHS=${EPOCHS:-1}

# Auto GPU detection: TP must be a power of two that divides the 32 attention heads.
NGPU=${NGPU:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}
[ "$NGPU" -ge 1 ] 2>/dev/null || NGPU=8
if [ -z "${TP:-}" ]; then
    TP=1
    for _c in 8 4 2; do
        [ "$NGPU" -ge "$_c" ] && TP=$_c && break
    done
fi
ACCUM=$((16 / (4 * NGPU) * 4))
[ ${ACCUM} -lt 1 ] && ACCUM=1

mkdir -p outputs logs

# 1) Turn-localized boundary construction from the judged trajectories.
python guard/build_boundary_data.py \
    --input "${TRAJECTORIES}" \
    --output data/boundary_selected.jsonl \
    --stats_output data/boundary_stats.json 2>&1 | tee logs/build.log

# 2) Group-level 95/5 split + SFT row assembly (answer-first contract).
python guard/split_train_test.py \
    --input data/boundary_selected.jsonl \
    --train_output data/train_selected.jsonl \
    --train_jsonl data/train_sft.jsonl \
    --test_output data/test.jsonl \
    --stats_output data/split_stats.json 2>&1 | tee logs/split.log

# 3) LoRA SFT (skip if a checkpoint already exists).
OUTPUT_DIR=outputs/companionguard-lg3-ep${EPOCHS}
if ls ${OUTPUT_DIR}/checkpoint-* >/dev/null 2>&1; then
    echo "skip training (checkpoint exists): ${OUTPUT_DIR}"
else
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPU-1))) \
    NPROC_PER_NODE=${NGPU} swift sft \
        --model "${MODEL_PATH}" \
        --model_type llama \
        --template llama3 \
        --dataset data/train_sft.jsonl \
        --torch_dtype bfloat16 \
        --tuner_type lora \
        --lora_rank 16 \
        --lora_alpha 16 \
        --lora_dropout 0.0 \
        --target_modules all-linear \
        --num_train_epochs ${EPOCHS} \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps ${ACCUM} \
        --learning_rate 2e-5 \
        --warmup_steps 50 \
        --weight_decay 0.01 \
        --max_length 8192 \
        --gradient_checkpointing True \
        --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
        --save_strategy steps \
        --save_steps 200 \
        --save_total_limit 2 \
        --logging_steps 10 \
        --eval_strategy no \
        --output_dir ${OUTPUT_DIR} \
        --add_version false \
        --dataloader_num_workers 4 \
        --dataset_num_proc 4 \
        2>&1 | tee logs/train.log
fi

# 4) Evaluate the fine-tuned checkpoint on the held-out test set.
CKPT=$(ls -td ${OUTPUT_DIR}/checkpoint-* | head -1)
echo "===== checkpoint: ${CKPT} ====="
LG3_PATH="${MODEL_PATH}" \
python guard/evaluate_companionguard.py --model llama-guard-3-8b --adapter_path "${CKPT}" --tp ${TP} \
    --source_dir data/test.jsonl --result_root results 2>&1 | tee logs/eval.log

echo "===== done: results/summaries.jsonl ====="
