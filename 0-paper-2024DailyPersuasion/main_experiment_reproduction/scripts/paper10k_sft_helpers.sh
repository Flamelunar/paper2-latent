#!/usr/bin/env bash

paper10k_run_sft() {
    local python_bin="$1"
    local model="$2"
    local output_root="$3"
    local log_file="$4"
    local script_name="$5"
    local output_name="$6"
    local output_dir="$output_root/$output_name"
    local checkpoint=""
    local -a resume_args=()

    if [[ -f "$output_dir/adapter_config.json" && -f "$output_dir/train_metrics.json" ]]; then
        printf 'Skipping completed SFT stage: %s\n' "$output_dir"
        return
    fi

    if [[ -d "$output_dir" ]]; then
        while IFS= read -r candidate; do
            if [[ -f "$candidate/trainer_state.json" && -f "$candidate/optimizer.pt" && -f "$candidate/scheduler.pt" ]]; then
                checkpoint="$candidate"
                break
            fi
        done < <(find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -Vr)
        if [[ -n "$checkpoint" ]]; then
            resume_args=(--resume-from-checkpoint "$checkpoint")
            printf 'Resuming SFT stage from %s\n' "$checkpoint"
        elif [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
            printf 'Output is nonempty but has no resumable Trainer checkpoint: %s\n' "$output_dir" >&2
            return 1
        fi
    fi

    "$python_bin" "$ROOT/$script_name" --profile paper10k \
        --model "$model" --output-dir "$output_root" --epochs 1 --finetuning-type lora \
        --cutoff-len 2048 --batch-size 1 --gradient-accumulation-steps 16 --save-steps 100 \
        "${resume_args[@]}" 2>&1 | tee -a "$log_file"
}
