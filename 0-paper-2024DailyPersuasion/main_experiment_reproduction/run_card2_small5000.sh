#!/usr/bin/env bash
set -Eeuo pipefail

# Run the reduced experiment on physical GPU 2. With CUDA_VISIBLE_DEVICES=2,
# the selected A100 is addressed as cuda:0 by each Python process.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$ROOT/data/small5000"
RUN_DATA="$ROOT/runs/small5000/data"
SAVE_ROOT="/data1/liujianjian/Persuader"
MODEL="/data1/liujianjian/model/Meta-Llama-3.1-8B-Instruct"
STAGE_ROOT="$SAVE_ROOT/.staging-small5000"
LOG_ROOT="$SAVE_ROOT/logs-small5000"

export CUDA_VISIBLE_DEVICES=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$SAVE_ROOT" "$STAGE_ROOT" "$LOG_ROOT"

move_model_pair() {
    local adapter_source="$1"
    local merged_source="$2"
    local adapter_target="$3"
    local merged_target="$4"
    if [[ -e "$adapter_target" || -e "$merged_target" ]]; then
        echo "Refusing to overwrite existing model directory: $adapter_target or $merged_target" >&2
        exit 1
    fi
    mv "$adapter_source" "$adapter_target"
    mv "$merged_source" "$merged_target"
}

run_dialogue() {
    local stage="$STAGE_ROOT/dialogue"
    local adapter="$stage/01_llama31_dialogue_sft"
    local merged="$stage/01_llama31_dialogue_sft_merged"
    mkdir -p "$stage"

    python "$ROOT/02_Llama31_Dialogue_SFT.py" \
        --profile small5000 \
        --model "$MODEL" \
        --output-dir "$stage" \
        --train-file "$DATA_ROOT/sft_dialogue.jsonl" \
        --finetuning-type lora \
        --epochs 2 \
        --cutoff-len 2048 \
        --batch-size 1 \
        --gradient-accumulation-steps 16 \
        --save-steps 100 \
        2>&1 | tee "$LOG_ROOT/dialogue_sft.log"

    test -s "$adapter/adapter_config.json"
    python "$ROOT/save_merged_models.py" \
        --profile small5000 \
        --output-dir "$stage" \
        --force \
        --strict \
        2>&1 | tee "$LOG_ROOT/dialogue_merge.log"
    test -s "$merged/config.json"
    move_model_pair \
        "$adapter" \
        "$merged" \
        "$SAVE_ROOT/llama3.1-8B-Instruct-dialogue" \
        "$SAVE_ROOT/llama3.1-8B-Instruct-dialogue-merged"
    echo "Dialogue-SFT adapter and merged model are ready under $SAVE_ROOT"
}

run_isr() {
    local stage="$STAGE_ROOT/ours"
    local adapter="$stage/02_llama31_isr_sft"
    local merged="$stage/02_llama31_isr_sft_merged"
    mkdir -p "$stage"

    python "$ROOT/03_Llama31_ISR_SFT.py" \
        --profile small5000 \
        --model "$MODEL" \
        --output-dir "$stage" \
        --train-file "$DATA_ROOT/sft_add_isr.jsonl" \
        --finetuning-type lora \
        --epochs 2 \
        --cutoff-len 2048 \
        --batch-size 1 \
        --gradient-accumulation-steps 16 \
        --save-steps 100 \
        2>&1 | tee "$LOG_ROOT/isr_sft.log"
    test -s "$adapter/adapter_config.json"

    python "$ROOT/save_merged_models.py" \
        --profile small5000 \
        --output-dir "$stage" \
        --force \
        --strict \
        2>&1 | tee "$LOG_ROOT/isr_merge.log"
    test -s "$merged/config.json"
    move_model_pair \
        "$adapter" \
        "$merged" \
        "$SAVE_ROOT/llama3.1-8B-Instruct-isr" \
        "$SAVE_ROOT/llama3.1-8B-Instruct-isr-merged"
    echo "ISR-SFT adapter and merged model are ready under $SAVE_ROOT"
}

run_ours() {
    local stage="$STAGE_ROOT/ours"
    local isr_adapter="$stage/02_llama31_isr_sft"
    local isr_merged="$stage/02_llama31_isr_sft_merged"
    local dpo_adapter="$stage/03_llama31_isr_sim_dpo"
    local dpo_merged="$stage/03_llama31_isr_sim_dpo_merged"
    mkdir -p "$stage"

    # Route 4 starts from a separately trained ISR adapter; Dialogue-SFT is not
    # a valid substitute for this prerequisite.
    python "$ROOT/03_Llama31_ISR_SFT.py" \
        --profile small5000 \
        --model "$MODEL" \
        --output-dir "$stage" \
        --train-file "$DATA_ROOT/sft_add_isr.jsonl" \
        --finetuning-type lora \
        --epochs 2 \
        --cutoff-len 2048 \
        --batch-size 1 \
        --gradient-accumulation-steps 16 \
        --save-steps 100 \
        2>&1 | tee "$LOG_ROOT/isr_sft.log"
    test -s "$isr_adapter/adapter_config.json"

    # Build all 1,000 simulation preference rows. The API persuadee and judge
    # use the configured ChatAnywhere-compatible endpoint; this stage is
    # resumable because the underlying writer appends one completed row at a time.
    python "$ROOT/04_Llama31_Ours_ISR_SimDPO_DPO.py" \
        --profile small5000 simulate \
        --persuader-url local://cuda:0 \
        --persuader-model "$isr_adapter" \
        --user-url https://api.chatanywhere.tech/v1 \
        --user-model gpt-4o-mini \
        --user-api-key-env CHATANYWHERE_API \
        --judge-url https://api.chatanywhere.tech/v1 \
        --judge-model gpt-4o-mini \
        --simulation-rounds 4 \
        --workers 4 \
        2>&1 | tee "$LOG_ROOT/simulation.log"
    test "$(wc -l < "$RUN_DATA/dpo_preferences.jsonl")" -eq 1000
    cp "$RUN_DATA/dpo_preferences.jsonl" "$DATA_ROOT/dpo_preferences.jsonl"
    if [[ -f "$RUN_DATA/simulation_preferences_summary.json" ]]; then
        cp "$RUN_DATA/simulation_preferences_summary.json" "$DATA_ROOT/dpo_preferences_summary.json"
    fi

    python "$ROOT/04_Llama31_Ours_ISR_SimDPO_DPO.py" \
        --profile small5000 dpo \
        --model "$MODEL" \
        --output-dir "$stage" \
        --finetuning-type lora \
        --epochs 1 \
        --cutoff-len 2048 \
        --batch-size 1 \
        --gradient-accumulation-steps 16 \
        --save-steps 100 \
        --isr-checkpoint "$isr_adapter" \
        2>&1 | tee "$LOG_ROOT/dpo.log"
    test -s "$dpo_adapter/adapter_config.json"

    python "$ROOT/save_merged_models.py" \
        --profile small5000 \
        --output-dir "$stage" \
        --force \
        --strict \
        2>&1 | tee "$LOG_ROOT/ours_merge.log"
    test -s "$isr_merged/config.json"
    test -s "$dpo_merged/config.json"

    move_model_pair \
        "$isr_adapter" \
        "$isr_merged" \
        "$SAVE_ROOT/llama3.1-8B-Instruct-isr" \
        "$SAVE_ROOT/llama3.1-8B-Instruct-isr-merged"
    move_model_pair \
        "$dpo_adapter" \
        "$dpo_merged" \
        "$SAVE_ROOT/llama3.1-8B-Instruct-ours" \
        "$SAVE_ROOT/llama3.1-8B-Instruct-ours-merged"
    echo "ISR-SFT and ISR-SimDPO-DPO adapter/merged models are ready under $SAVE_ROOT"
}

case "${1:-}" in
    dialogue)
        run_dialogue
        ;;
    isr)
        run_isr
        ;;
    ours)
        run_ours
        ;;
    *)
        echo "Usage: $0 {dialogue|isr|ours}" >&2
        exit 2
        ;;
esac
