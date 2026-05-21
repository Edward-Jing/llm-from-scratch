#!/bin/bash -l
# Submit on MSBC with: sbatch scripts/msbc_train_h100.sbatch

#SBATCH --job-name=scratch-llm
#SBATCH --output=scratch-llm-%j.out
#SBATCH --error=scratch-llm-%j.err
#SBATCH --gpus=7g.80gb:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-cpu=4gb

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
conda activate pytorch
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

DATA_PATH="${DATA_PATH:-data/pretrain.jsonl}"
TOKENIZER_DIR="${TOKENIZER_DIR:-scratch_llm_runs/tokenizer}"
OUTPUT_DIR="${OUTPUT_DIR:-scratch_llm_runs/checkpoints}"
VOCAB_SIZE="${VOCAB_SIZE:-6144}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-1}"
DIM="${DIM:-512}"
N_LAYERS="${N_LAYERS:-8}"
N_HEADS="${N_HEADS:-8}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"

mkdir -p "$OUTPUT_DIR"

python3 scripts/check_cuda_env.py
python3 -m unittest discover -v tests

if [ ! -f "$TOKENIZER_DIR/tokenizer.json" ]; then
    python3 scripts/01_train_tokenizer.py \
        --data-path "$DATA_PATH" \
        --output-dir "$TOKENIZER_DIR" \
        --vocab-size "$VOCAB_SIZE"
fi

python3 scripts/02_pretrain.py \
    --data-path "$DATA_PATH" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --dtype bfloat16 \
    --batch-size "$BATCH_SIZE" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --epochs "$EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --dim "$DIM" \
    --n-layers "$N_LAYERS" \
    --n-heads "$N_HEADS" \
    --vocab-size "$VOCAB_SIZE"
