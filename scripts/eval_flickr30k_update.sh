#!/bin/bash
# Unified Flickr30k evaluation script:
# - In-domain:  train_flickr30k -> use memory/flickr30k
# - Cross-domain: train_coco -> use memory/coco

set -e

SHELL_FOLDER=$(cd "$(dirname "$0")"; pwd)
cd "$SHELL_FOLDER/.."

EXP_NAME=$1
DEVICE=$2
OTHER_ARGS=$3
EPOCH=$4

WEIGHT_PATH="checkpoints/$EXP_NAME/coco_prefix-00${EPOCH}.pt"
FLICKR_OUT_PATH="checkpoints/$EXP_NAME"

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER="logs/${EXP_NAME}_EVAL_UPDATE_FLICKR"
mkdir -p "$LOG_FOLDER"
FLICKR_LOG_FILE="$LOG_FOLDER/FLICKR_UPDATE_${TIME_START}.log"

# Memory bank policy:
# - train_flickr30k => flickr30k memory (in-domain)
# - otherwise       => coco memory (cross-domain baseline)
if [ "$EXP_NAME" = "train_flickr30k" ]; then
  MEMORY_ID="flickr30k"
else
  MEMORY_ID="coco"
fi

echo "Using memory bank: $MEMORY_ID" | tee -a "$FLICKR_LOG_FILE"

python validation_flickr30k.py \
--device "cuda:$DEVICE" \
--clip_model "ViT-B/32" \
--language_model "gpt2" \
--continuous_prompt_length 10 \
--clip_project_length 10 \
--using_image_features \
--name_of_datasets "flickr30k" \
--path_of_val_datasets "./annotations/flickr30k/test_captions.json" \
--image_folder "./annotations/flickr30k/flickr30k-images/" \
--weight_path "$WEIGHT_PATH" \
--out_path "$FLICKR_OUT_PATH" \
--using_hard_prompt \
--soft_prompt_first \
--vl_model "openai/clip-vit-base-patch32" \
--parser_checkpoint "lizhuang144/flan-t5-base-VG-factual-sg" \
--wte_model_path "sentence-transformers/all-MiniLM-L6-v2" \
--memory_id "$MEMORY_ID" \
--memory_caption_num 5 \
$OTHER_ARGS \
|& tee -a "$FLICKR_LOG_FILE"

echo "==========================FLICKR30K EVAL (UPDATE)================================" | tee -a "$FLICKR_LOG_FILE"
RESULT_FILE="$FLICKR_OUT_PATH/flickr30k_generated_captions_update.json"
if [ ! -f "$RESULT_FILE" ]; then
  echo "Result file not found: $RESULT_FILE" | tee -a "$FLICKR_LOG_FILE"
  exit 1
fi
python evaluation/cocoeval.py --result_file_path "$RESULT_FILE" |& tee -a "$FLICKR_LOG_FILE"
