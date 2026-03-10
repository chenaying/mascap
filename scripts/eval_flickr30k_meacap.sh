#!/bin/bash
# Evaluation script for Flickr30k dataset using MeaCap Retrieve-then-Filter / EF module

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

# Arguments:
#   $1: EXP_NAME       (e.g., train_coco)
#   $2: DEVICE         (GPU id, e.g., 0)
#   $3: OTHER_ARGS     (extra args, e.g., '--use_entity_filtering --ef_filter_method threshold --ef_threshold 1')
#   $4: EPOCH          (e.g., 14 -> coco_prefix-0014.pt)

EXP_NAME=$1
DEVICE=$2
OTHER_ARGS=$3
EPOCH=$4

WEIGHT_PATH=checkpoints/$EXP_NAME/coco_prefix-00${EPOCH}.pt
FLICKR_OUT_PATH=checkpoints/$EXP_NAME

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/${EXP_NAME}_EVAL_MEACAP_FLICKR
mkdir -p $LOG_FOLDER

FLICKR_LOG_FILE="$LOG_FOLDER/FLICKR_MEACAP_${TIME_START}.log"

python validation_meacap.py \
--device cuda:$DEVICE \
--clip_model ViT-B/32 \
--language_model gpt2 \
--continuous_prompt_length 10 \
--clip_project_length 10 \
--using_image_features \
--name_of_datasets flickr30k \
--path_of_val_datasets ./annotations/flickr30k/test_captions.json \
--image_folder ./annotations/flickr30k/flickr30k-images/ \
--weight_path=$WEIGHT_PATH \
--out_path=$FLICKR_OUT_PATH \
--using_hard_prompt \
--soft_prompt_first \
--vl_model openai/clip-vit-base-patch32 \
--parser_checkpoint lizhuang144/flan-t5-base-VG-factual-sg \
--wte_model_path sentence-transformers/all-MiniLM-L6-v2 \
--memory_id coco \
--memory_caption_path data/memory/coco/memory_captions.json \
--memory_caption_num 5 \
$OTHER_ARGS \
|& tee -a  ${FLICKR_LOG_FILE}

echo "==========================FLICKR30K EVAL (MeaCap)================================"
python evaluation/cocoeval.py --result_file_path $FLICKR_OUT_PATH/flickr30k*_meacap.json |& tee -a  ${FLICKR_LOG_FILE}


