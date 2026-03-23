#!/bin/bash
# COCO test-set evaluation with MeaCap, but memory retrieval uses Flickr30k memory bank.
# Typical use: Flickr30k-trained weights (EXP_NAME=train_flickr30k) + aligned Flickr30k memory.

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

EXP_NAME=$1
DEVICE=$2
OTHER_ARGS=$3
EPOCH=$4
WEIGHT_PATH=checkpoints/$EXP_NAME/coco_prefix-00${EPOCH}.pt
COCO_OUT_PATH=checkpoints/$EXP_NAME

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/${EXP_NAME}_EVAL_MEACAP_COCO_FLICKR
mkdir -p $LOG_FOLDER

COCO_LOG_FILE="$LOG_FOLDER/COCO_MEACAP_FLICKR_${TIME_START}.log"

echo "Memory bank: flickr30k (COCO images, Flickr30k captions for retrieval)" | tee -a ${COCO_LOG_FILE}

python validation_meacap.py \
--device cuda:$DEVICE \
--clip_model ViT-B/32 \
--language_model gpt2 \
--continuous_prompt_length 10 \
--clip_project_length 10 \
--using_image_features \
--name_of_datasets coco \
--path_of_val_datasets ./annotations/coco/test_captions.json \
--image_folder ./annotations/coco/val2014/ \
--weight_path=$WEIGHT_PATH \
--out_path=$COCO_OUT_PATH \
--using_hard_prompt \
--soft_prompt_first \
--vl_model openai/clip-vit-base-patch32 \
--parser_checkpoint lizhuang144/flan-t5-base-VG-factual-sg \
--wte_model_path sentence-transformers/all-MiniLM-L6-v2 \
--memory_id flickr30k \
--memory_caption_path data/memory/flickr30k/memory_captions.json \
--memory_caption_num 5 \
$OTHER_ARGS \
|& tee -a  ${COCO_LOG_FILE}

echo "==========================COCO EVAL (MeaCap, Flickr30k memory)================================"
python evaluation/cocoeval.py --result_file_path $COCO_OUT_PATH/coco*_meacap.json |& tee -a  ${COCO_LOG_FILE}
