#!/bin/bash
# Evaluation script for NoCaps dataset using validation_nocaps.py (MeaCap mode)

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

EXP_NAME=$1
DEVICE=$2
OTHER_ARGS=$3
EPOCH=$4
WEIGHT_PATH=checkpoints/$EXP_NAME/coco_prefix-00${EPOCH}.pt
NOCAPS_OUT_PATH=checkpoints/$EXP_NAME

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/${EXP_NAME}_EVAL_MEACAP_NOCAPS
mkdir -p $LOG_FOLDER

NOCAPS_LOG_FILE="$LOG_FOLDER/NOCAPS_MEACAP_${TIME_START}.log"

python validation_nocaps.py \
--device cuda:$DEVICE \
--clip_model ViT-B/32 \
--language_model gpt2 \
--continuous_prompt_length 10 \
--clip_project_length 10 \
--top_k 3 \
--threshold 0.2 \
--using_image_features \
--name_of_datasets nocaps \
--path_of_val_datasets ./annotations/nocaps/nocaps_corpus.json \
--image_folder ./annotations/nocaps/ \
--weight_path=$WEIGHT_PATH \
--out_path=$NOCAPS_OUT_PATH \
--using_hard_prompt \
--soft_prompt_first \
--vl_model openai/clip-vit-base-patch32 \
--parser_checkpoint lizhuang144/flan-t5-base-VG-factual-sg \
--wte_model_path sentence-transformers/all-MiniLM-L6-v2 \
--memory_id coco \
--memory_caption_path data/memory/coco/memory_captions.json \
--memory_caption_num 5 \
$OTHER_ARGS \
|& tee -a  ${NOCAPS_LOG_FILE}

echo "==========================NOCAPS IN-DOMAIN (MeaCap)================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/indomain*_meacap.json |& tee -a  ${NOCAPS_LOG_FILE}
echo "==========================NOCAPS NEAR-DOMAIN (MeaCap)================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/neardomain*_meacap.json |& tee -a  ${NOCAPS_LOG_FILE}
echo "==========================NOCAPS OUT-DOMAIN (MeaCap)================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/outdomain*_meacap.json |& tee -a  ${NOCAPS_LOG_FILE}
echo "==========================NOCAPS ALL-DOMAIN (MeaCap)================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/overall*_meacap.json |& tee -a  ${NOCAPS_LOG_FILE}
